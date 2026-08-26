"""world_ft box-flip policy inference worker (~10 Hz).

Runs the trained ACT policy and publishes the predicted commanded-q chunk for the high-rate eval
controller to track. Handles BOTH model types automatically from the checkpoint's info.yaml:
  * state model      (image_only=False): obs = camera + follower state (+ tau_interaction)
  * image-only model (image_only=True):  obs = camera only (no follower state / torque read)

Color note: the RealSense worker publishes `camera_head:image` in BGR; training mp4s are stored as
RGB (the recorder does BGR->RGB), so we swap BGR->RGB here to match training.

Run in the training env (torch/CUDA), e.g.:
    python world_ft/learning/eval_inference_worker.py --model-path <ckpt>/model.pth
"""
import os
import sys
import time
import argparse

import cv2
import yaml
import torch
import numpy as np

# make imports work regardless of launch dir: script dir (for dataset/model) + repo root (for nova)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
from dataset import center_crop_resize
from ckpt_select import resolve_ckpt, is_spec
from nova.redis_client import RedisClient

# arm name -> tau_interaction suffix
_TAU_SUFFIX = {"left": "L", "right": "R"}


def load_model(model_path):
    info_path = model_path.replace("model.pth", "info.yaml")
    with open(info_path, "r") as f:
        config = yaml.safe_load(f)["config"]
    image_only = config.get("image_only", False)
    # DETR's build_ACT_model_and_optimizer calls argparse.parse_args() on sys.argv, which would
    # choke on THIS script's CLI flags (--model-path, ...). Hide them during the model build.
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        if image_only:
            from model_image_only import ACT_ImageOnly
            act_model, _ = ACT_ImageOnly.build_model_and_optimizer(
                n_obs_steps=config["obs_horizon"],
                action_chunk_size=config["action_horizon"],
                seed=config["seed"],
                arm_names=config["arms"],
            )
        else:
            from model import ACT_Abs
            act_model, _ = ACT_Abs.build_model_and_optimizer(
                n_obs_steps=config["obs_horizon"],
                action_chunk_size=config["action_horizon"],
                seed=config["seed"],
                arm_names=config["arms"],
                use_angle=config.get("use_angle", True),
                use_torque=config.get("use_torque", True),
                extra_obs_dim=2 * config.get("ink_keypoints_k", 0),
            )
    finally:
        sys.argv = saved_argv
    act_model.load_state_dict(torch.load(model_path))
    act_model.cuda()
    act_model.eval()
    return act_model, config


def main(
    model_path: str = "../data/model_ckpts/latest/ckpt_latest/model.pth",
    perception_redis_host: str = "localhost",
    perception_redis_port: int = 6380,
    control_redis_host: str = "localhost",
    control_redis_port: int = 6379,
    image_size: int = -1,  # -1 = auto from the ckpt's training config (else mismatch!)
    match_blur_res: int = -1,
    name: str = "model_inference",
    vis: bool = True,
):
    if is_spec(model_path) or not os.path.exists(model_path):
        model_path, _ = resolve_ckpt(model_path)  # variant spec, exp-prefix, or clear error
        print(f"Resolved --model-path shortcut to {model_path}")
    print("Loading model", model_path)
    act_model, config = load_model(model_path)
    arms = list(config["arms"])
    image_only = config.get("image_only", False)
    use_angle = config.get("use_angle", True)
    use_torque = config.get("use_torque", False)
    # crop mode MUST match training or the policy sees a shifted field of view; older
    # ckpts have no 'crop' key and were all trained center-cropped
    crop = config.get("crop", "center")
    obs_horizon = config["obs_horizon"]
    action_horizon = config["action_horizon"]
    action_duration = config["action_duration"]
    print(f"Model loaded: image_only={image_only}, arms={arms}, use_angle={use_angle}, "
          f"use_torque={use_torque}, obs_horizon={obs_horizon}, action_horizon={action_horizon}, "
          f"crop={crop}")

    if image_size < 0:  # auto: deploy at the resolution the ckpt was trained on
        image_size = int(config.get("image_size", 224))
        print(f"--image-size auto -> {image_size} (from ckpt config)")

    # ink-enhanced checkpoints (wipe task): apply the SAME detector + red overlay to the
    # live frame that dataset.py applied at training time (ink_detect.py)
    ink_enhance = bool(config.get("ink_enhance", False))
    if ink_enhance:
        from ink_detect import enhance_ink
        print("ink_enhance ON (from ckpt config): painting detected ink red before crop")

    # ink-keypoint checkpoints (wipe task): the 2k-d drawing-shape feature. Training used
    # each episode's FIRST frame; here we recompute it live while the controller is NOT
    # rolling out (eval_controller:rollout_active == 0) and FREEZE it during the rollout,
    # so the policy keeps the full drawing shape after partial wiping. If the controller
    # doesn't publish the flag, the first computed value is latched forever.
    ink_kp_k = int(config.get("ink_keypoints_k", 0))
    latched_kp = None
    if ink_kp_k:
        from ink_detect import stroke_keypoints
        print(f"ink_keypoints ON (from ckpt config): {ink_kp_k} pts; live until "
              "rollout_active, then frozen")
    elif image_size != int(config.get("image_size", 224)):
        print(f"WARNING: --image-size {image_size} != ckpt training image_size "
              f"{config.get('image_size', 224)} -- the policy will see a train/deploy "
              "resolution mismatch.")

    # sharpness-gap guard: sim-trained policies need the deploy image blurred to match the soft
    # WM renders (PIPELINE §5/§6); real-trained ones must NOT be blurred.
    # Sim datasets in this project all carry "sim" in the folder name
    # (sim_training, wipe_sim_act, wipe_sim_act_nofeel, ...).
    sim_trained = "sim" in os.path.basename(config.get("dataset_path", "").rstrip("/"))
    if match_blur_res < 0:  # auto (the default): pick from the ckpt's training domain
        match_blur_res = 128 if sim_trained else 0
        print(f"--match-blur-res auto -> {match_blur_res} "
              f"({'sim' if sim_trained else 'real'}-trained checkpoint)")
    elif sim_trained and not match_blur_res:
        print("WARNING: sim-trained checkpoint but --match-blur-res=0 -- the real camera is ~12x "
              "sharper than the sim renders; you probably want --match-blur-res 128.")
    elif not sim_trained and match_blur_res:
        print("WARNING: real-trained checkpoint but --match-blur-res is set -- "
              "you probably want --match-blur-res 0.")

    # sim-trained + torque: clamp the live torque obs to the TRAINING data's per-dim
    # mean±3σ. The WM-predicted training torque has near-constant dims (gripper:
    # std 0.013 Nm) whose normalizer amplifies any real-world offset into a huge
    # activation (measured +57σ -> the policy collapses to its end-pose attractor,
    # 2026-08-06). Clamping keeps every dim inside the distribution the policy saw.
    tau_clip = None
    if use_torque:  # ALL torque ckpts: near-constant training dims (gripper: std
        # 0.01-0.02 Nm in both sim AND real demos) turn small physical offsets at
        # deploy into 40-60 sigma obs -> mean-pose collapse (sim 2026-08-06,
        # REAL 2026-08-07: eval grasp seated at 2.1 Nm vs demos' 1.38+/-0.02).
        import glob as _glob
        _eps = sorted(_glob.glob(os.path.join(
            config["dataset_path"], "*", "low_dim_npys",
            "follower_controller_state:tau_interaction_R.npy")))
        if _eps:
            _S = np.concatenate([np.load(p) for p in _eps[:50]])
            _mu, _sd = _S.mean(0), _S.std(0)
            tau_clip = (_mu - 3 * _sd, _mu + 3 * _sd)
            print(f"[torque-ckpt] torque obs clamped to training mean±3σ "
                  f"(gripper dim pinned ≈ {_mu[7]:+.2f} Nm)")
        else:
            print(f"WARNING: torque ckpt but no tau_interaction found under "
                  f"{config['dataset_path']} -- torque obs NOT clamped")

    perception_client = RedisClient(host=perception_redis_host, port=perception_redis_port)
    control_client = RedisClient(host=control_redis_host, port=control_redis_port)

    # low-dim streams to read (state model only; the image-only policy reads no proprioception).
    # Must mirror the dataset's obs layout: per arm [angle (if use_angle)] [torque (if use_torque)].
    lowdim_keys = {}
    if not image_only:
        for arm in arms:
            if use_angle:
                lowdim_keys[f"follower_{arm}_state:actual_angle_rad"] = np.ndarray
            if use_torque:
                lowdim_keys[f"follower_controller_state:tau_interaction_{_TAU_SUFFIX[arm]}"] = np.ndarray

    iter_idx = 0
    while True:
        t_loop_start = time.time()

        image_inputs = perception_client.stream_get_batch({
            "camera_head:timestamp": float,
            "camera_head:image": np.ndarray,
        })
        lowdim_inputs = control_client.stream_get_batch(lowdim_keys) if lowdim_keys else {}

        if any(v is None for v in image_inputs.values()) or \
           any(v is None for v in lowdim_inputs.values()):
            print("waiting for camera" + ("" if image_only else " / follower state / tau_interaction")
                  + " streams...")
            time.sleep(0.05)
            continue

        reference_timestamp = image_inputs["camera_head:timestamp"]

        # ---- image obs (BGR -> [ink overlay] -> center-crop/resize -> RGB, [0,1]) ----
        img_bgr = image_inputs["camera_head:image"]
        if ink_kp_k:
            rollout_active = control_client.stream_get_batch(
                {"eval_controller:rollout_active": float})["eval_controller:rollout_active"]
            if latched_kp is None or not rollout_active:
                latched_kp = stroke_keypoints(img_bgr, k=ink_kp_k, bgr=True).reshape(-1)
        if ink_enhance:
            img_bgr = enhance_ink(img_bgr, bgr=True)  # full-res, before the downscale
        img_bgr = center_crop_resize(img_bgr[None, ...], size=image_size,
                                     crop=crop)[0]  # (H, W, 3) BGR
        if match_blur_res and match_blur_res < image_size:
            # match the soft sim renders: downsample then upsample so the deploy image's detail
            # level matches training (sim renders are ~128px, ~12x softer than the real camera)
            img_bgr = cv2.resize(cv2.resize(img_bgr, (match_blur_res, match_blur_res)),
                                 (image_size, image_size))
        img_rgb = img_bgr[..., [2, 1, 0]]
        if vis:
            cv2.imshow("camera_head (eval obs)", cv2.resize(img_bgr, (672, 672)))
            cv2.waitKey(1)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().cuda() / 255.0  # (3, H, W)
        img_t = img_t.unsqueeze(0).repeat(obs_horizon, 1, 1, 1)  # (obs_horizon, 3, H, W)
        img_t = img_t.unsqueeze(0)  # (1, obs_horizon, 3, H, W)

        obs_dict = {"camera_head": img_t}

        # ---- low-dim obs (state model only): per arm [actual_angle (if use_angle)]
        #      [tau_interaction (if use_torque)] -- same order as the dataset ----
        if not image_only:
            obs_parts = []
            for arm in arms:
                if use_angle:
                    obs_parts.append(lowdim_inputs[f"follower_{arm}_state:actual_angle_rad"])
                if use_torque:
                    _tau = lowdim_inputs[f"follower_controller_state:tau_interaction_{_TAU_SUFFIX[arm]}"]
                    if tau_clip is not None:
                        _tau = np.clip(_tau, tau_clip[0], tau_clip[1])
                    obs_parts.append(_tau)
            if ink_kp_k:
                obs_parts.append(latched_kp)  # episode-constant drawing shape (2k,)
            joint_state = np.concatenate(obs_parts, axis=-1)  # (obs_dim,)
            joint_t = torch.from_numpy(joint_state).float().cuda()
            joint_t = joint_t.unsqueeze(0).repeat(obs_horizon, 1).unsqueeze(0)  # (1, obs_horizon, obs_dim)
            obs_dict["joint_state"] = joint_t

        action_predicted = act_model.predict_action(obs_dict)
        action_timestamp = reference_timestamp + np.linspace(0, action_duration, action_horizon)

        output_data = {f"{name}:action_timestamp": action_timestamp,
                       f"{name}:ckpt_path": model_path}  # lets eval_controller verify the ckpt
        output_maxlen = {f"{name}:action_timestamp": 1, f"{name}:ckpt_path": 1}
        for arm in arms:
            key = f"{name}:action_desired_qpos_{arm}"
            output_data[key] = action_predicted[f"openarm_{arm}_action"].cpu().numpy()[0]  # (H, 8)
            output_maxlen[key] = 1
        control_client.stream_add_batch(output_data, batch_maxlen=output_maxlen)

        iter_idx += 1
        if iter_idx == 1:
            print("model inference worker ready")
        dt = time.time() - t_loop_start
        # diagnostic: without a measured angle obs (image-only or torque-only) show the chunk's own
        # motion range; with one, show how far the last predicted step is from the measured pose.
        if image_only or not use_angle:
            info = " | ".join(
                f"{arm} chunk motion(deg)="
                f"{np.rad2deg(np.ptp(action_predicted[f'openarm_{arm}_action'].cpu().numpy()[0], axis=0).max()):.1f}"
                for arm in arms)
        else:
            info = " | ".join(
                f"{arm} delta max(deg)="
                f"{np.rad2deg(np.abs(action_predicted[f'openarm_{arm}_action'].cpu().numpy()[0, -1] - lowdim_inputs[f'follower_{arm}_state:actual_angle_rad']).max()):.1f}"
                for arm in arms)
        print(f"inference {1.0 / dt:.1f} Hz | {info}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="world_ft box-flip policy inference worker")
    p.add_argument("--model-path", default="../data/model_ckpts/latest/ckpt_latest/model.pth",
                   help="model.pth path, or a selector for the LATEST matching ckpt: "
                        "i = image-only, it = image+torque, its = image+state+torque, "
                        "optionally with a training-domain suffix _real/_sim (e.g. it_sim)")
    p.add_argument("--perception-redis-host", default="localhost")
    p.add_argument("--perception-redis-port", type=int, default=6380)
    p.add_argument("--control-redis-host", default="localhost")
    p.add_argument("--control-redis-port", type=int, default=6379)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--match-blur-res", type=int, default=-1,
                   help="downsample the deploy image to this res then back to --image-size to match "
                        "soft sim renders. Default -1 = AUTO from the ckpt's training data "
                        "(sim-trained -> 128, real-trained -> 0); 0 = explicitly off")
    p.add_argument("--name", default="model_inference")
    p.add_argument("--no-vis", dest="vis", action="store_false")
    args = p.parse_args()
    main(
        model_path=args.model_path,
        perception_redis_host=args.perception_redis_host,
        perception_redis_port=args.perception_redis_port,
        control_redis_host=args.control_redis_host,
        control_redis_port=args.control_redis_port,
        image_size=args.image_size,
        match_blur_res=args.match_blur_res,
        name=args.name,
        vis=args.vis,
    )
