#!/usr/bin/env python
"""Closed-loop ACT policy rollout INSIDE the world model (no robot, no redis, offline).

The trained world model plays the environment (as in wm_follower_worker.py) and the trained
ACT policy plays the human: each cycle the current latent is decoded to the camera view, the
policy predicts a 20-step chunk from it (+ predicted interaction torque for torque policies),
the first --exec-steps actions are fed back into the latent dynamics (receding horizon,
matching the real eval controller's 0.5 s chunk renewal).

Rollout protocol mirrors the real rig:
  1. seed the WM from a RANDOM episode matching --seed-episodes (frame 0) -- across episodes the
     BOX sits at different positions (and the arm start pose differs), so sampling the seed
     randomizes the initial scene; point it at a single file to pin the scene,
  2. hold the seed pose briefly to warm up the WM's context (default), or with --random-home
     additionally home to a randomized start pose (EE offset x,y +/-0.10 m, z [-0.02,0.03] m,
     like eval_controller) -- unnecessary when the seeds were captured with varied arm poses
     (capture_wm_seeds.py teleop flow), which is the recommended way,
  3. hand control to the policy for --steps steps (10 Hz),
  4. save <out>/<policy>/rollout_<k>.mp4 (+ .npz log). Success is scored by eye.

Torque policies get tau_interaction = WM torque_pred - gravity(commanded q), the same
definition as training (postprocess_sim_tau_interaction.py). Gravity/IK need the MJCF
pinocchio solver -- a helper server (robot/world_ft/learning/gravity_ik_server.py)
is spawned in this same environment automatically.

Env: hapticworld, run from HapticWorld/worldsim:
    python scripts/inference/rollout_policy_in_wm.py --policies i_sim,it_sim
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = str(Path(__file__).resolve().parents[3])
ROBOT = os.path.join(_REPO, "robot")
_LEARNING = os.path.join(ROBOT, "world_ft/learning")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, _LEARNING)
sys.path.insert(0, ROBOT)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # sys.path bootstrap: worldsim root
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm  # noqa: E402
from wm_follower_worker import load_model as load_wm, seed_from_episode  # noqa: E402
from wm_follower_worker_state import (  # noqa: E402
    StateOnlineRollout,
    load_model as load_wm_state,
    seed_from_episode as seed_from_episode_state,
)
from ckpt_select import resolve_ckpt  # noqa: E402
from dataset import center_crop_resize  # noqa: E402
from eval_inference_worker import load_model as load_policy  # noqa: E402

CKPT_ROOT = os.path.join(_REPO, "checkpoints", "policies")


class GravityIK:
    """Client for gravity_ik_server.py (spawned as a subprocess of this env)."""

    def __init__(self):
        server = os.path.join(_LEARNING, "gravity_ik_server.py")
        self.p = subprocess.Popen(
            [sys.executable, "-u", server],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=ROBOT)
        for _ in range(60):  # wait for {"ready": true} (solver import takes a few seconds)
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("gravity/IK server exited during startup")
            try:
                if json.loads(line).get("ready"):
                    break
            except json.JSONDecodeError:
                continue  # skip any non-protocol startup noise
        else:
            raise RuntimeError("gravity/IK server never became ready")

    def _call(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("gravity/IK server died")
        out = json.loads(line)
        if "error" in out:
            raise RuntimeError(f"gravity/IK server error: {out['error']}")
        return out

    def gravity8(self, q7):
        return np.asarray(self._call({"op": "gravity", "q7": [float(x) for x in q7]})["tau8"])

    def ik_offset(self, q7, dxyz):
        return np.asarray(self._call({"op": "ik_offset", "q7": [float(x) for x in q7],
                                      "dxyz": [float(x) for x in dxyz]})["q7"])

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except Exception:
            pass


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 3 * s ** 2 - 2 * s ** 3


class WMSim:
    """Thin stepping wrapper: holds (z_ctx, a_ctx), steps dynamics, decodes, predicts torque."""

    def __init__(self, wm, n_ctx, res, device, amp_dtype, amp_on):
        self.wm, self.n_ctx, self.res = wm, n_ctx, res
        self.device, self.amp_dtype, self.amp_on = device, amp_dtype, amp_on
        self.z_ctx = self.a_ctx = None

    def reset(self, z0, a0):
        self.z_ctx, self.a_ctx = z0.clone(), a0.clone()

    @torch.no_grad()
    def step(self, action8):
        a_new = self.wm.normalizer["action"].normalize(
            torch.from_numpy(np.asarray(action8, np.float32))[None, None]
            .to(device=self.device, dtype=self.wm.dtype)).clamp(-1.0, 1.0)
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_on):
            z_seq = self.wm.dynamics_forward(self.z_ctx, torch.cat([self.a_ctx, a_new], dim=1))
        self.z_ctx = torch.cat([self.z_ctx, z_seq[:, -1:].float()], dim=1)[:, -self.n_ctx:]
        self.a_ctx = torch.cat([self.a_ctx, a_new], dim=1)[:, -self.n_ctx:]

    @torch.no_grad()
    def render(self):
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_on):
            img = render_img_cm(self.wm, self.z_ctx[:, -1], self.res,
                                self.wm.normalizer, num_views=self.wm.num_views)
        img = img[0, :3].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        return (img * 255).astype(np.uint8)  # (res, res, 3) RGB

    @torch.no_grad()
    def torque_pred8(self):
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_on):
            tau = self.wm._val_torque_pred(self.z_ctx, self.a_ctx)[:, -1]
        return tau[0].float().cpu().numpy().astype(np.float64)


class WMSimState:
    """WMSim-compatible wrapper around the STATE-conditioned WM's coupled rollout
    (LatentWorldModelState via StateOnlineRollout): dynamics conditioned on
    [q_des, q_measured] with q_measured predicted by the GRU state branch."""

    def __init__(self, wm, n_ctx, res, device, amp_dtype, amp_on):
        self.wm, self.n_ctx, self.res = wm, n_ctx, res
        self.device, self.amp_dtype, self.amp_on = device, amp_dtype, amp_on
        self._ro = self._z_disp = self._tau = None

    def reset(self, z0, s0_raw, a0_raw):
        self._ro = StateOnlineRollout(self.wm, z0, s0_raw, a0_raw, self.device,
                                      self.amp_dtype, self.amp_on, self.n_ctx)
        self._z_disp = self._tau = None

    @torch.no_grad()
    def step(self, action8):
        q = torch.from_numpy(np.asarray(action8, np.float32))[None].to(self.device)
        self._z_disp, self._tau, _ = self._ro.step(q)

    @torch.no_grad()
    def render(self):
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_on):
            img = render_img_cm(self.wm, self._z_disp, self.res,
                                self.wm.normalizer, num_views=self.wm.num_views)
        img = img[0, :3].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        return (img * 255).astype(np.uint8)  # (res, res, 3) RGB

    def torque_pred8(self):
        return self._tau[0].float().cpu().numpy().astype(np.float64)


def policy_obs(frame_rgb, image_size, obs_horizon):
    img = center_crop_resize(frame_rgb[None].astype(np.float32), size=image_size)[0] / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).float().cuda()
    return t.unsqueeze(0).repeat(obs_horizon, 1, 1, 1).unsqueeze(0)  # (1, T, 3, H, W)


class VideoOut:
    def __init__(self, path, fps, upscale):
        self.path, self.fps, self.upscale, self.proc = str(path), fps, upscale, None

    def add(self, frame_rgb, label):
        h, w = frame_rgb.shape[:2]
        img = cv2.resize(frame_rgb, (w * self.upscale, h * self.upscale),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        if self.proc is None:
            H, W = img.shape[:2]
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{W}x{H}", "-r", f"{self.fps:g}", "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-preset", "fast", "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", default="i_sim,it_sim,i_real,it_real",
                    help="comma list of ckpt selectors (ckpt_select specs) or model.pth paths")
    ap.add_argument("--wm", choices=["default", "state"], default="default",
                    help="world model variant: default (stage2+stage3) or state "
                         "(stage2_state+stage3_state, coupled q-measured rollout; the ckpt "
                         "defaults switch automatically)")
    ap.add_argument("--ckpt", default=os.path.join(_REPO, "checkpoints/wm/boxflip/stage2/checkpoints/last.ckpt"))
    ap.add_argument("--stage3-ckpt", default=os.path.join(_REPO, "checkpoints/wm/boxflip/stage3/checkpoints/last.ckpt"))
    ap.add_argument("--seed-episodes", default=os.path.join(_REPO, "data/wm_seed_live/episode_*.hdf5"),
                    help="glob of seed hdf5s; ONE is sampled per rollout (capture them with "
                         "robot/world_ft/capture_wm_seeds.py -- one ENTER per box "
                         "position). Pass a single file to pin the scene.")
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=185,
                    help="policy steps per rollout (10 Hz; 185 + 15 homing = 20 s video)")
    ap.add_argument("--exec-steps", type=int, default=5,
                    help="chunk actions executed per inference (5 = 0.5 s receding horizon)")
    ap.add_argument("--random-home", action=argparse.BooleanOptionalAction, default=False,
                    help="add a random EE start offset on top of the seed pose (off by default: "
                         "capture seeds with varied arm poses instead)")
    ap.add_argument("--home-rand-xy-m", type=float, default=0.1)
    ap.add_argument("--home-rand-z-min-m", type=float, default=-0.02)
    ap.add_argument("--home-rand-z-max-m", type=float, default=0.03)
    ap.add_argument("--home-steps", type=int, default=15, help="simulated homing steps (10 Hz)")
    ap.add_argument("--out", default="wm_policy_rollouts")
    ap.add_argument("--seed", type=int, default=0, help="numpy seed for the start-pose offsets")
    ap.add_argument("--replay", default=None,
                    help="comma list of rollout .npz logs: re-run each with ITS recorded seed "
                         "episode + start offset (e.g. with a longer --steps). Outputs are named "
                         "<npz stem>_<seconds>s.*; --n-rollouts / --random-home are ignored")
    ap.add_argument("--dec-infer-steps", type=int, default=3)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    amp_on = args.dtype != "float32" and device.startswith("cuda")

    if args.wm == "state":  # switch untouched ckpt defaults to the state-matched pair
        if args.ckpt == ap.get_default("ckpt"):
            args.ckpt = "world_ft_ckpts/stage2_state/checkpoints/last.ckpt"
        if args.stage3_ckpt == ap.get_default("stage3_ckpt"):
            args.stage3_ckpt = "world_ft_ckpts/stage3_state/checkpoints/last.ckpt"
        wm, cfg = load_wm_state(args.ckpt, device, stage3_ckpt=args.stage3_ckpt)
    else:
        wm, cfg = load_wm(args.ckpt, device, stage3_ckpt=args.stage3_ckpt)
    wm.dec_infer_steps = args.dec_infer_steps
    res = ((int(cfg.dataset.res_h), int(cfg.dataset.res_w)) if "res_h" in cfg.dataset
           else (int(cfg.dataset.resolution),) * 2)  # square-era ckpts (box-flip)
    n_ctx = int(getattr(wm, "n_tokens", 10))
    sim_cls = WMSimState if args.wm == "state" else WMSim
    sim = sim_cls(wm, n_ctx, res, device, amp_dtype, amp_on)
    print(f"[rollout] WM variant: {args.wm} ({args.ckpt} + {args.stage3_ckpt})")

    import glob as globlib
    import h5py
    seed_paths = sorted(globlib.glob(args.seed_episodes, recursive=True))
    if not seed_paths and not args.replay:
        raise FileNotFoundError(f"no seed episodes match {args.seed_episodes!r}")
    seed_cache = {}  # path -> (z0, a0, seed_gripper, seed_action0)

    def get_seed(path):
        """-> (reset_args tuple for sim.reset(*...), seed_gripper, seed_action0 (8,) raw)."""
        if path not in seed_cache:
            with h5py.File(path, "r") as f:
                action0 = f["action"][0].astype(np.float64)  # (8,) raw
            if args.wm == "state":
                z0, s0_raw, a0_raw, grip = seed_from_episode_state(
                    wm, path, res, device, amp_dtype, amp_on)
                seed_cache[path] = ((z0, s0_raw, a0_raw), grip, action0)
            else:
                z0, a0, grip = seed_from_episode(wm, path, res, device, amp_dtype, amp_on)
                seed_cache[path] = ((z0, a0), grip, action0)
        return seed_cache[path]

    print(f"[rollout] WM ready (res={res}, n_ctx={n_ctx}); "
          f"{len(seed_paths)} seed episodes (box position varies across them)")

    grav = GravityIK()
    print("[rollout] gravity/IK server ready (hapticworld)")
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out)

    # the job list -- (seed episode, start offset, output stem) -- is built ONCE and reused for
    # every policy, so rollout #k is a paired comparison (same initial scene) across policies.
    jobs = []
    if args.replay:
        for npz_path in [s.strip() for s in args.replay.split(",") if s.strip()]:
            d = np.load(npz_path)
            stem = f"{Path(npz_path).stem}_{(args.home_steps + args.steps) / 10:g}s"
            jobs.append((str(d["seed_episode"]), np.asarray(d["dxyz"], float), stem))
        print(f"[rollout] REPLAY mode: {len(jobs)} rollouts from recorded logs")
    else:
        for k in range(args.n_rollouts):
            seed_path = seed_paths[int(rng.integers(len(seed_paths)))]
            dxyz = np.zeros(3)
            if args.random_home:
                dxyz = np.array([rng.uniform(-args.home_rand_xy_m, args.home_rand_xy_m),
                                 rng.uniform(-args.home_rand_xy_m, args.home_rand_xy_m),
                                 rng.uniform(args.home_rand_z_min_m, args.home_rand_z_max_m)])
            jobs.append((seed_path, dxyz, f"rollout_{k}"))

    for spec in [s.strip() for s in args.policies.split(",") if s.strip()]:
        model_path, _ = resolve_ckpt(spec, root=CKPT_ROOT)
        policy, pcfg = load_policy(model_path)
        image_only = pcfg.get("image_only", False)
        use_torque = pcfg.get("use_torque", False)
        if not image_only and pcfg.get("use_angle", True):
            print(f"[rollout] SKIP {spec}: needs measured joint angle, which the sim cannot provide")
            continue
        obs_horizon, image_size = pcfg["obs_horizon"], pcfg.get("image_size", 224)
        # short-spec -> subdir named after the spec; explicit model.pth path -> derive
        # "<run_dir>_<ckpt_step>" (joining out_root with an ABSOLUTE path would resolve
        # to the path itself and mkdir onto the model file)
        if "/" in spec:
            _p = Path(spec)
            pname = f"{_p.parent.parent.name}_{_p.parent.name}"
        else:
            pname = spec
        pdir = out_root / pname
        pdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[rollout] === {spec} -> {model_path} (image_only={image_only}, "
              f"use_torque={use_torque}) ===")

        for seed_path, dxyz, stem in jobs:
            t0 = time.time()
            if args.replay:  # replays never clobber an earlier attempt: rollout_1_20s, _try2, ...
                base, attempt = stem, 2
                while (pdir / f"{stem}.mp4").exists():
                    stem = f"{base}_try{attempt}"
                    attempt += 1
            seed_tag = "/".join(Path(seed_path).parts[-2:]).replace("episode_", "ep").replace(".hdf5", "")
            reset_args, seed_gripper, seed_action0 = get_seed(seed_path)
            sim.reset(*reset_args)
            q_cmd = seed_action0.copy()

            # ---- phase 1: simulated homing (job offset; zero offset = hold the seed pose) ----
            if np.any(dxyz):
                target_q7 = grav.ik_offset(seed_action0[:7], dxyz)
            else:
                target_q7 = seed_action0[:7].copy()
            video = VideoOut(pdir / f"{stem}.mp4", fps=10, upscale=args.upscale)
            tag = (f"{spec} {stem} {seed_tag} "
                   f"off[cm] {dxyz[0]*100:+.0f},{dxyz[1]*100:+.0f},{dxyz[2]*100:+.0f}")
            for i in range(args.home_steps):
                s = smoothstep((i + 1) / args.home_steps)
                q_cmd = np.concatenate([seed_action0[:7] + s * (target_q7 - seed_action0[:7]),
                                        [seed_gripper]])
                sim.step(q_cmd)
                video.add(sim.render(), f"{tag} home")

            # ---- phase 2: policy in the loop (receding horizon) ----
            actions_log, tau_log = [], []
            queue = []
            for t in range(args.steps):
                if not queue:
                    obs = {"camera_head": policy_obs(sim.render(), image_size, obs_horizon)}
                    if use_torque:
                        tau_int = sim.torque_pred8() - grav.gravity8(q_cmd[:7])
                        tau_log.append(tau_int)
                        j = torch.from_numpy(tau_int.astype(np.float32)).float().cuda()
                        obs["joint_state"] = j.unsqueeze(0).repeat(obs_horizon, 1).unsqueeze(0)
                    with torch.no_grad():
                        chunk = policy.predict_action(obs)["openarm_right_action"][0].cpu().numpy()
                    queue = list(chunk[:args.exec_steps])
                q_cmd = np.asarray(queue.pop(0), np.float64)
                sim.step(q_cmd)
                actions_log.append(q_cmd)
                video.add(sim.render(), f"{tag} t={t}")

            video.close()
            np.savez(pdir / f"{stem}.npz",
                     dxyz=dxyz, actions=np.stack(actions_log),
                     tau_interaction=(np.stack(tau_log) if tau_log else np.zeros((0, 8))),
                     seed_action0=seed_action0, seed_episode=str(seed_path),
                     exec_steps=args.exec_steps)
            print(f"[rollout] {spec} {stem}: seed={seed_tag} offset(cm)=({dxyz[0]*100:+.1f},"
                  f"{dxyz[1]*100:+.1f},{dxyz[2]*100:+.1f}) {args.steps} steps in "
                  f"{time.time()-t0:.0f}s -> {pdir / f'{stem}.mp4'}")
        del policy
        torch.cuda.empty_cache()

    grav.close()
    print(f"\n[rollout] done. Videos under {out_root}/<policy>/ -- score success by eye.")


if __name__ == "__main__":
    main()
