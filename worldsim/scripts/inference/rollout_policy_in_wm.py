#!/usr/bin/env python
"""Closed-loop ACT policy rollout inside HapticWorld.

The world model replaces the real robot and scene. Each cycle: the current latent is decoded
to a camera frame and the predicted torque is read from the torque head; the policy receives
both and predicts a 20-step chunk of joint positions; the first --exec-steps of the chunk are
stepped through the latent dynamics; repeat. This is the same receding-horizon loop the
real-robot controller runs, with the simulator in place of the hardware.

Rollout protocol (mirrors the real evaluation):
  1. seed the simulator from one of the task's seed episodes (frame 0 + start pose). Seeds
     differ in object placement and arm pose, so sampling one per rollout randomizes the
     initial scene; pass a single file to --seed-episodes to pin it,
  2. hold the seed pose for --home-steps steps to warm up the latent context (add --random-home
     for an extra random end-effector offset),
  3. hand control to the policy for --steps steps at 10 Hz,
  4. write <out>/<policy>/rollout_<k>.mp4 and an .npz log. Success is judged from the video.

The policy's torque input is tau_interaction = predicted total torque - gravity(commanded q),
the same definition the training demonstrations use. Gravity and IK
come from the arm model through policy/gravity_ik_server.py, spawned as a subprocess.

Run from worldsim/ in the hapticworld env, for example:
    python scripts/inference/rollout_policy_in_wm.py --task boxflip --policies it_sim --n-rollouts 5
    python scripts/inference/rollout_policy_in_wm.py --task wipe --policies it_sim
Policies are named <policy> (of --task) or <task>/<policy>, or given as a model.pth path.
"""
import argparse
import glob as globlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

_REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "worldsim"))

from hapticworld.algorithms.common.diffusion_helper import render_img_cm  # noqa: E402
from hapticworld.inference import load_model as load_wm, seed_from_episode  # noqa: E402
from policy.ckpt_select import resolve_ckpt  # noqa: E402
from policy.data.dataset import center_crop_resize  # noqa: E402
from policy.load_policy import load_policy  # noqa: E402
from policy.tasks import TASKS, seeds_glob, wm_ckpt  # noqa: E402


class GravityIK:
    """Client for gravity_ik_server.py (spawned as a subprocess of this env)."""

    def __init__(self):
        server = os.path.join(_REPO, "policy", "gravity_ik_server.py")
        self.p = subprocess.Popen(
            [sys.executable, "-u", server],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=_REPO)
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
        return (img * 255).astype(np.uint8)  # (H, W, 3) RGB

    @torch.no_grad()
    def torque_pred8(self):
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_on):
            tau = self.wm._val_torque_pred(self.z_ctx, self.a_ctx)[:, -1]
        return tau[0].float().cpu().numpy().astype(np.float64)


def policy_obs(frame_rgb, image_size, obs_horizon, crop):
    img = center_crop_resize(frame_rgb[None].astype(np.float32), size=image_size, crop=crop)[0] / 255.0
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
    ap.add_argument("--task", choices=list(TASKS), default="boxflip",
                    help="task: sets the world model, seeds, chunk renewal and rollout length")
    ap.add_argument("--policies", default="it_sim",
                    help="comma list of policies: <policy> of --task, <task>/<policy>, or a model.pth path")
    ap.add_argument("--ckpt", default=None, help="stage-2 world-model checkpoint (default: the task's)")
    ap.add_argument("--stage3-ckpt", default=None, help="stage-3 decoder checkpoint (default: the task's)")
    ap.add_argument("--seed-episodes", default=None,
                    help="glob of seed hdf5s, one sampled per rollout (default: the task's data/seeds). "
                         "Pass a single file to pin the scene.")
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=None, help="policy steps per rollout at 10 Hz (default: the task's)")
    ap.add_argument("--exec-steps", type=int, default=None,
                    help="chunk actions executed per policy call (default: the task's; 5 = 0.5 s)")
    ap.add_argument("--random-home", action=argparse.BooleanOptionalAction, default=False,
                    help="add a random end-effector start offset on top of the seed pose")
    ap.add_argument("--home-rand-xy-m", type=float, default=0.1)
    ap.add_argument("--home-rand-z-min-m", type=float, default=-0.02)
    ap.add_argument("--home-rand-z-max-m", type=float, default=0.03)
    ap.add_argument("--home-steps", type=int, default=15, help="simulated homing steps (10 Hz)")
    ap.add_argument("--out", default=None, help="output dir (default: wm_policy_rollouts/<task>)")
    ap.add_argument("--seed", type=int, default=0, help="numpy seed for the seed-episode / offset draw")
    ap.add_argument("--replay", default=None,
                    help="comma list of rollout .npz logs: re-run each with ITS recorded seed "
                         "episode + start offset (e.g. with a longer --steps). Outputs are named "
                         "<npz stem>_<seconds>s.*; --n-rollouts / --random-home are ignored")
    ap.add_argument("--dec-infer-steps", type=int, default=3)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    task = TASKS[args.task]
    args.ckpt = args.ckpt or wm_ckpt(args.task, 2)
    args.stage3_ckpt = args.stage3_ckpt or wm_ckpt(args.task, 3)
    args.seed_episodes = args.seed_episodes or seeds_glob(args.task)
    args.steps = args.steps or task["steps"]
    args.exec_steps = args.exec_steps or task["exec_steps"]
    args.out = args.out or os.path.join("wm_policy_rollouts", args.task)

    device = args.device if torch.cuda.is_available() else "cpu"
    amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    amp_on = args.dtype != "float32" and device.startswith("cuda")

    wm, cfg = load_wm(args.ckpt, device, stage3_ckpt=args.stage3_ckpt)
    wm.dec_infer_steps = args.dec_infer_steps
    res = ((int(cfg.dataset.res_h), int(cfg.dataset.res_w)) if "res_h" in cfg.dataset
           else (int(cfg.dataset.resolution),) * 2)
    n_ctx = int(getattr(wm, "n_tokens", 10))
    sim = WMSim(wm, n_ctx, res, device, amp_dtype, amp_on)
    print(f"[rollout] task={args.task}  world model: {args.ckpt} + {args.stage3_ckpt}")

    seed_paths = sorted(globlib.glob(args.seed_episodes, recursive=True))
    if not seed_paths and not args.replay:
        raise FileNotFoundError(f"no seed episodes match {args.seed_episodes!r} "
                                "(python checkpoints/download_assets.py fetches them)")
    seed_cache = {}  # path -> ((z0, a0), seed_gripper, seed_action0)

    def get_seed(path):
        if path not in seed_cache:
            with h5py.File(path, "r") as f:
                action0 = f["action"][0].astype(np.float64)  # (8,) raw
            z0, a0, grip = seed_from_episode(wm, path, res, device, amp_dtype, amp_on)
            seed_cache[path] = ((z0, a0), grip, action0)
        return seed_cache[path]

    print(f"[rollout] world model ready (res={res}, n_ctx={n_ctx}); {len(seed_paths)} seed episodes")

    grav = GravityIK()
    print("[rollout] gravity/IK server ready")
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out)

    # the job list (seed episode, start offset, output stem) is built once and reused for every
    # policy, so rollout k is a paired comparison (same initial scene) across policies
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
        if not os.path.exists(spec) and "/" not in spec:
            spec = f"{args.task}/{spec}"
        model_path, pcfg = resolve_ckpt(spec)
        if not pcfg.get("use_torque", False) or pcfg.get("use_angle", False):
            print(f"[rollout] SKIP {spec}: the simulator provides image + torque only "
                  f"(use_torque={pcfg.get('use_torque')}, use_angle={pcfg.get('use_angle')})")
            continue
        policy, pcfg = load_policy(model_path)
        obs_horizon, image_size = pcfg["obs_horizon"], pcfg.get("image_size", 224)
        crop = pcfg.get("crop", "center")
        if os.path.exists(spec):  # explicit model.pth -> "<run_dir>_<ckpt_step>"
            _p = Path(spec)
            pname = f"{_p.parent.parent.name}_{_p.parent.name}"
        else:
            pname = spec.split("/", 1)[1]
        pdir = out_root / pname
        pdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[rollout] === {spec} -> {model_path} ===")

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

            # phase 1: hold the seed pose (or move to the offset pose) to fill the latent context
            if np.any(dxyz):
                target_q7 = grav.ik_offset(seed_action0[:7], dxyz)
            else:
                target_q7 = seed_action0[:7].copy()
            video = VideoOut(pdir / f"{stem}.mp4", fps=10, upscale=args.upscale)
            tag = (f"{pname} {stem} {seed_tag} "
                   f"off[cm] {dxyz[0]*100:+.0f},{dxyz[1]*100:+.0f},{dxyz[2]*100:+.0f}")
            for i in range(args.home_steps):
                s = smoothstep((i + 1) / args.home_steps)
                q_cmd = np.concatenate([seed_action0[:7] + s * (target_q7 - seed_action0[:7]),
                                        [seed_gripper]])
                sim.step(q_cmd)
                video.add(sim.render(), f"{tag} home")

            # phase 2: policy in the loop (receding horizon)
            actions_log, tau_log = [], []
            queue = []
            for t in range(args.steps):
                if not queue:
                    obs = {"camera_head": policy_obs(sim.render(), image_size, obs_horizon, crop)}
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
                     tau_interaction=np.stack(tau_log),
                     seed_action0=seed_action0, seed_episode=str(seed_path),
                     exec_steps=args.exec_steps, task=args.task, policy=model_path)
            print(f"[rollout] {pname} {stem}: seed={seed_tag} offset(cm)=({dxyz[0]*100:+.1f},"
                  f"{dxyz[1]*100:+.1f},{dxyz[2]*100:+.1f}) {args.steps} steps in "
                  f"{time.time()-t0:.0f}s -> {pdir / f'{stem}.mp4'}")
        del policy
        torch.cuda.empty_cache()

    grav.close()
    print(f"\n[rollout] done. Videos under {out_root}/<policy>/")


if __name__ == "__main__":
    main()
