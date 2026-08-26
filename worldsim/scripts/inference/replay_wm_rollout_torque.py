#!/usr/bin/env python
"""Per-frame tau_interaction for an already-recorded in-WM policy rollout.

`rollout_policy_in_wm.py` only evaluates the WM torque head when a NEW ACTION CHUNK is
predicted (once every --exec-steps), so its `tau_interaction` log is one sample per chunk
(10 samples for a 200-step rollout at --exec-steps 20) — far too coarse for the force-viz
composites, which need a value per rendered frame.

This replays a recorded rollout: same seed, same homing, and the exact action sequence from
its .npz, stepping the world model and evaluating the torque head EVERY step. The rollout is
deterministic given (seed episode, dxyz, actions), so the replayed trajectory matches the one
already on video frame-for-frame.

This is how the wipe task's `tau.npz` files were produced (videos/wipping/*/raw/tau.npz); that
script only ever existed in a /tmp scratchpad, so this is the durable version.

Writes `<rollout>_tau.npz` next to each input npz:
    tau_interaction (T, 8)   WM torque_pred - gravity(commanded q), same definition as
                             training and as postprocess_sim_tau_interaction.py
    actions         (T, 8)   the replayed commands (copy of the input, for alignment)

Run (hapticworld env, from HapticWorld/worldsim):
    python scripts/inference/replay_wm_rollout_torque.py   # from HapticWorld/worldsim \
        --rollouts 'wm_policy_rollouts_microwave/<policy dir>/rollout_{15,5,18,2}.npz' \
        --ckpt world_ft_ckpts/.../stage2_trunk/checkpoints/last.ckpt \
        --stage3-ckpt world_ft_ckpts/.../stage3/checkpoints/last.ckpt
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import rollout_policy_in_wm as R  # noqa: E402  (reuse its WMSim / seeding / gravity client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="glob or comma list of rollout_*.npz files to replay")
    ap.add_argument("--ckpt", required=True, help="stage-2 (torque) WM checkpoint")
    ap.add_argument("--stage3-ckpt", default=None)
    ap.add_argument("--home-steps", type=int, default=15,
                    help="must match the --home-steps the rollout was recorded with")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-each-step", action="store_true",
                    help="also call sim.render() every step, exactly as the rollout did when "
                         "writing its video -- test whether decoding perturbs the trajectory")
    args = ap.parse_args()

    paths = []
    for part in args.rollouts.split(","):
        part = part.strip()
        paths.extend(sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part])
    assert paths, f"no rollout npz matched {args.rollouts!r}"

    device = torch.device(args.device)
    amp_dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16,
                     float16=torch.float16)[args.dtype]
    amp_on = args.dtype != "float32"
    wm, cfg = R.load_wm(args.ckpt, device, stage3_ckpt=args.stage3_ckpt)
    wm.dec_infer_steps = 3
    res = (int(cfg.dataset.res_h), int(cfg.dataset.res_w))
    n_ctx = int(getattr(wm, "n_tokens", 10))
    sim = R.WMSim(wm, n_ctx, res, device, amp_dtype, amp_on)
    grav = R.GravityIK()

    import h5py

    def get_seed(path):
        """Same as rollout_policy_in_wm.main()'s nested get_seed (trunk WM path)."""
        with h5py.File(path, "r") as f:
            action0 = f["action"][0].astype(np.float64)      # (8,) raw
        z0, a0, grip = R.seed_from_episode(wm, path, res, device, amp_dtype, amp_on)
        return (z0, a0), grip, action0

    for p in paths:
        d = np.load(p)
        actions = np.asarray(d["actions"], np.float64)          # (T, 8)
        dxyz = np.asarray(d["dxyz"], float)
        stored = np.asarray(d["tau_interaction"], float)        # chunk-boundary log
        exec_steps = int(d["exec_steps"])
        seed_path = str(d["seed_episode"])
        reset_args, seed_gripper, seed_action0 = get_seed(seed_path)
        sim.reset(*reset_args)

        # The video contains BOTH phases (home_steps + steps frames), so both are replayed
        # and logged -- otherwise every torque value lands on a frame home_steps too early.
        # Torque is measured AFTER the step, because that is the state the frame shows.
        target_q7 = (grav.ik_offset(seed_action0[:7], dxyz) if np.any(dxyz)
                     else seed_action0[:7].copy())
        qs, taus = [], []
        for i in range(args.home_steps):          # phase 1: simulated homing
            s = R.smoothstep((i + 1) / args.home_steps)
            q_cmd = np.concatenate([seed_action0[:7] + s * (target_q7 - seed_action0[:7]),
                                    [seed_gripper]])
            sim.step(q_cmd)
            if args.render_each_step:
                sim.render()
            qs.append(q_cmd.copy())
            taus.append(sim.torque_pred8() - grav.gravity8(q_cmd[:7]))
        for q_cmd in actions:                     # phase 2: the policy's recorded actions
            sim.step(q_cmd)
            if args.render_each_step:
                sim.render()
            qs.append(q_cmd.copy())
            taus.append(sim.torque_pred8() - grav.gravity8(q_cmd[:7]))
        qs, taus = np.stack(qs), np.stack(taus)

        # self-check: the rollout logged torque BEFORE executing actions[exec_steps*j],
        # i.e. the state after frame home_steps + exec_steps*j - 1. If the replay is
        # faithful these must match; a large error means the trajectory diverged.
        err = "n/a (image-only, nothing logged)"
        if len(stored):
            k = args.home_steps + exec_steps * np.arange(len(stored)) - 1
            err = f"{np.abs(taus[k] - stored).max():.4f} Nm"
        out = str(Path(p).with_suffix("")) + "_tau.npz"
        np.savez(out, q=qs, tau_interaction=taus, actions=actions,
                 video=str(Path(p).with_suffix("")) + ".mp4")
        print(f"{Path(p).name}: {len(qs)} frames (= {args.home_steps} home + {len(actions)}), "
              f"|grip tau| max {np.abs(taus[:, 7]).max():.2f} N.m, "
              f"max|replay - stored| = {err}")


if __name__ == "__main__":
    main()
