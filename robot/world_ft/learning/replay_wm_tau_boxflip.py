"""Replay saved box-flip WM-rollout actions through the WM torque head -> per-frame
tau_interaction (WM torque_pred - gravity(commanded q)) for the force visualization.

Box-flip twin of the wipe task's replay_wm_tau.py: same mechanics, but the TRUNK world
model (stage2 + stage3, stage-1 lineage 2026-07-15_02-35-54) and the four box-flip
policies in worldsim/wm_policy_rollouts/.

Why a replay is needed: the rollout npz stores actions at every step but tau only at
inference times (every exec_steps), and never at all for the image-only policies. So we
re-run the deterministic latent dynamics from the same seed + dxyz homing and query the
torque head at every step. Validation: replayed tau at the inference frames must match
the stored values (torque policies only) -- measured max deviation 0.24-0.53 Nm, bf16
sampling noise over a 200-step horizon.

Run (hapticworld env):
  python robot/world_ft/learning/replay_wm_tau_boxflip.py   # from the HapticWorld root
"""
import os
import sys

import numpy as np
import torch

_HW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(os.path.join(_HW, "worldsim"))
sys.path.insert(0, os.path.join(_HW, "worldsim", "scripts", "inference"))
import rollout_policy_in_wm as R  # noqa: E402

OUTD = (os.path.join(_HW, "data", "force_viz") + "/"
        "boxflip_wm/tau")
HOME_STEPS = 15  # rollout_policy_in_wm.py --home-steps default, as recorded in the npz
# The same rollout index is the same initial scene for all four policies
# (rollout_policy_in_wm.py builds the job list once and reuses it), so equal indices are
# paired comparisons.
#
# GATE_SET = the WM_EVAL 3b composite realism gate selection: 8 = all four PASS (control);
# 3/13/16/17 = i_sim FAILs while i_real/it_real/it_sim PASS (the headline contrast). Hard
# seeds 9 and 15 excluded there -- they fail for multiple healthy policies.
# The extra indices per policy are the hand-picked rollouts for the paper video set
# (videos/flipping/), chosen by eye from the rollout mp4s.
GATE_SET = [3, 8, 13, 16, 17]
SELECTION = {
    "i_real": GATE_SET,
    "i_sim": GATE_SET + [0, 2, 9],
    "it_real": GATE_SET + [0, 4, 18],
    "it_sim": GATE_SET + [2, 14],
}
POLICIES = tuple(SELECTION)

os.makedirs(OUTD, exist_ok=True)
device = "cuda:0"
amp_dtype = torch.bfloat16
wm, cfg = R.load_wm(os.path.join(_HW, "checkpoints/wm/boxflip/stage2/checkpoints/last.ckpt"), device,
                    stage3_ckpt=os.path.join(_HW, "checkpoints/wm/boxflip/stage3/checkpoints/last.ckpt"))
wm.dec_infer_steps = 3
res = ((int(cfg.dataset.res_h), int(cfg.dataset.res_w)) if "res_h" in cfg.dataset
       else (int(cfg.dataset.resolution),) * 2)  # square-era ckpts (box-flip)
n_ctx = int(getattr(wm, "n_tokens", 10))
sim = R.WMSim(wm, n_ctx, res, device, amp_dtype, True)
grav = R.GravityIK()
print(f"[replay] WM ready (res={res}, n_ctx={n_ctx})")

for policy in POLICIES:
    folder = f"wm_policy_rollouts/{policy}"
    for idx in sorted(SELECTION[policy]):
        out_npz = os.path.join(OUTD, f"{policy}__rollout_{idx}_tau.npz")
        if os.path.exists(out_npz):
            print(f"[replay] {policy} rollout_{idx}: cached, skip")
            continue
        d = np.load(os.path.join(folder, f"rollout_{idx}.npz"))
        seed_path = str(d["seed_episode"])
        dxyz = np.asarray(d["dxyz"], float)
        actions = np.asarray(d["actions"], float)
        stored = np.asarray(d["tau_interaction"], float)
        exec_steps = int(d["exec_steps"])
        seed_action0 = np.asarray(d["seed_action0"], float)

        z0, a0, grip = R.seed_from_episode(wm, seed_path, res, device, amp_dtype, True)
        sim.reset(z0, a0)
        q_cmd = seed_action0.copy()
        target_q7 = (grav.ik_offset(seed_action0[:7], dxyz) if np.any(dxyz)
                     else seed_action0[:7].copy())
        qs, taus = [], []
        for i in range(HOME_STEPS):
            s = R.smoothstep((i + 1) / HOME_STEPS)
            q_cmd = np.concatenate([seed_action0[:7] + s * (target_q7 - seed_action0[:7]),
                                    [grip]])
            sim.step(q_cmd)
            qs.append(q_cmd.copy())
            taus.append(sim.torque_pred8() - grav.gravity8(q_cmd[:7]))
        for t in range(len(actions)):
            q_cmd = actions[t]
            sim.step(q_cmd)
            qs.append(q_cmd.copy())
            taus.append(sim.torque_pred8() - grav.gravity8(q_cmd[:7]))
        qs, taus = np.stack(qs), np.stack(taus)

        # stored tau_log[j] was computed BEFORE executing actions[exec_steps*j]:
        # context after frame HOME_STEPS + exec_steps*j - 1, gravity at that frame's q
        err = "n/a (image-only)"
        if len(stored):
            k = HOME_STEPS + exec_steps * np.arange(len(stored)) - 1
            err = f"{np.abs(taus[k] - stored).max():.4f} Nm"
        np.savez(out_npz, q=qs, tau_interaction=taus,
                 video=os.path.join(folder, f"rollout_{idx}.mp4"))
        print(f"[replay] {policy} rollout_{idx}: {len(qs)} frames, "
              f"max|replay - stored| = {err}")
grav.close()
print("done")
