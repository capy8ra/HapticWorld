#!/usr/bin/env python
"""Cycle-consistency realism score for WM policy-rollout videos (offline, from mp4s).

For each frame: downscale to the WM res, encode -> decode, and measure the reconstruction
error (MSE x1000, excluding the overlay-text rows). Frames the autoencoder knows (on the
training manifold) round-trip cleanly; drifted/blurred frames reconstruct poorly. Per video
this reports:  early err (frames ~5-15), late err (last 25%), late/early ratio, and max.
A degraded rollout shows late >> early; healthy ones stay flat.

Baselines: a few sim_training mp4s (teleop-generated = in-distribution dynamics) define the
healthy range. Uses the SAME stage2+stage3 decoder stack as the rollouts.

Env: hapticworld, from HapticWorld/worldsim:
    python scripts/inference/score_wm_realism.py \
        --videos 'wm_policy_rollouts/*/rollout_*.mp4' --out realism_scores.csv
"""
import os
import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "robot"))  # nova.redis_client
from wm_follower_worker import load_model as load_wm  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # sys.path bootstrap: worldsim root
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm  # noqa: E402


def read_frames(path, res, stride):
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            f = cv2.resize(f, (res[1], res[0]), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return np.stack(frames) if frames else None


@torch.no_grad()
def cycle_err(model, frames, obs_key, device, amp_dtype, amp_on, text_rows, batch=32):
    x = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)  # (T,3,H,W)
    errs = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
            z = model.encoder_forward(model.normalizer[obs_key].normalize(xb)).float()
            rec = render_img_cm(model, z, xb.shape[-1], model.normalizer,
                                num_views=model.num_views, batch_size=16)
        rec = rec[:, :3].clamp(0, 1).float()
        d = (rec - xb.to(rec.device)) ** 2
        errs.append(d[:, :, text_rows:, :].mean(dim=(1, 2, 3)).cpu().numpy() * 1000.0)
    return np.concatenate(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="wm_policy_rollouts/*/rollout_*.mp4")
    ap.add_argument("--baseline", default=(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "robot", "world_ft") + "/"
                                           "data/sim_training/*/camera_head.mp4"))
    ap.add_argument("--n-baseline", type=int, default=5)
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "checkpoints/wm/boxflip/stage2/checkpoints/last.ckpt"))
    ap.add_argument("--stage3-ckpt", default="world_ft_ckpts/stage3/checkpoints/last.ckpt")
    ap.add_argument("--stride", type=int, default=2, help="score every Nth frame")
    ap.add_argument("--text-rows", type=int, default=12,
                    help="top rows excluded from the error (video overlay text)")
    ap.add_argument("--out", default="realism_scores.csv")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    amp_on = args.dtype != "float32" and device.startswith("cuda")
    model, cfg = load_wm(args.ckpt, device, stage3_ckpt=args.stage3_ckpt)
    res = ((int(cfg.dataset.res_h), int(cfg.dataset.res_w)) if "res_h" in cfg.dataset
           else (int(cfg.dataset.resolution),) * 2)  # square-era ckpts (box-flip)
    obs_key = model.obs_keys[0]

    videos = sorted(glob.glob(args.videos))
    baselines = sorted(glob.glob(args.baseline))[:args.n_baseline]
    rows = []
    for kind, paths in (("baseline", baselines), ("rollout", videos)):
        for p in paths:
            frames = read_frames(p, res, args.stride)
            if frames is None or len(frames) < 20:
                print(f"skip {p} (too few frames)")
                continue
            e = cycle_err(model, frames, obs_key, device, amp_dtype, amp_on, args.text_rows)
            early = float(np.mean(e[2:8]))
            late = float(np.mean(e[-max(3, len(e) // 4):]))
            name = "/".join(Path(p).parts[-2:])
            rows.append(dict(kind=kind, video=name, early=round(early, 2), late=round(late, 2),
                             ratio=round(late / max(early, 1e-6), 2), max=round(float(e.max()), 2)))
            print(f"{kind:>8} {name:<44} early={early:6.2f}  late={late:6.2f}  "
                  f"ratio={late / max(early, 1e-6):5.2f}  max={e.max():6.2f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "video", "early", "late", "ratio", "max"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
