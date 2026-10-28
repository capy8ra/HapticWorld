"""Build the static-scene background for dynamic-pixel loss weighting (wipe task).

The camera is fixed, so a pixelwise MEDIAN over frames sampled across the whole
train split converges to the static scene (board, tape, glare, furniture): the
arm/eraser move too much to survive the median, and drawings occupy few pixels
in few episodes each.

Output (in --data_dir):
    background_128.npy    (128, 128, 3) float32 in [0, 1], the LOADER's view
                          (center square crop -> 128) so it aligns pixel-for-pixel
                          with what stage-1 training sees
    background_preview.png              the background + example weight masks

Usage:
    python scripts/data_collection/make_static_background.py \
        --data_dir ../data/play/wipe [--stride 20] [--thresh 0.10]
"""
import argparse
import glob
import os

import cv2
import h5py
import numpy as np


def loader_view(frame):
    """Center square crop -> 128, mirroring the replay-buffer conversion."""
    h, w = frame.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return cv2.resize(frame[top:top + s, left:left + s], (128, 128))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/play/wipe")
    ap.add_argument("--stride", type=int, default=20, help="sample every k-th frame")
    ap.add_argument("--thresh", type=float, default=0.10,
                    help="mask threshold on mean |frame-bg| (preview only)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "train", "episode_*.hdf5")))
    assert files, f"no episodes under {args.data_dir}/train"
    samples = []
    for p in files:
        with h5py.File(p, "r") as f:
            imgs = f["obs/images/camera_1_color"]
            for i in range(0, imgs.shape[0], args.stride):
                samples.append(loader_view(imgs[i]))
    stack = np.stack(samples, 0).astype(np.float32) / 255.0  # (N, 128, 128, 3)
    print(f"{len(files)} episodes -> {stack.shape[0]} sampled frames")
    bg = np.median(stack, axis=0)  # (128, 128, 3) in [0, 1]

    out_npy = os.path.join(args.data_dir, "background_128.npy")
    np.save(out_npy, bg.astype(np.float32))
    print("wrote", out_npy)

    # preview: background + weight masks for a few frames spread across the set
    rows = [np.hstack([bg, bg])]  # top row: background twice (placeholder layout)
    idxs = np.linspace(0, stack.shape[0] - 1, 4).astype(int)
    for i in idxs:
        fr = stack[i]
        diff = np.abs(fr - bg).mean(-1)
        mask = np.clip((diff - args.thresh) / 0.05, 0, 1)
        overlay = fr.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], mask)  # dynamic pixels -> red
        rows.append(np.hstack([fr, overlay]))
    sheet = (np.vstack(rows) * 255).astype(np.uint8)
    sheet = cv2.resize(sheet, (sheet.shape[1] * 3, sheet.shape[0] * 3),
                       interpolation=cv2.INTER_NEAREST)
    out_png = os.path.join(args.data_dir, "background_preview.png")
    cv2.imwrite(out_png, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", out_png)
    cov = [float(np.clip((np.abs(stack[i] - bg).mean(-1) - args.thresh) / 0.05, 0, 1).mean())
           for i in idxs]
    print("dynamic-pixel fraction in preview frames:", [f"{c:.3f}" for c in cov])


if __name__ == "__main__":
    main()
