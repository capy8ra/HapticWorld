#!/usr/bin/env python
"""Box-sharpness realism score for WM rollout videos (pure CPU, no model).

Quantifies the eyeball test "the blue box keeps a sharp edge and its shape": per frame,
segment the blue box (HSV threshold -> largest component), then measure
  * edge sharpness: variance of the Laplacian in the box's edge band (dilate minus erode),
  * mean saturation inside the box (smearing desaturates),
  * mask solidity (area / convex hull area; a bleeding box gets ragged),
  * box area (tracked to catch fade-out; occluded frames are excluded, not penalized).
Frames whose box area falls below 25% of the video's own p90 area are treated as occluded
and skipped. Everything is aggregated as medians and self-normalized late/early ratios, so
different seed scenes stay comparable. A degraded rollout: sharp_ratio << 1, sat_ratio < 1,
low valid_late (box faded). Healthy: ratios near 1.

Usage (any env with cv2+numpy, from HapticWorld/worldsim):
    python scripts/inference/score_box_sharpness.py --videos 'wm_policy_rollouts/*/rollout_*.mp4'
    # segmentation sanity check (writes <out-dir>/<video>_vis.png contact sheets):
    python scripts/inference/score_box_sharpness.py --videos '...' --debug-vis 2
"""
import argparse
import csv
import glob
from pathlib import Path

import cv2
import numpy as np


def read_frames(path, res):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (res[1], res[0]), interpolation=cv2.INTER_AREA))  # BGR
    cap.release()
    return frames


def box_mask(frame_bgr, hsv_lo, hsv_hi):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros_like(m), 0
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8) * 255
    return mask, int(stats[biggest, cv2.CC_STAT_AREA])


def frame_stats(frame_bgr, mask):
    k = np.ones((5, 5), np.uint8)
    band = cv2.dilate(mask, k) & ~cv2.erode(mask, k)
    if band.sum() < 50 * 255:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = float(lap[band > 0].var())
    sat = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[..., 1][mask > 0].mean())
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = cv2.contourArea(cv2.convexHull(np.vstack(cnts)))
    solidity = float((mask > 0).sum() / max(hull_area, 1.0))
    return sharp, sat, solidity


def analyze(path, res, hsv_lo, hsv_hi, vis_path=None, window=None):
    frames = read_frames(path, res)
    if len(frames) < 20:
        return None
    masks, areas = zip(*[box_mask(f, hsv_lo, hsv_hi) for f in frames])
    areas = np.array(areas, float)
    min_area = max(80.0, 0.25 * np.percentile(areas[areas > 0], 90)) if (areas > 0).any() else 1e9

    rows = []  # (t, sharp, sat, solidity, area)
    for t, (f, m, a) in enumerate(zip(frames, masks, areas)):
        if a < min_area:
            continue
        st = frame_stats(f, m)
        if st is not None:
            rows.append((t, *st, a))
    if len(rows) < 10:
        return dict(valid=len(rows), error="box not found in enough frames")

    arr = np.array(rows)
    if window is None:
        n_early = min(8, len(arr) // 3)
        early, late = arr[1:1 + n_early], arr[-max(3, len(arr) // 4):]
        span_lo, span_hi = len(frames) * 3 // 4, len(frames) - 1
    else:  # baseline = valid frames BEFORE the window (skipping the first 2 s); late = the window
        lo, hi = window
        early = arr[(arr[:, 0] >= 20) & (arr[:, 0] < lo)]
        late = arr[(arr[:, 0] >= lo) & (arr[:, 0] <= hi)]
        span_lo, span_hi = lo, min(hi, len(frames) - 1)
        if len(early) < 5 or len(late) < 3:
            return dict(valid=len(arr),
                        error=f"too few usable frames (baseline={len(early)}, window={len(late)})")
    n_in_span = int(((arr[:, 0] >= span_lo) & (arr[:, 0] <= span_hi)).sum())
    valid_late = n_in_span / max(1, span_hi - span_lo + 1)  # usable-box fraction in the span
    out = dict(
        valid=len(arr),
        sharp_early=round(float(np.median(early[:, 1])), 1),
        sharp_late=round(float(np.median(late[:, 1])), 1),
        sharp_ratio=round(float(np.median(late[:, 1]) / max(np.median(early[:, 1]), 1e-6)), 2),
        sat_ratio=round(float(np.median(late[:, 2]) / max(np.median(early[:, 2]), 1e-6)), 2),
        solidity_p10=round(float(np.percentile(arr[:, 3], 10)), 2),
        area_ratio=round(float(np.median(late[:, 4]) / max(np.median(early[:, 4]), 1e-6)), 2),
        valid_late=round(float(np.clip(valid_late, 0, 1)), 2),
    )

    if vis_path is not None:  # contact sheet: frames + mask contour + per-frame sharpness
        picks = [int(p * (len(frames) - 1)) for p in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        tiles = []
        for t in picks:
            f = frames[t].copy()
            cnts, _ = cv2.findContours(masks[t], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(f, cnts, -1, (0, 0, 255), 1)
            st = frame_stats(frames[t], masks[t]) if areas[t] >= min_area else None
            label = f"t={t} " + (f"sh={st[0]:.0f}" if st else "occ/none")
            f = cv2.resize(f, (res * 2, res * 2), interpolation=cv2.INTER_NEAREST)
            cv2.putText(f, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(f)
        cv2.imwrite(str(vis_path), np.concatenate(tiles, axis=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="wm_policy_rollouts/*/rollout_*.mp4")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=[95, 60, 40])
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=[135, 255, 255])
    ap.add_argument("--debug-vis", type=int, default=0,
                    help="write mask-overlay contact sheets for the first N videos")
    ap.add_argument("--window-s", type=float, nargs=2, default=None, metavar=("START", "END"),
                    help="score ONLY this time window (seconds) against a pre-window baseline, "
                         "e.g. --window-s 11 17 for the flip phase")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default="box_sharpness_scores.csv")
    args = ap.parse_args()

    window = None
    if args.window_s is not None:
        window = (int(args.window_s[0] * args.fps), int(args.window_s[1] * args.fps))
        print(f"scoring window: {args.window_s[0]:g}-{args.window_s[1]:g}s "
              f"(frames {window[0]}-{window[1]}) vs pre-window baseline")

    videos = sorted(glob.glob(args.videos))
    rows = []
    for i, p in enumerate(videos):
        name = "/".join(Path(p).parts[-2:])
        vis = Path(p).with_suffix(".seg_vis.png") if i < args.debug_vis else None
        r = analyze(p, args.res, args.hsv_lo, args.hsv_hi, vis_path=vis, window=window)
        if r is None:
            print(f"skip {name} (too few frames)")
            continue
        r["video"] = name
        rows.append(r)
        if "error" in r:
            print(f"{name:<44} INVALID: {r['error']} (valid frames: {r['valid']})")
        else:
            print(f"{name:<44} sharp {r['sharp_early']:6.1f} -> {r['sharp_late']:6.1f} "
                  f"(ratio {r['sharp_ratio']:5.2f}) | sat_ratio {r['sat_ratio']:5.2f} | "
                  f"solidity_p10 {r['solidity_p10']:4.2f} | area_ratio {r['area_ratio']:5.2f} | "
                  f"valid_late {r['valid_late']:4.2f}")

    keys = ["video", "valid", "sharp_early", "sharp_late", "sharp_ratio", "sat_ratio",
            "solidity_p10", "area_ratio", "valid_late", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nsaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
