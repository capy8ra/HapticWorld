"""Derive WM rollout seeds from RECORDED wipe episodes -- no live capture session.

A seed is exactly what capture_wm_seeds.py wrote live for the box task: ONE image
frame + ONE 8-DoF action row (all seed_from_episode reads). Every wipe episode's first
frame IS a valid seed (drawn board, arm at home), so seeds are free:

    python scripts/data_collection/make_wm_seeds_from_episodes.py \
        --src <collection folder (raw episode dirs, full or _upload copy)> \
        --out data/wm_seed_wipe [--n 20] [--ink-enhance]

Picks --n episodes evenly spread across the folder (drawing variety), decodes each
first video frame, pairs it with the first commanded right-arm q, and writes
episode_N.hdf5. --ink-enhance paints the detected drawing red -- REQUIRED iff the WM
was trained on enhanced frames (--ink-enhance replace/both at conversion): the seed
stream must match the training stream.
"""
import argparse
import os
import sys

import av
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "world_ft"))

OBS_KEY = "camera_1_color"  # the WM's obs_keys[0] (or camera_2_color training via override)


def first_frame_rgb(video_path):
    with av.open(video_path) as container:
        for frame in container.decode(container.streams.video[0]):
            return frame.to_ndarray(format="rgb24")
    raise ValueError(f"no frames in {video_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="collection folder with *_PDT episode dirs")
    ap.add_argument("--out", default="data/wm_seed_wipe")
    ap.add_argument("--n", type=int, default=20, help="number of seeds (evenly spread)")
    ap.add_argument("--ink-enhance", action="store_true",
                    help="paint detected ink red (match a WM trained on enhanced frames)")
    args = ap.parse_args()

    eps = sorted(d for d in os.listdir(args.src)
                 if os.path.isdir(os.path.join(args.src, d, "low_dim_npys")))
    assert eps, f"no episodes under {args.src}"
    picks = [eps[i] for i in np.linspace(0, len(eps) - 1, min(args.n, len(eps))).astype(int)]

    enhance = None
    if args.ink_enhance:
        from ink_detect import enhance_ink
        enhance = enhance_ink

    os.makedirs(args.out, exist_ok=True)
    for i, ep in enumerate(picks):
        epd = os.path.join(args.src, ep)
        frame = first_frame_rgb(os.path.join(epd, "camera_head.mp4"))
        if enhance is not None:
            frame = enhance(frame, bgr=False)
        q8 = np.load(os.path.join(epd, "low_dim_npys",
                                  "follower_right_mit_command:q.npy"))[0]
        out = os.path.join(args.out, f"episode_{i}.hdf5")
        with h5py.File(out, "w") as f:
            f.create_dataset(f"obs/images/{OBS_KEY}", data=frame[None].astype(np.uint8))
            f.create_dataset("action", data=np.asarray(q8, np.float32)[None])
        print(f"episode_{i}.hdf5 <- {ep}")
    print(f"{len(picks)} seeds -> {args.out} (ink_enhance={args.ink_enhance})")


if __name__ == "__main__":
    main()
