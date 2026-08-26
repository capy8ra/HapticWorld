"""Make a compact COPY of a collection folder for upload (originals untouched).

The mp4 is already small (h264, ~1.3 MB/episode); ~90% of an episode is the 2 kHz
float64 low-dim streams. This writes <src>_upload/ with:
  * camera_head.mp4 copied as-is;
  * low_dim_npys decimated to --hz (default 100 -- still 10x above the 10-20 Hz that
    WM conversion and ACT training interpolate at), values cast to float32.
    Timestamps STAY float64 (float32 epoch seconds only resolve ~2 minutes).
    Streams sharing a source prefix (e.g. follower_right_state:*) are decimated with
    the SAME row indices so they stay aligned.

Typical shrink: ~14 MB/episode -> ~2 MB/episode. Then zip and upload, e.g.:
    cd world_ft/data && zip -r wipe_collecting_two_upload.zip wipe_collecting_two_upload
    modelscope upload <owner/dataset-repo> wipe_collecting_two_upload.zip --repo-type dataset

Env: any with numpy (hapticworld). Usage:
    python world_ft/prepare_upload.py --src world_ft/data/wipe_collecting_two
"""
import argparse
import os
import shutil

import numpy as np


def prepare_episode(ep_src, ep_dst, target_hz):
    os.makedirs(os.path.join(ep_dst, "low_dim_npys"), exist_ok=True)
    shutil.copy2(os.path.join(ep_src, "camera_head.mp4"),
                 os.path.join(ep_dst, "camera_head.mp4"))
    ld = os.path.join(ep_src, "low_dim_npys")
    files = sorted(os.listdir(ld))
    # group streams by source prefix; all fields of a source share row indices
    groups = {}
    for fn in files:
        groups.setdefault(fn.split(":")[0], []).append(fn)
    for prefix, fns in groups.items():
        ts_fn = f"{prefix}:timestamp.npy"
        ts = np.load(os.path.join(ld, ts_fn))
        t0 = ts[0, 0] if ts.ndim > 1 else ts[0]
        t1 = ts[-1, 0] if ts.ndim > 1 else ts[-1]
        native_hz = (len(ts) - 1) / max(float(t1 - t0), 1e-9)
        stride = max(1, int(round(native_hz / target_hz)))
        idx = np.arange(0, len(ts), stride)
        for fn in fns:
            a = np.load(os.path.join(ld, fn))
            a = a[idx] if len(a) == len(ts) else a  # scalar/odd streams kept whole
            if "timestamp" not in fn and a.dtype == np.float64:
                a = a.astype(np.float32)
            np.save(os.path.join(ep_dst, "low_dim_npys", fn), a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="collection folder (episode dirs inside)")
    ap.add_argument("--hz", type=float, default=100.0, help="low-dim target rate")
    ap.add_argument("--out", default="", help="default: <src>_upload")
    args = ap.parse_args()
    src = args.src.rstrip("/")
    out = args.out or (src + "_upload")
    eps = sorted(d for d in os.listdir(src)
                 if os.path.isdir(os.path.join(src, d, "low_dim_npys")))
    print(f"{len(eps)} episodes -> {out} (low-dim @ {args.hz:g} Hz, float32 values)")
    tot_src = tot_out = 0
    for i, ep in enumerate(eps):
        prepare_episode(os.path.join(src, ep), os.path.join(out, ep), args.hz)
        for root_dir, tot in ((src, "src"), (out, "out")):
            pass
        if (i + 1) % 25 == 0 or i == len(eps) - 1:
            print(f"  {i + 1}/{len(eps)}")
    du = lambda p: sum(os.path.getsize(os.path.join(r, f))
                       for r, _, fs in os.walk(p) for f in fs) / 1e6
    print(f"size: {du(src):.0f} MB -> {du(out):.0f} MB")


if __name__ == "__main__":
    main()
