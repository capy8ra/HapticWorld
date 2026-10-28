#!/usr/bin/env python
"""Download the HapticWorld assets from HuggingFace (ShaotingPeng/HapticWorld).

Per task (boxflip, wipe, microwave): the three world-model stages, the ACT policies, the
rollout seed frames, the in-simulator demonstration sets the policies were trained on, and
sample play data for world-model training.
Idempotent: re-run to fill in whatever is missing.

    python checkpoints/download_assets.py                    # everything, all tasks
    python checkpoints/download_assets.py --task wipe        # one task
    python checkpoints/download_assets.py --demo             # only what the no-robot rollout needs
    python checkpoints/download_assets.py --what wm,seeds    # a subset

Layout after download (relative to the repository root):
    checkpoints/wm/<task>/stage{1,2,3}/{checkpoints/last.ckpt, .hydra/config.yaml}
    checkpoints/policies/<task>/<task>_it_sim/ckpt_step_<N>/{model.pth, info.yaml}
    data/seeds/<task>/episode_*.hdf5
    data/demos/<task>/<episode>/...         demonstrations the policy was trained on
    data/play/<task>/{train,val}/           sample play data (10 episodes) for world-model training
    worldsim/.../metrics/i3d_torchscript.pt (FVD metric, training only)
"""
import argparse
import os
import sys
import zipfile

from huggingface_hub import hf_hub_download

REPO = "ShaotingPeng/HapticWorld"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from policy.tasks import TASKS  # noqa: E402

I3D_DST = os.path.join(ROOT, "worldsim", "hapticworld", "algorithms",
                       "common", "metrics", "i3d_torchscript.pt")
WHAT = ("wm", "policies", "seeds", "demos", "play", "metrics")


def fetch(path_in_repo, dst):
    if os.path.exists(dst):
        print(f"  [skip] {os.path.relpath(dst, ROOT)} (exists)")
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    got = os.path.realpath(hf_hub_download(REPO, path_in_repo))  # the cache path is a symlink to the blob
    try:
        os.link(got, dst)  # hard-link out of the HF cache when on the same filesystem
    except OSError:
        import shutil
        shutil.copy2(got, dst)
    if os.path.getsize(dst) != os.path.getsize(got):
        raise RuntimeError(f"incomplete copy of {path_in_repo} -> {dst}")
    print(f"  [ok]   {os.path.relpath(dst, ROOT)} ({os.path.getsize(dst) / 2**20:.1f} MB)")
    return dst


def fetch_zip(path_in_repo, extract_to, done_marker):
    """Download a zip and extract it into extract_to unless done_marker already exists."""
    if os.path.exists(done_marker) and os.listdir(done_marker):
        print(f"  [skip] {os.path.relpath(done_marker, ROOT)} (exists)")
        return
    os.makedirs(extract_to, exist_ok=True)
    zp = fetch(path_in_repo, os.path.join(extract_to, os.path.basename(path_in_repo)))
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(extract_to)
    os.remove(zp)
    print(f"  [ok]   extracted -> {os.path.relpath(done_marker, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", choices=list(TASKS), help="repeatable; default all")
    ap.add_argument("--what", default=",".join(WHAT), help=f"comma subset of {WHAT}")
    ap.add_argument("--demo", action="store_true",
                    help="only the no-robot rollout assets: stage 2+3, policies, seeds")
    a = ap.parse_args()
    tasks = a.task or list(TASKS)
    what = set(a.what.split(","))
    if a.demo:
        what &= {"wm", "policies", "seeds"}

    for task in tasks:
        print(f"== {task} ==")
        if "wm" in what:
            for stage in (1, 2, 3):
                if a.demo and stage == 1:
                    continue
                for f in ("checkpoints/last.ckpt", ".hydra/config.yaml"):
                    fetch(f"wm/{task}/stage{stage}/{f}",
                          os.path.join(ROOT, "checkpoints", "wm", task, f"stage{stage}", *f.split("/")))
        if "policies" in what:
            pol_root = os.path.join(ROOT, "checkpoints", "policies", task)
            for run, _ in TASKS[task]["policies"].values():
                fetch_zip(f"policies/{task}/{run}.zip", pol_root, os.path.join(pol_root, run))
        if "seeds" in what:
            seed_dir = os.path.join(ROOT, "data", "seeds", task)
            fetch_zip(f"seeds/{task}/seeds.zip", os.path.dirname(seed_dir), seed_dir)
        if "demos" in what:
            dst = os.path.join(ROOT, TASKS[task]["demos"])
            fetch_zip(f"demos/{task}.zip", os.path.dirname(dst), dst)
        if "play" in what:  # sample play data (10 episodes) for world-model training
            dst = os.path.join(ROOT, "data", "play", task)
            fetch_zip(f"play/{task}.zip", os.path.dirname(dst), dst)

    if "metrics" in what and not a.demo:
        print("== FVD metric model (training only) ==")
        fetch("metrics/i3d_torchscript.pt", I3D_DST)
    print("done.")


if __name__ == "__main__":
    main()
