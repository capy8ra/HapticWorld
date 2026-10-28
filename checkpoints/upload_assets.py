#!/usr/bin/env python
"""Maintainer script: upload local assets to HuggingFace (ShaotingPeng/HapticWorld).

Mirror of download_assets.py. Zips are built in a temp dir; optimizer.pth files are left out
of policy zips (not needed for evaluation).

    python checkpoints/upload_assets.py --task wipe --what wm,seeds,demos --dry-run
    python checkpoints/upload_assets.py --task wipe --what policies
"""
import argparse
import os
import sys
import tempfile
import zipfile

from huggingface_hub import HfApi

REPO = "ShaotingPeng/HapticWorld"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from policy.tasks import TASKS  # noqa: E402

I3D = os.path.join(ROOT, "worldsim", "hapticworld", "algorithms", "common", "metrics", "i3d_torchscript.pt")
WHAT = ("wm", "configs", "policies", "seeds", "demos", "play", "metrics")


def zip_dir(src_dir, zip_path, exclude=("optimizer.pth",)):
    """Zip src_dir so that extracting next to it recreates <basename(src_dir)>/..."""
    base = os.path.basename(os.path.normpath(src_dir))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d, _, files in os.walk(src_dir):
            for f in files:
                if f in exclude:
                    continue
                full = os.path.join(d, f)
                zf.write(full, os.path.join(base, os.path.relpath(full, src_dir)))
    return zip_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", choices=list(TASKS), help="repeatable; default all")
    ap.add_argument("--what", default=",".join(WHAT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delete", action="append", default=[], help="repeatable: path in the repo to delete")
    a = ap.parse_args()
    tasks = a.task or list(TASKS)
    what = set(a.what.split(","))
    api = HfApi()
    tmp = tempfile.mkdtemp(prefix="hapticworld_upload_")
    jobs = []  # (local path, path in repo)

    for task in tasks:
        if "wm" in what or "configs" in what:  # configs = only the small per-stage .hydra/config.yaml
            for stage in (1, 2, 3):
                for f in (("checkpoints/last.ckpt", ".hydra/config.yaml") if "wm" in what else (".hydra/config.yaml",)):
                    local = os.path.join(ROOT, "checkpoints", "wm", task, f"stage{stage}", *f.split("/"))
                    if os.path.exists(local):
                        jobs.append((local, f"wm/{task}/stage{stage}/{f}"))
        if "policies" in what:
            for run, _ in TASKS[task]["policies"].values():
                local = os.path.join(ROOT, "checkpoints", "policies", task, run)
                if os.path.isdir(local):
                    jobs.append((zip_dir(local, os.path.join(tmp, f"{run}.zip")), f"policies/{task}/{run}.zip"))
        if "seeds" in what:
            local = os.path.join(ROOT, "data", "seeds", task)
            if os.path.isdir(local):
                jobs.append((zip_dir(local, os.path.join(tmp, f"{task}_seeds.zip")), f"seeds/{task}/seeds.zip"))
        if "demos" in what:
            local = os.path.join(ROOT, TASKS[task]["demos"])
            if os.path.isdir(local):
                jobs.append((zip_dir(local, os.path.join(tmp, f"{task}_demos.zip")), f"demos/{task}.zip"))
        if "play" in what:
            local = os.path.join(ROOT, "data", "play", task)
            if os.path.isdir(local):
                jobs.append((zip_dir(local, os.path.join(tmp, f"{task}_play.zip"), exclude=("cache_right_qpos.zarr.zip", "cache_right_qpos.zarr.zip.lock")), f"play/{task}.zip"))
    if "metrics" in what and os.path.exists(I3D):
        jobs.append((I3D, "metrics/i3d_torchscript.pt"))

    for local, remote in jobs:
        print(f"{os.path.getsize(local)/2**20:8.1f} MB  {remote}")
        if not a.dry_run:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO)
    for remote in a.delete:
        print(f"  delete  {remote}")
        if not a.dry_run:
            api.delete_file(path_in_repo=remote, repo_id=REPO)
    print("dry run, nothing uploaded" if a.dry_run else f"uploaded {len(jobs)} files, deleted {len(a.delete)}")


if __name__ == "__main__":
    main()
