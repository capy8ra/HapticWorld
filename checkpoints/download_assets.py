#!/usr/bin/env python
"""Download every asset HapticWorld needs from HuggingFace (ShaotingPeng/HapticWorld):
world-model checkpoints, ACT policy checkpoints, rollout seed episodes, and the FVD
metric model. Idempotent -- re-run to fill in whatever is missing.

    python checkpoints/download_assets.py             # everything (~2.3 GB)
    python checkpoints/download_assets.py --demo      # only what the no-robot demo needs
"""
import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download

REPO = "ShaotingPeng/HapticWorld"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WM_FILES = [f"wm/boxflip/{s}/{f}" for s in ("stage1", "stage2_trunk", "stage3")
            for f in ("checkpoints/last.ckpt", ".hydra/config.yaml")]
POLICY_ZIPS = [
    "policies/boxflip/image_only_sim_2026_07_22_11_14_57_PDT.zip",
    "policies/boxflip/image_torque_sim_2026_07_22_15_08_29_PDT.zip",
    "policies/boxflip/it_sim_wo_t_2026_07_24_20_09_01_PDT.zip",
]
SEEDS_ZIP = "seeds/boxflip/wm_seed_live.zip"
I3D = "metrics/i3d_torchscript.pt"
I3D_DST = os.path.join(ROOT, "worldsim", "interactive_world_sim", "algorithms",
                       "common", "metrics", "i3d_torchscript.pt")


def fetch(path_in_repo: str, dst: str) -> str:
    if os.path.exists(dst):
        print(f"  [skip] {dst} (exists)")
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    got = hf_hub_download(REPO, path_in_repo)
    # hard-link out of the HF cache when possible, else copy
    try:
        os.link(got, dst)
    except OSError:
        import shutil
        shutil.copy2(got, dst)
    print(f"  [ok]   {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="only the no-robot demo assets (WM stages 2+3, policies, seeds)")
    a = ap.parse_args()

    print("== world model ==")
    for f in WM_FILES:
        if a.demo and "stage1" in f:
            continue  # the demo only needs stage2_trunk (dynamics+torque) and stage3 (decoder)
        fetch(f, os.path.join(ROOT, "checkpoints", "wm", "boxflip",
                              f.split("wm/boxflip/")[1]))

    print("== ACT policies ==")
    pol_root = os.path.join(ROOT, "checkpoints", "policies")
    for z in POLICY_ZIPS:
        run = os.path.splitext(os.path.basename(z))[0]
        if os.path.isdir(os.path.join(pol_root, run)):
            print(f"  [skip] {run} (exists)")
            continue
        zp = fetch(z, os.path.join(pol_root, os.path.basename(z)))
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(pol_root)
        os.remove(zp)
        print(f"  [ok]   extracted {run}")

    print("== rollout seeds ==")
    seed_dir = os.path.join(ROOT, "data", "wm_seed_live")
    if os.path.isdir(seed_dir) and os.listdir(seed_dir):
        print(f"  [skip] {seed_dir} (exists)")
    else:
        zp = fetch(SEEDS_ZIP, os.path.join(ROOT, "data", "wm_seed_live.zip"))
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(os.path.join(ROOT, "data"))
        os.remove(zp)
        print(f"  [ok]   {seed_dir}")

    if not a.demo:
        print("== FVD metric model (training only) ==")
        fetch(I3D, I3D_DST)

    print("done.")


if __name__ == "__main__":
    main()
