"""Ablation entry: IMAGE + TORQUE on REAL data, with SHARPNESS AUGMENTATION.

Same as train_image_torque_real.py, but 50% of samples get the image downsampled to 128 and
back (the deploy --match-blur-res op), making the policy sharpness-invariant: it runs on the
sharp real camera AND on soft world-model renders (in-simulator eval via
rollout_policy_in_wm.py -- see docs/WM_EVAL.md). Deploy with --match-blur-res 0 or 128.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_core import run_training, BASE_CONFIG

if __name__ == "__main__":
    run_training({
        **BASE_CONFIG,
        "exp_prefix": "image_torque_real_blur",
        "wandb_project": "world_ft-act-ablation",
        "dataset_path": "../data/real_training",
        "image_only": False,
        "use_angle": False,
        "use_torque": True,
        "blur_aug_res": 128,
        "blur_aug_p": 0.5,
    })
