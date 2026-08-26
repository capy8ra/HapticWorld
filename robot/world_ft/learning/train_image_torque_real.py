"""Ablation entry: IMAGE + TORQUE (torque only, no joint angle) on REAL data.

Torque input = follower_controller_state:tau_interaction_R (auto-selected by the dataset).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_core import run_training, BASE_CONFIG

if __name__ == "__main__":
    run_training({
        **BASE_CONFIG,
        "exp_prefix": "image_torque_real",
        "wandb_project": "world_ft-act-ablation",
        "dataset_path": "../data/real_training",
        "image_only": False,
        "use_angle": False,
        "use_torque": True,
    })
