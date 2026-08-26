"""Ablation entry: IMAGE ONLY (no state, no torque) on SIM (world-model) data."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_core import run_training, BASE_CONFIG

if __name__ == "__main__":
    run_training({
        **BASE_CONFIG,
        "exp_prefix": "image_only_sim",
        "wandb_project": "world_ft-act-ablation",
        "dataset_path": "../data/sim_training",
        "image_only": True,
    })
