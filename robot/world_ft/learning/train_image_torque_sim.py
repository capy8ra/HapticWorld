"""Ablation entry: IMAGE + TORQUE (torque only, no joint angle) on SIM (world-model) data.

Torque input = follower_controller_state:tau_interaction_R (= WM torque_pred_nm − gravity at
commanded q, written by world_ft/postprocess_sim_tau_interaction.py; auto-selected by
the dataset, same definition as real data). Run the post-process after every sim-collection
batch, BEFORE training.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_core import run_training, BASE_CONFIG

if __name__ == "__main__":
    run_training({
        **BASE_CONFIG,
        "exp_prefix": "image_torque_sim",
        "wandb_project": "world_ft-act-ablation",
        "dataset_path": "../data/sim_training",
        "image_only": False,
        "use_angle": False,
        "use_torque": True,
    })
