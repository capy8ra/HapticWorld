"""Ablation entry: IMAGE + TORQUE on SIM data collected WITHOUT force feedback (claim 2b).

Same recipe as train_image_torque_sim.py, but on data/sim_training_wo_ft -- the 2026-07-24
collection where the operator drove the world model WITHOUT torque reflection to the leader.
Compare against it_sim (with-feedback data) to isolate the effect of force feedback on
demonstration quality. Torque input = tau_interaction_R (post-processed, same as it_sim).
Select the trained ckpt anywhere with the exp-prefix spec 'it_sim_wo_t'.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_core import run_training, BASE_CONFIG

if __name__ == "__main__":
    run_training({
        **BASE_CONFIG,
        "exp_prefix": "it_sim_wo_t",
        "wandb_project": "world_ft-act-ablation",
        "dataset_path": "../data/sim_training_wo_ft",
        "image_only": False,
        "use_angle": False,
        "use_torque": True,
    })
