"""Whiteboard wipe, it_sim: image + torque ACT trained on demonstrations collected inside HapticWorld with force feedback on.

Data: demonstrations collected with the operator feeling the real follower's contact (conducted rig) (fetched by checkpoints/download_assets.py --task wipe).
Observation = one head-camera frame + the 8-D interaction torque (tau_interaction_R). Training
stops at step 2000, the checkpoint the paper's evaluations used; checkpoints go to
checkpoints/policies/wipe/wipe_it_sim_<timestamp>/ and are selected as "wipe/wipe_it_sim".

Run from the repository root in the hapticworld env:
    python policy/train_wipe_it_sim.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root
from policy.train_core import run_training, BASE_CONFIG  # noqa: E402
from policy.tasks import TASKS, abs_path  # noqa: E402

TASK = "wipe"

if __name__ == "__main__":
    run_training({
        **BASE_CONFIG,
        "task": TASK,
        "exp_prefix": "wipe_it_sim",
        "wandb_project": "hapticworld-act",
        "dataset_path": abs_path(TASKS[TASK]["demos"]),
        "crop": TASKS[TASK]["crop"],
        "image_only": False,
        "use_angle": False,
        "use_torque": True,
        "save_every_n_steps": 1000,
        "max_steps": TASKS[TASK]["train_stop_step"],
    })
