"""Checkpoint selection shared by the in-simulator rollout and the real-robot eval scripts.

A policy is named ``<task>/<policy>``, for example ``wipe/it_sim``. The policy part is
  * the released policy of that task, ``it_sim`` (see tasks.py), or
  * the exp_prefix of a run you trained yourself: ``wipe/wipe_it_sim`` resolves to the newest
    ``checkpoints/policies/wipe/wipe_it_sim_*/ckpt_latest/model.pth``.
An explicit path to a ``model.pth`` is also accepted everywhere a spec is.
"""
import glob
import os

import numpy as np
import yaml

from policy.tasks import TASKS, abs_path, demo_dir, policy_path, split_spec


def _load_config(info_path):
    with open(info_path, "r") as f:
        return yaml.safe_load(f)["config"]


def is_spec(spec):
    """True for '<task>/<policy>' names (as opposed to file paths)."""
    if not spec or os.path.exists(spec):
        return False
    try:
        split_spec(spec)
        return True
    except ValueError:
        return False


def resolve_ckpt(spec, root=None):
    """Resolve a policy spec or model.pth path -> (model_path, training config)."""
    if os.path.isfile(spec):
        return spec, _load_config(spec.replace("model.pth", "info.yaml"))
    task, name = split_spec(spec)
    if name in TASKS[task]["policies"]:
        model_path = policy_path(task, name)
        return model_path, _load_config(model_path.replace("model.pth", "info.yaml"))
    # a run trained with hapticworld/learning/train_*.py: newest <exp_prefix>_<timestamp> run
    root = root or abs_path(os.path.join("checkpoints", "policies", task))
    runs = []
    for info_path in glob.glob(os.path.join(root, f"{name}_*", "ckpt_latest", "info.yaml")):
        model_path = os.path.join(os.path.dirname(info_path), "model.pth")
        if os.path.isfile(model_path):
            runs.append((os.path.getmtime(model_path), model_path, _load_config(info_path)))
    if not runs:
        raise FileNotFoundError(
            f"{spec!r}: not a released policy of task {task!r} ({list(TASKS[task]['policies'])}) "
            f"and no run named {name}_* under {root}")
    _, model_path, cfg = max(runs)
    return model_path, cfg


def pick_reference_episode(dataset_path):
    """Home-reference episode for the real-robot eval controller: the training episode whose
    first commanded joint position is closest to the dataset-mean start pose. Returns its
    low_dim_npys directory."""
    dataset_path = demo_dir(dataset_path)
    starts = []
    for q_path in sorted(glob.glob(os.path.join(
            dataset_path, "*", "low_dim_npys", "follower_right_mit_command:q.npy"))):
        starts.append((os.path.dirname(q_path), np.load(q_path)[0]))
    if not starts:
        raise FileNotFoundError(
            f"No episodes with follower_right_mit_command:q.npy under {dataset_path} "
            "(download the demo sets: python checkpoints/download_assets.py)")
    mean_start = np.mean([q0 for _, q0 in starts], axis=0)
    lowdim_dir, _ = min(starts, key=lambda x: float(np.linalg.norm(x[1] - mean_start)))
    return lowdim_dir


if __name__ == "__main__":
    for task, t in TASKS.items():
        for name in t["policies"]:
            spec = f"{task}/{name}"
            try:
                path, cfg = resolve_ckpt(spec)
                print(f"{spec:>22} -> {path}")
            except FileNotFoundError as e:
                print(f"{spec:>22} -> (missing) {e}")
