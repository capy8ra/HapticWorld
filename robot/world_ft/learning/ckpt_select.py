"""Checkpoint selection shared by eval_inference_worker.py and eval_controller.py.

Resolves a short spec to the LATEST matching checkpoint under model_ckpts/:

    variant:  'i'    image only              (image_only=True)
              'it'   image + torque          (use_angle=False, use_torque=True)
              'its'  image + state + torque  (use_angle=True,  use_torque=True)
    optional training-data domain suffix:  '_real' | '_sim'

e.g. 'it_sim' = latest image+torque trained on sim_training; bare 'it' = latest image+torque
from EITHER domain. Matching is by each run's ckpt_latest/info.yaml config (flags +
dataset_path), not the folder name, so old and new exp_prefix naming schemes both work;
"latest" = newest model.pth mtime. Paths are repo-root-relative like the rest of the
project (run from HapticWorld/robot).
"""
import glob
import os

import numpy as np
import yaml

CKPT_ROOT = "../data/model_ckpts"

_VARIANTS = {
    "i": dict(image_only=True),
    "it": dict(image_only=False, use_angle=False, use_torque=True),
    "its": dict(image_only=False, use_angle=True, use_torque=True),
}


def _load_config(info_path):
    with open(info_path, "r") as f:
        return yaml.safe_load(f)["config"]


def _matches(cfg, want):
    # .get defaults mirror the training-side defaults (old ckpts may omit keys)
    image_only = cfg.get("image_only", False)
    if want["image_only"]:
        return image_only
    return (not image_only
            and cfg.get("use_angle", True) == want["use_angle"]
            and cfg.get("use_torque", True) == want["use_torque"])


def _parse_spec(spec):
    """'it_sim' -> ('it', 'sim'); 'it' -> ('it', None); not a spec -> (None, None)."""
    variant, _, domain = spec.partition("_")
    if variant in _VARIANTS and (domain in ("real", "sim") or not domain):
        return variant, domain or None
    return None, None


def is_spec(spec):
    return _parse_spec(spec)[0] is not None


def resolve_ckpt(spec, root=CKPT_ROOT):
    """Resolve a checkpoint spec, in precedence order:
    1. variant spec 'i[_real|_sim]' / 'it[...]' / 'its[...]' -> newest matching run
       (domain matches the dataset dir BASENAME exactly, so e.g. sim_training_wo_ft
       runs do NOT hijack the '_sim' specs);
    2. an existing model.pth path;
    3. an exp_prefix, e.g. 'it_sim_wo_t' -> newest model_ckpts/it_sim_wo_t_*/ckpt_latest.
    Returns (model_path, config)."""
    variant, domain = _parse_spec(spec)
    if variant is None:
        if os.path.exists(spec):
            return spec, _load_config(spec.replace("model.pth", "info.yaml"))
        prefixed = []
        for info_path in glob.glob(os.path.join(root, f"{spec}_*", "ckpt_latest", "info.yaml")):
            model_path = os.path.join(os.path.dirname(info_path), "model.pth")
            if os.path.isfile(model_path):
                prefixed.append((os.path.getmtime(model_path), model_path, _load_config(info_path)))
        if prefixed:
            _, model_path, cfg = max(prefixed)
            return model_path, cfg
        raise FileNotFoundError(
            f"'{spec}' is not a variant spec, an existing ckpt path, or an exp_prefix under {root}")
    candidates = []
    for info_path in glob.glob(os.path.join(root, "*", "ckpt_latest", "info.yaml")):
        model_path = os.path.join(os.path.dirname(info_path), "model.pth")
        if not os.path.isfile(model_path):
            continue
        cfg = _load_config(info_path)
        if _matches(cfg, _VARIANTS[variant]) and (
                domain is None or os.path.basename(
                    os.path.normpath(cfg.get("dataset_path", ""))) == f"{domain}_training"):
            candidates.append((os.path.getmtime(model_path), model_path, cfg))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching '{spec}' under {root}")
    _, model_path, cfg = max(candidates)
    return model_path, cfg


def pick_reference_episode(dataset_path):
    """Home-reference episode for eval_controller: the episode whose FIRST commanded q is
    closest to the dataset-mean start pose (the PIPELINE §5 rule). Returns its low_dim_npys
    dir, ready for reference_episode_lowdim_path."""
    starts = []
    for q_path in sorted(glob.glob(os.path.join(
            dataset_path, "*", "low_dim_npys", "follower_right_mit_command:q.npy"))):
        starts.append((os.path.dirname(q_path), np.load(q_path)[0]))
    if not starts:
        raise FileNotFoundError(
            f"No episodes with follower_right_mit_command:q.npy under {dataset_path}")
    mean_start = np.mean([q0 for _, q0 in starts], axis=0)
    lowdim_dir, _ = min(starts, key=lambda x: float(np.linalg.norm(x[1] - mean_start)))
    return lowdim_dir


if __name__ == "__main__":
    for variant in ("i", "it", "its"):
        for suffix in ("", "_real", "_sim"):
            spec = variant + suffix
            try:
                path, cfg = resolve_ckpt(spec)
                print(f"{spec:>8} -> {path}  (data: {cfg.get('dataset_path', '?')})")
            except FileNotFoundError:
                print(f"{spec:>8} -> (none)")
