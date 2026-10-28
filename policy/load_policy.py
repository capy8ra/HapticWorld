"""Load a released or self-trained ACT policy from its model.pth (+ sibling info.yaml)."""
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root
from policy.act.model import ACT_Abs  # noqa: E402


def load_policy(model_path):
    """-> (model on cuda in eval mode, training config dict)."""
    with open(model_path.replace("model.pth", "info.yaml"), "r") as f:
        config = yaml.safe_load(f)["config"]
    assert not config.get("image_only", False), "image-only policies are not part of this release"
    # DETR's build function parses sys.argv; hide the calling script's flags during the build
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        act_model, _ = ACT_Abs.build_model_and_optimizer(
            n_obs_steps=config["obs_horizon"],
            action_chunk_size=config["action_horizon"],
            seed=config["seed"],
            arm_names=config["arms"],
            use_angle=config.get("use_angle", True),
            use_torque=config.get("use_torque", True),
            extra_obs_dim=2 * config.get("ink_keypoints_k", 0),
        )
    finally:
        sys.argv = saved_argv
    act_model.load_state_dict(torch.load(model_path))
    act_model.cuda()
    act_model.eval()
    return act_model, config
