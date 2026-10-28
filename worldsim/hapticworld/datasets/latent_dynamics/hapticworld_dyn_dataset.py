"""Stage-2 dynamics dataset (NEW, isolated).

Subclass of ``PlayDataset`` that exposes the right-arm joint-torque target
for stage-2 training. Use with ``action_mode=right_qpos`` (the base loader then
returns the raw 8-DoF ``q_des`` as the action) on a dataset whose episodes
contain ``obs/joint_torque`` (8,), as produced by the unified converter.

Adds, on top of the base dataset:
  * ``batch["joint_torque"]``            -> (B, T, 8), raw N·m

Torque is intentionally NOT normalized: the model predicts and is supervised in
physical N·m so the predictions are directly usable as actual torque downstream,
with no normalizer statistics required. (Images and actions are still normalized
by the base class as before.)

The base ``PlayDataset`` replay-buffer builder already loads ``joint_torque``
into the buffer when present (small guarded addition); everything else lives
here so the base class behavior is otherwise untouched.
"""
from typing import Dict

import numpy as np
import torch

from hapticworld.datasets.latent_dynamics.play_dataset import (
    PlayDataset,
)
from hapticworld.utils.normalizer import LinearNormalizer


class HapticWorldDynDataset(PlayDataset):
    """PlayDataset + per-frame right-arm joint-torque target (raw N·m)."""

    def get_normalizer(self, mode: str = "none", **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(mode, **kwargs)
        assert "joint_torque" in self.replay_buffer.keys(), (
            "HapticWorldDynDataset requires `obs/joint_torque` in the episodes; "
            "regenerate the dataset with the unified converter."
        )
        return normalizer

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        # read before super() consumes/deletes its own keys (joint_torque is not
        # one of them, but be defensive)
        torque = np.asarray(sample["joint_torque"], dtype=np.float32)  # (T, 8)
        data = super()._sample_to_data(sample)
        data["joint_torque"] = torch.from_numpy(torque)
        return data
