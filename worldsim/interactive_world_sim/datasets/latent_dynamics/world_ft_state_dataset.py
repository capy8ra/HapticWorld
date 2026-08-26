"""Factored stage-2 dataset (NEW pipeline, isolated).

Extends ``WorldFtDynDataset`` with the command/state streams the factored robot
branch needs, WITHOUT modifying any existing file. The base replay-buffer
builder (and its on-disk cache format) only knows about action / joint_torque /
images, so this class uses a construct-then-augment strategy:

  1. ``super().__init__`` builds or loads the cached replay buffer as usual;
  2. the extra low-dim keys are then read straight from the episode HDF5s
     (same sorted episode order as the builder), appended to the in-memory
     zarr data group, and the ``SequenceSampler`` is rebuilt over all keys.

The extra arrays are tiny (4 keys x T x 8 float32), so re-reading them on
every startup costs well under a second and the cache file format stays
byte-identical with the old pipeline.

Batch additions (all raw physical units, (B, T, 8) float32):
  * ``qpos_right``   measured joint angle    [robot-branch state supervision]
  * ``joint_vel``    measured joint velocity
  * ``dq_des``       commanded velocity (mit_command:dq)
  * ``tau_ff``       recorded gravity/Coriolis feedforward (mit_command:tau);
                     the training target is tau_int = joint_torque - tau_ff
  * ``joint_torque`` measured torque (inherited from WorldFtDynDataset)

Normalizer addition: ``tau_int`` (range normalizer over train-split
``joint_torque - tau_ff``) -- used ONLY to normalize the scene-branch
conditioning input; supervision stays in raw N*m.

Requires episodes produced by ``scripts/data_collection/convert_world_ft_state.py``
(``data/world_ft_v4``); errors out with a pointer otherwise.
"""
import glob
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from interactive_world_sim.datasets.latent_dynamics.world_ft_dyn_dataset import (
    WorldFtDynDataset,
)
from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

EXTRA_KEYS = ["qpos_right", "joint_vel", "dq_des", "tau_ff"]


def _augment_replay_buffer(replay_buffer: ReplayBuffer, dataset_dir: str) -> None:
    """Load EXTRA_KEYS from the episode HDF5s into the in-memory replay buffer.

    Episode order replicates ``_convert_real_to_dp_replay`` exactly (sorted by
    integer episode index) so per-step alignment with the cached arrays holds.
    """
    paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    idxs = sorted(int(Path(p).stem.split("_")[-1]) for p in paths)
    data: Dict[str, List[np.ndarray]] = {k: [] for k in EXTRA_KEYS}
    for epi_idx in idxs:
        path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        with h5py.File(path, "r") as f:
            missing = [k for k in EXTRA_KEYS if k not in f["obs"]]
            if missing:
                raise KeyError(
                    f"{path} lacks obs/{missing}. The factored pipeline needs "
                    "episodes from scripts/data_collection/convert_world_ft_state.py "
                    "(e.g. data/world_ft_v4), not the old converter output."
                )
            for k in EXTRA_KEYS:
                data[k].append(f["obs"][k][()].astype(np.float32))
    for k in EXTRA_KEYS:
        arr = np.concatenate(data[k], axis=0)
        assert arr.shape[0] == replay_buffer.n_steps, (
            f"{k}: {arr.shape[0]} steps from HDF5 vs {replay_buffer.n_steps} in the "
            f"cached replay buffer for {dataset_dir}. The cache is stale -- delete "
            "cache*.zarr.zip and rerun."
        )
        if k in replay_buffer:
            continue
        replay_buffer.data.array(
            name=k, data=arr, shape=arr.shape, chunks=arr.shape,
            compressor=None, dtype=arr.dtype,
        )


class WorldFtStateDataset(WorldFtDynDataset):
    """WorldFtDynDataset + realized-state / command streams for the robot branch."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        train_dir = os.path.join(cfg.dataset_dir, "train")
        _augment_replay_buffer(self.replay_buffer, train_dir)
        # rebuild the sampler so the sampled sequences include the new keys
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.sampler.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask,
            goal_sample=self.goal_sample,
            keys=list(self.replay_buffer.keys()),
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )

    def get_validation_dataset(self) -> "WorldFtStateDataset":
        val_set = super().get_validation_dataset()
        val_dir = os.path.join(self.dataset_dir, "val")
        _augment_replay_buffer(val_set.replay_buffer, val_dir)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_set.train_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            keys=list(val_set.replay_buffer.keys()),
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        return val_set

    def get_normalizer(self, mode: str = "none", **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(mode, **kwargs)
        # range normalizer for the interaction torque -- ONLY for the scene-branch
        # conditioning input; the robot-branch targets stay raw N*m.
        tau = np.asarray(self.replay_buffer["joint_torque"])
        tff = np.asarray(self.replay_buffer["tau_ff"])
        normalizer["tau_int"] = get_range_normalizer_from_stat(
            array_to_stats(tau - tff)
        )
        # dedicated measured-state normalizer. Degenerate dims (range < 0.05 rad,
        # i.e. the unused gripper) pass through UN-scaled: the default range
        # normalizer would blow them up ~1000x, letting a physically meaningless
        # dim dominate any normalized-space state loss and poison conditioning.
        qpos = np.asarray(self.replay_buffer["qpos_right"])
        stat = array_to_stats(qpos)
        normalizer["qpos"] = get_range_normalizer_from_stat(
            stat, ignore_dim=(stat["max"] - stat["min"]) < 0.05
        )
        return normalizer

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        extras = {
            k: np.asarray(sample[k], dtype=np.float32) for k in EXTRA_KEYS
        }
        data = super()._sample_to_data(sample)
        for k, v in extras.items():
            data[k] = torch.from_numpy(v)
        return data
