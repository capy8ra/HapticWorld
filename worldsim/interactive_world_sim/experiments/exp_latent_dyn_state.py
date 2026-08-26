"""State-conditioned stage-2 experiment (NEW pipeline, isolated).

Registers the state-conditioned algorithm with the state dataset; nothing in
the existing experiments is modified.
"""
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model_state import (
    LatentWorldModelState,
)
from interactive_world_sim.datasets.latent_dynamics.world_ft_state_dataset import (
    WorldFtStateDataset,
)

from .exp_latent_dyn import LatentDynExperiment


class LatentDynStateExperiment(LatentDynExperiment):
    """Latent dynamics + trunk torque + autoregressive state branch."""

    compatible_algorithms = dict(
        LatentDynExperiment.compatible_algorithms,
        latent_world_model_state=LatentWorldModelState,
    )

    compatible_datasets = dict(
        LatentDynExperiment.compatible_datasets,
        world_ft_state_dataset=WorldFtStateDataset,
    )
