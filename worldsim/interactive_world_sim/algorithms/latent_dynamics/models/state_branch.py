"""Deterministic autoregressive state branch (NEW pipeline, isolated).

Predicts the measured joint state sequence for the state-conditioned stage-2
(``latent_world_model_state.py``): within each training window the branch is
unrolled AUTOREGRESSIVELY (its own predictions feed back), the loss is the
mean MSE over every predicted frame of the chunk, and gradients flow through
the whole unroll (BPTT; the trainer's global gradient_clip_val=1.0 covers the
resulting gradient-explosion risk).

One step (frame t -> t+1, 10 Hz), all quantities in NORMALIZED units (the
action range normalizer is reused for the measured state -- same units and
nearly identical range as q_des):

    x_t     = [s_hat_t, a_{t+1}, phi(z_t)]
    h_{t+1} = GRU(x_t, h_t)
    s_hat_{t+1} = s_hat_t + head(h_{t+1})      # residual: persistence prior

The residual head is zero-initialized, so at init the branch predicts
"state stays where it was" -- the correct 10 Hz prior and a stable starting
point for the BPTT unroll.
"""
from typing import Optional, Tuple

import torch
from torch import nn

from interactive_world_sim.algorithms.latent_dynamics.models.robot_branch import (
    LatentPool,
)


class StateBranch(nn.Module):
    """Latent + action + previous state -> next state, unrolled with a GRU."""

    def __init__(
        self,
        latent_dim: int = 4,
        state_dim: int = 8,
        action_dim: int = 8,
        feat_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        out_clamp: float = 3.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # SMOOTH bound on the (normalized) predicted state:
        # s <- out_clamp * tanh(s / out_clamp). Long unrolls stay bounded
        # (divergence/NaN-proof) while the gradient is nonzero EVERYWHERE --
        # a hard torch.clamp has zero gradient outside the range, so any early
        # training transient that pushes outputs past the bound gets stuck
        # there permanently (constant saturated predictions). Data lives in
        # [-1, 1]; at |s|=1 the bound distorts by only ~3.5%.
        self.out_clamp = out_clamp
        self.pool = LatentPool(latent_dim, feat_dim)
        self.gru = nn.GRU(
            input_size=state_dim + action_dim + feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, state_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def step(
        self,
        s_prev: torch.Tensor,     # (B, state_dim) normalized current state
        a_next: torch.Tensor,     # (B, action_dim) normalized command of the step
        scene_feat: torch.Tensor,  # (B, feat_dim) pooled latent feature phi(z_t)
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One transition; returns (s_next, h)."""
        x = torch.cat([s_prev, a_next, scene_feat], dim=-1)
        y, h = self.gru(x.unsqueeze(1), h)
        s_next = s_prev + self.head(y.squeeze(1))
        return self.out_clamp * torch.tanh(s_next / self.out_clamp), h
