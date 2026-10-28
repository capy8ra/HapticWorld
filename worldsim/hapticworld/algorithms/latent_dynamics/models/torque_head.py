"""Torque head of HapticWorld.

A per-frame MLP that reads the latent dynamics U-Net's mid-block features (the widest, most
action-conditioned and causally time-filtered representation in the trunk) and maps them to the
8 joint torques in N·m. The features are captured by a forward hook during the regular dynamics
forward, so predicting torque costs no extra network pass.
"""
import torch
from torch import nn


class TrunkTorqueHead(nn.Module):
    """Per-frame torque head that branches off the dynamics U-Net trunk.

    Consumes the spatially-pooled features produced *inside* ``CMLatentDynamics``
    after its spatio-temporal attention (the mid-block bottleneck). Those features
    are already causally temporal-attended, so a per-frame MLP is enough (no GRU).

    Because the trunk runs on *noisy* latents (terminal frame ~ pure noise under
    ``terminal_only`` sampling, context frames near-clean via
    ``prev_frame_noise_scale``), the head is optionally conditioned on the
    per-frame diffusion noise level so it can normalize across the noise regime it
    sees in training (and again at inference, where the same sampler is reused).

    Input
    -----
    feat        : (B, T, feat_dim)   spatially-pooled mid-block features
    noise_level : (B, T) long        per-frame diffusion noise level (optional)

    Output
    ------
    torque      : (B, T, torque_dim) per-frame joint torque (raw physical N·m)
    """

    def __init__(
        self,
        feat_dim: int,
        torque_dim: int = 8,
        hidden_dim: int = 256,
        noise_emb_dim: int = 64,
        timesteps: int = 1000,
        use_noise_cond: bool = True,
    ) -> None:
        super().__init__()
        self.torque_dim = torque_dim
        self.use_noise_cond = use_noise_cond
        in_dim = feat_dim
        if use_noise_cond:
            self.noise_emb = nn.Embedding(timesteps, noise_emb_dim)
            in_dim += noise_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, torque_dim),
        )

    def forward(
        self, feat: torch.Tensor, noise_level: torch.Tensor = None
    ) -> torch.Tensor:
        """feat: (B,T,feat_dim); noise_level: (B,T) -> torque: (B,T,torque_dim)."""
        if self.use_noise_cond:
            assert noise_level is not None, "TrunkTorqueHead needs noise_level"
            ne = self.noise_emb(noise_level.long()).to(feat.dtype)  # (B,T,noise_emb_dim)
            feat = torch.cat([feat, ne], dim=-1)
        return self.mlp(feat)
