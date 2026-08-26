"""Factored world model -- robot branch (NEW pipeline, isolated).

The factored stage-2 splits the world model along the physical contact
interface:

  * scene branch: the existing CTM latent diffusion (untouched class), with its
    per-frame FiLM condition widened from ``q_des`` to ``[q_des, q_hat,
    tau_int_hat]``;
  * robot branch (this file): a small deterministic GRU that predicts the
    realized joint state under contact, from which the measured joint torque is
    recovered ANALYTICALLY -- no learned torque head at all.

The follower runs 2 kHz MIT impedance control (supernova
``teleop_follower_solver.py`` + Damiao firmware):

    tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff(q, dq)

with known constant gains and ``tau_ff`` the gravity/Coriolis feedforward the
solver computes with pinocchio RNEA. Verified on the raw 2026_06_26_1
recordings: this law reproduces the measured ``actual_torque_nm`` with
R^2 ~ 0.995 at native rate AND at the 10 Hz frame times (residual ~0.25 N*m =
motor friction/current-estimation floor). Hence torque prediction ==
realized-state prediction, and all learning capacity goes into the tracking
deviation ``delta = q_des - q`` -- which in free space is a small smooth lag
and under contact is exactly the scene-dependent quantity the visual world
model exists to provide (via the latent feature input).

The model predicts the INTERACTION torque (gravity-free part):

    tau_int = tau - tau_ff = kp * delta + kd * (dq_des - dq)

so no tau_ff surrogate is needed anywhere in the loop: tau_int is what force
feedback (teleop milestone 2), the scene-branch conditioning, and the contact
story all consume. The training target ``tau_meas - tau_ff`` uses the RECORDED
per-frame ``tau_ff``; for a full-torque metric the recorded ``tau_ff`` is
added back (q_hat is within tens of mrad of q, and tau_ff is smooth, so this
reconstruction error is negligible). A learned tau_ff(q, dq) surrogate was
evaluated and rejected: an MLP fit on the play data fails to extrapolate off
the visited pose manifold (cross-episode R^2 collapse).

Anti-drift parameterization: the branch predicts the DEVIATION ``delta`` and
reconstructs ``q_hat = q_des - delta``. ``q_des`` is an exogenous ground-truth
input at every step (teleop leader / dataset), so state errors do not
integrate over time the way an odometry-style ``q_{t+1} = q_t + dq`` would.

Modules
-------
LatentPool        latent map (B, C, 32, 32) -> feature (B, F). Scene context.
LatentStateProbe  latent map -> q (8).  The cross-branch consistency probe.
RobotBranch       causal GRU over [state, command, scene feature] -> per-step
                  (delta, dq_next) + analytic interaction-torque decode.
"""
from typing import Optional, Tuple

import torch
from torch import nn


class LatentPool(nn.Module):
    """Per-frame latent map -> pooled feature vector.

    Same conv/GAP design as ``TorquePredictor.latent_enc`` in the existing
    pipeline: (B, C, 32, 32) -> (B, feat_dim).
    """

    def __init__(self, latent_dim: int = 4, feat_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, 32, kernel_size=3, stride=2, padding=1),   # 32 -> 16
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 16 -> 8
            nn.SiLU(),
            nn.Conv2d(64, feat_dim, kernel_size=3, stride=2, padding=1),     # 8 -> 4
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_dim = feat_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, H, W) -> (B, feat_dim)."""
        return self.net(z).flatten(1)


class LatentStateProbe(nn.Module):
    """Decode the robot configuration from an image latent: z -> q (raw rad).

    Used for (a) the train-time consistency loss on predicted latents (the
    generated frame must contain the arm at the right metric configuration --
    gradient flows into the dynamics U-Net, not into the probe) and (b) the
    optional rollout-time anchoring of the robot-branch state.
    """

    def __init__(self, latent_dim: int = 4, dof: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.pool = LatentPool(latent_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dof),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, H, W) -> q: (B, dof) raw rad."""
        return self.head(self.pool(z))


class RobotBranch(nn.Module):
    """Contact-aware realized-state dynamics + analytic torque decoder.

    One step (frame t -> t+1, 10 Hz):

        x_t   = [q_t, dq_t, (q_des_{t+1} - q_t) * diff_scale, dq_des_{t+1}, phi(z_t)]
        h_t+1 = GRU(x_t, h_t)
        delta_{t+1}, dq_{t+1} = head(h_{t+1})
        q_{t+1}       = q_des_{t+1} - delta_{t+1}
        tau_int_{t+1} = kp*delta_{t+1} + kd*(dq_des_{t+1} - dq_{t+1})

    All joint quantities are RAW physical units (rad, rad/s, N*m). The pending
    tracking error is fed pre-scaled (mrad-level signal made order-1) so the
    network does not have to recover a tiny difference of two large inputs.
    The delta/dq head is zero-initialized: at init q_hat == q_des (perfect
    tracking), which is the correct free-space prior.
    """

    def __init__(
        self,
        kp: list,
        kd: list,
        latent_dim: int = 4,
        dof: int = 8,
        scene_feat_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        diff_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.diff_scale = diff_scale
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.register_buffer("kp", torch.as_tensor(kp, dtype=torch.float32))
        self.register_buffer("kd", torch.as_tensor(kd, dtype=torch.float32))
        assert self.kp.shape == (dof,) and self.kd.shape == (dof,)

        self.scene_enc = LatentPool(latent_dim, scene_feat_dim)
        self.gru = nn.GRU(
            input_size=4 * dof + scene_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 2 * dof)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.weight.mul_(0.01)

    def init_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch, self.hidden_dim, device=device)

    def step(
        self,
        q: torch.Tensor,          # (B, dof) raw rad      -- current state
        dq: torch.Tensor,         # (B, dof) raw rad/s
        a_next: torch.Tensor,     # (B, dof) raw rad      -- q_des of the step
        da_next: torch.Tensor,    # (B, dof) raw rad/s    -- dq_des of the step
        scene_feat: torch.Tensor,  # (B, F)               -- phi(z_t)
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One 10 Hz transition. Returns (q_next, dq_next, delta, h)."""
        x = torch.cat(
            [q, dq, (a_next - q) * self.diff_scale, da_next, scene_feat], dim=-1
        )
        y, h = self.gru(x.unsqueeze(1), h)
        out = self.head(y.squeeze(1))
        delta = out[:, : self.dof]
        dq_next = out[:, self.dof :]
        q_next = a_next - delta
        return q_next, dq_next, delta, h

    def torque_int(
        self,
        delta: torch.Tensor,     # (B, dof) rad
        dq_next: torch.Tensor,   # (B, dof) rad/s
        da_next: torch.Tensor,   # (B, dof) rad/s
    ) -> torch.Tensor:
        """Analytic interaction-torque decode (zero learned parameters), N*m.

        tau_int = tau - tau_ff; add the recorded tau_ff back when a
        full-torque number is needed for reporting.
        """
        return self.kp * delta + self.kd * (da_next - dq_next)
