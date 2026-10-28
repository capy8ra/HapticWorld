"""HapticWorld stage 2: latent dynamics + joint-torque head.

Subclass of ``LatentWorldModel`` that adds the torque head on top of the consistency-model latent
dynamics. Only stage 2 changes; stages 1 and 3 delegate to the parent.

The head (``TrunkTorqueHead``) branches off the dynamics U-Net's mid-block features, captured by a
forward hook during the regular dynamics forward, so ``CMLatentDynamics.forward`` and the
consistency-model arithmetic are untouched. The torque loss backpropagates through the whole trunk,
so the features that generate the next frame also carry the contact information. The features come
from noisy latents, so the head is also conditioned on each frame's noise level. Torque is not part
of the diffusion: it is a deterministic prediction in N·m, supervised on every frame of the window.
"""
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from hapticworld.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from hapticworld.algorithms.latent_dynamics.models.torque_head import TrunkTorqueHead


class LatentWorldModelTorque(LatentWorldModel):
    """LatentWorldModel + stage-2 right-arm joint-torque prediction."""

    def _build_model(self) -> None:
        super()._build_model()
        self.torque_loss_weight = float(self.cfg.get("torque_loss_weight", 1.0))
        self._trunk_feat = None  # set by the mid-block forward hook
        # mid-block feature width = dynamics.dim * dynamics.dim_mults[-1]
        feat_dim = int(self.cfg.dynamics.dim) * int(list(self.cfg.dynamics.dim_mults)[-1])
        self.torque_head = TrunkTorqueHead(
            feat_dim=feat_dim,
            torque_dim=int(self.cfg.torque_dim),
            timesteps=int(self.cfg.diffusion.timesteps),
            use_noise_cond=bool(self.cfg.get("torque_noise_cond", True)),
        )
        # capture the bottleneck output during every dynamics forward
        self.dynamics.module.mid_block.register_forward_hook(self._capture_trunk_feat)

    # ----- mid-block hook: stash features in EinopsWrapper "b c f h w" layout ----
    def _capture_trunk_feat(self, _module: Any, _inp: Any, out: torch.Tensor) -> None:
        self._trunk_feat = out  # (B, C, T, H, W)

    def _pool_trunk_feat(self, t_levels: torch.Tensor) -> torch.Tensor:
        """(B,C,T,H,W) trunk feat + (T,B) noise levels -> torque (B,T,8)."""
        feat = self._trunk_feat.mean(dim=(-1, -2))  # (B, C, T) spatial GAP
        feat = feat.permute(0, 2, 1)                # (B, T, C)
        return self.torque_head(feat, noise_level=t_levels.transpose(0, 1))

    # ----- optimizer: stage-2 optimizes dynamics + torque head -----
    def configure_optimizers(self) -> Any:
        if self.training_stage != 2:
            return super().configure_optimizers()
        param_groups = [
            {"params": self.dynamics.parameters(), "lr": self.cfg.lr},
            {"params": self.torque_head.parameters(), "lr": self.cfg.lr},
        ]
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        if self.lr_scheduler == "linear":
            lr_scheduler = LinearLR(
                optimizer, start_factor=1e-4, end_factor=1.0,
                total_iters=self.cfg.warmup_steps,
            )
        elif self.lr_scheduler == "plateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=50000,
                threshold=1e-3, threshold_mode="rel",
            )
        else:
            raise NotImplementedError(f"LR scheduler {self.lr_scheduler} not included")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler, "interval": "step", "frequency": 1,
                "monitor": "training/loss", "strict": True, "name": "lr_scheduler",
            },
        }

    def _torque_target(self, batch: dict) -> torch.Tensor:
        # raw torque in N·m -- intentionally NOT normalized, so the head's output
        # is directly usable as actual torque (no normalizer stats needed later).
        return batch["joint_torque"].float()

    # ----- training: stage-2 latent loss + dense torque loss -----
    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        if self.training_stage != 2:
            return super().training_step(batch, batch_idx)
        if batch["obs"][self.obs_keys[0]].shape[0] == 0:
            return None

        obs = torch.cat(
            [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys], dim=2
        ).float()
        action_bt = self.normalizer["action"].normalize(batch["action"]).float()  # (B,T,A)
        b = obs.shape[0]
        xs = rearrange(obs, "b t c h w -> (b t) c h w")

        with torch.no_grad():
            z = self.encoder_forward(xs)  # frozen encoder
        z = rearrange(z, "(b t) c h w -> t b c h w", b=b)
        action = rearrange(action_bt, "b t a -> t b a")

        t, s = self._generate_noise_levels(z, self.dyn_infer_steps)
        weights_t = self.noise_scheduler.get_weights(t)
        weights_s = self.noise_scheduler.get_weights(s)
        noisy_z_t, noisy_z_s = self.noise_scheduler.add_noise_to_t_s(z, t, s)
        u = torch.zeros_like(t).to(self.device)
        if self.mask_prev_action:
            action = action.clone()
            action[:-1] = 0
        pred_s = self._forward(self.dynamics, noisy_z_t, t, s, external_cond=action)
        # capture trunk features from THIS (pred_s) forward before any other call.
        trunk_feat_seen = self._trunk_feat
        pred_u = None
        if self.dyn_infer_steps > 1:
            pred_u = self._forward(self.dynamics, noisy_z_s, s, u, external_cond=action)

        # ---- latent dynamics loss (faithful to parent stage-2) ----
        if self.last_frame_loss_only:
            loss_s = F.mse_loss(pred_s[-1:], noisy_z_s[-1:].detach(), reduction="none")
            wt = weights_t.view(*weights_t.shape, *((1,) * (loss_s.ndim - 2)))[-1:]
            loss = loss_s * wt
            if pred_u is not None:
                loss_u = F.mse_loss(pred_u[-1:], z[-1:].detach(), reduction="none")
                ws = weights_s.view(*weights_s.shape, *((1,) * (loss_u.ndim - 2)))[-1:]
                loss = loss + loss_u * ws
        else:
            loss_s = F.mse_loss(pred_s, noisy_z_s.detach(), reduction="none")
            wt = weights_t.view(*weights_t.shape, *((1,) * (loss_s.ndim - 2)))
            loss = loss_s * wt
            if pred_u is not None:
                loss_u = F.mse_loss(pred_u, z.detach(), reduction="none")
                ws = weights_s.view(*weights_s.shape, *((1,) * (loss_s.ndim - 2)))
                loss = loss + loss_u * ws
        dyn_loss = loss.mean()

        # ---- dense torque loss ----
        # read the trunk features of the pred_s forward
        self._trunk_feat = trunk_feat_seen
        torque_pred = self._pool_trunk_feat(t)  # (B,T,8)
        self._trunk_feat = None  # release graph ref
        torque_target = self._torque_target(batch)  # raw N·m
        torque_loss = F.mse_loss(torque_pred, torque_target)  # in N·m^2

        total_loss = dyn_loss + self.torque_loss_weight * torque_loss
        self.log("training/loss", total_loss)
        self.log("training/dyn_loss", dyn_loss)
        self.log("training/torque_mse", torque_loss)
        # predictions are already in N·m -> RMSE is just sqrt(MSE).
        self.log("training/torque_rmse_nm", torch.sqrt(torque_loss.detach()))
        return {"loss": total_loss}

    # ----- validation: parent latent metrics + torque metric -----
    def _val_torque_pred(
        self, z_btchw: torch.Tensor, action_bt: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced torque over the (possibly long) val sequence.

        The head reads features produced inside the dynamics, so the training-style
        (terminal_only) noise + dynamics forward is replayed in non-overlapping windows of
        ``n_tokens`` (the trained context length) and the per-frame predictions are stitched.
        """

        z_tb = rearrange(z_btchw, "b t c h w -> t b c h w")
        action_tb = rearrange(action_bt, "b t a -> t b a")
        total = z_tb.shape[0]
        outs = []
        for start in range(0, total, self.n_tokens):
            end = min(start + self.n_tokens, total)
            zc, ac = z_tb[start:end], action_tb[start:end]
            t, s = self._generate_noise_levels(zc, self.dyn_infer_steps)
            noisy_zt, _ = self.noise_scheduler.add_noise_to_t_s(zc, t, s)
            _ = self._forward(self.dynamics, noisy_zt, t, s, external_cond=ac)
            outs.append(self._pool_trunk_feat(t))  # (B, t_chunk, 8)
        self._trunk_feat = None
        return torch.cat(outs, dim=1)  # (B, T, 8)

    def validation_step(
        self, batch: dict, batch_idx: int, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        super().validation_step(batch, batch_idx, namespace)
        if self.training_stage != 2:
            return None
        with torch.no_grad():
            obs = torch.cat(
                [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys],
                dim=2,
            ).float()
            action_bt = self.normalizer["action"].normalize(batch["action"]).float()
            b = obs.shape[0]
            xs = rearrange(obs, "b t c h w -> (b t) c h w")
            z = self.encoder_forward(xs)
            z = rearrange(z, "(b t) c h w -> b t c h w", b=b)
            torque_pred = self._val_torque_pred(z, action_bt)  # raw N·m
            torque_target = self._torque_target(batch)
            tmse = F.mse_loss(torque_pred, torque_target)  # in N·m^2
            trmse = torch.sqrt(tmse)  # predictions already in N·m
        self.log(f"{namespace}/torque_mse", tmse)
        self.log(f"{namespace}/torque_rmse_nm", trmse)
        # ---- predicted vs measured torque curves -> wandb (rank 0, first val batch, first sample) ----
        if batch_idx == 0 and self.trainer.is_global_zero and self.logger is not None:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import wandb
            tp = torque_pred[0].float().cpu().numpy()    # (T, 8)
            tg = torque_target[0].float().cpu().numpy()  # (T, 8)
            T, D = tp.shape
            ncol, nrow = 4, (D + 3) // 4
            fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow), squeeze=False)
            for j in range(D):
                ax = axes[j // ncol][j % ncol]
                ax.plot(tg[:, j], lw=1.3, label="gt")
                ax.plot(tp[:, j], lw=1.0, ls="--", label="pred")
                rmse = float(((tp[:, j] - tg[:, j]) ** 2).mean() ** 0.5)
                ax.set_title(f"joint {j}  rmse={rmse:.2f} N·m", fontsize=8)
                ax.tick_params(labelsize=6)
                if j == 0:
                    ax.legend(fontsize=7)
            for j in range(D, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.tight_layout()
            self.logger.experiment.log({f"{namespace}/torque_curves": wandb.Image(fig)})
            plt.close(fig)

        return None
