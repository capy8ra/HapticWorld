"""State-conditioned stage-2: trunk torque head + autoregressive state branch.

NEW pipeline (isolated). Extends the trunk-head stage-2
(``LatentWorldModelTorque`` with ``torque_head_source=trunk_midblock``) with a
measured-state stream, per the agreed design:

  * the per-frame FiLM condition of the dynamics U-Net is widened from
    ``q_des`` (8) to ``[q_des, q_measured]`` (16) -- the image prediction knows
    where the arm REALLY is (bidirectional coupling, state -> image);
  * a small deterministic ``StateBranch`` (GRU over pooled latent features)
    predicts the state sequence. Within each 10-frame window it is unrolled
    AUTOREGRESSIVELY (feeding its own predictions back) and supervised with a
    dense chunk loss: MSE on every predicted frame, averaged (image -> state);
  * gradients flow through the whole unroll (BPTT); the experiment's global
    ``gradient_clip_val=1.0`` (exp_latent_dyn.yaml, passed to the Trainer)
    guards against explosion;
  * the trunk torque head is inherited UNCHANGED (same features, same raw-N*m
    loss) so torque metrics stay directly comparable with the trunk baseline.

Validation (stage 2) is fully overridden because the parent's 200-frame latent
rollout only knows the 8-d action condition. It reports:
  * ``state_rmse_tf_rad``  -- state branch on clean latents (teacher-forced scene);
  * ``state_rmse_ar_rad``  -- fully-autoregressive coupled rollout (state branch
    and diffusion feeding each other; deployment condition);
  * ``dyn_loss``           -- latent error of that coupled rollout;
  * ``torque_rmse_nm``     -- trunk head, teacher-forced windows (baseline-comparable).

Training data: ``world_ft_state_dataset`` (provides ``qpos_right``) with
``dataset.horizon=10`` (no extra frame; the unroll covers transitions 0->1 ..
8->9, i.e. 9 supervised frames per window).
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model_torque import (
    LatentWorldModelTorque,
)
from interactive_world_sim.algorithms.latent_dynamics.models.state_branch import (
    StateBranch,
)


class LatentWorldModelState(LatentWorldModelTorque):
    """Trunk-head stage-2 + state conditioning + autoregressive state branch."""

    # ------------------------------------------------------------------ build
    def _build_model(self) -> None:
        super()._build_model()  # dynamics (cond_dim-wide FiLM), trunk torque head
        sb = self.cfg.state_branch
        self.state_branch = StateBranch(
            latent_dim=self.num_latent_channel,
            state_dim=int(self.cfg.state_dim),
            action_dim=int(self.cfg.action_dim),
            feat_dim=int(sb.feat_dim),
            hidden_dim=int(sb.hidden_dim),
            num_layers=int(sb.num_layers),
            out_clamp=float(sb.get("out_clamp", 2.0)),
        )
        self.w_state = float(self.cfg.w_state)
        self.cond_state_noise_std = float(self.cfg.cond_state_noise_std)

    def configure_optimizers(self):
        if self.training_stage != 2:
            return super().configure_optimizers()
        # Build ALL param groups BEFORE the scheduler: LR schedulers only track
        # groups that exist at construction time, so appending the state branch
        # afterwards (add_param_group) would leave it at full un-warmed LR --
        # exactly the early-training transient that saturated the bounded
        # output in the first run.
        import torch as _torch
        from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

        param_groups = [
            {"params": self.dynamics.parameters(), "lr": self.cfg.lr},
            {"params": self._torque_module.parameters(), "lr": self.cfg.lr},
            {"params": self.state_branch.parameters(), "lr": self.cfg.lr},
        ]
        optimizer = _torch.optim.AdamW(
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

    # ------------------------------------------------------------- condition
    def _norm_q(self, q_raw: torch.Tensor) -> torch.Tensor:
        """Measured q uses the dedicated `qpos` normalizer (degenerate dims,
        e.g. the unused gripper, pass through un-scaled instead of being blown
        up ~1000x by the raw range fit)."""
        return self.normalizer["qpos"].normalize(q_raw).float()

    def _unnorm_q(self, q_n: torch.Tensor) -> torch.Tensor:
        return self.normalizer["qpos"].unnormalize(q_n)

    def _make_cond(self, a_raw: torch.Tensor, s_raw: torch.Tensor) -> torch.Tensor:
        """[q_des, q_measured] -> normalized 16-d FiLM condition."""
        return torch.cat([self._norm_q(a_raw), self._norm_q(s_raw)], dim=-1)

    # ------------------------------------------------------------ state utils
    def _state_unroll(
        self,
        scene_latents: torch.Tensor,  # (T-1, B, C, H, W): frame-t latent for step t->t+1
        a_n: torch.Tensor,            # (B, T, 8) normalized commands
        s0_n: torch.Tensor,           # (B, 8) normalized initial state
    ) -> torch.Tensor:
        """Autoregressive unroll over T-1 transitions -> (B, T-1, 8) normalized.

        Forced fp32 (bf16 would quantize the small residual steps; BPTT over
        the window makes precision errors compound).
        """
        dev_type = "cuda" if s0_n.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            s_hat = s0_n.float()
            h: Optional[torch.Tensor] = None
            preds = []
            for t in range(scene_latents.shape[0]):
                feat = self.state_branch.pool(scene_latents[t].float())
                s_hat, h = self.state_branch.step(s_hat, a_n[:, t + 1].float(), feat, h)
                preds.append(s_hat)
        return torch.stack(preds, dim=1)

    # -------------------------------------------------------------- training
    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        if self.training_stage != 2:
            return super().training_step(batch, batch_idx)
        if batch["obs"][self.obs_keys[0]].shape[0] == 0:
            return None

        obs = torch.cat(
            [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys],
            dim=2,
        ).float()  # (B, T, C, H, W)
        b, T = obs.shape[:2]
        a_raw = batch["action"].float()
        s_raw = batch["qpos_right"].float()
        a_n = self._norm_q(a_raw)
        s_n = self._norm_q(s_raw)

        xs = rearrange(obs, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            z = self.encoder_forward(xs)
        z = rearrange(z, "(b t) c h w -> t b c h w", b=b)

        # ---- scene branch: CTM loss with 16-d [q_des, q] condition ----
        cond = torch.cat([a_n, s_n], dim=-1)  # (B, T, 16), teacher-forced state
        if self.training and self.cond_state_noise_std > 0:
            cond = cond.clone()
            cond[..., 8:] += torch.randn_like(cond[..., 8:]) * self.cond_state_noise_std
        cond_tb = rearrange(cond, "b t d -> t b d")
        if self.mask_prev_action:
            cond_tb = cond_tb.clone()
            cond_tb[:-1] = 0

        t_lv, s_lv = self._generate_noise_levels(z, self.dyn_infer_steps)
        weights_t = self.noise_scheduler.get_weights(t_lv)
        weights_s = self.noise_scheduler.get_weights(s_lv)
        noisy_z_t, noisy_z_s = self.noise_scheduler.add_noise_to_t_s(z, t_lv, s_lv)
        pred_s = self._forward(self.dynamics, noisy_z_t, t_lv, s_lv, external_cond=cond_tb)
        trunk_feat_seen = self._trunk_feat  # trunk-head features from THIS forward
        pred_u = None
        if self.dyn_infer_steps > 1:
            u_lv = torch.zeros_like(t_lv).to(self.device)
            pred_u = self._forward(
                self.dynamics, noisy_z_s, s_lv, u_lv, external_cond=cond_tb
            )

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

        # ---- trunk torque head (inherited, unchanged semantics) ----
        self._trunk_feat = trunk_feat_seen
        torque_pred = self._pool_trunk_feat(t_lv)  # (B, T, 8)
        self._trunk_feat = None
        torque_loss = F.mse_loss(torque_pred, self._torque_target(batch))  # raw N*m^2

        # ---- state branch: autoregressive chunk loss over frames 1..T-1 ----
        state_pred_n = self._state_unroll(z[:-1], a_n, s_n[:, 0])  # (B, T-1, 8)
        state_loss = F.mse_loss(state_pred_n, s_n[:, 1:])          # normalized units
        with torch.no_grad():
            s_pred_raw = self._unnorm_q(state_pred_n)
            state_rmse_rad = torch.sqrt(F.mse_loss(s_pred_raw, s_raw[:, 1:]))

        total = (
            dyn_loss
            + self.torque_loss_weight * torque_loss
            + self.w_state * state_loss
        )
        self.log("training/loss", total)
        self.log("training/dyn_loss", dyn_loss)
        self.log("training/torque_mse", torque_loss)
        self.log("training/torque_rmse_nm", torch.sqrt(torque_loss.detach()))
        self.log("training/state_mse", state_loss)
        self.log("training/state_rmse_rad", state_rmse_rad)
        return {"loss": total}

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def coupled_rollout(
        self,
        z0: torch.Tensor,      # (B, T0, C, H, W) latent context (encoder outputs)
        s0_raw: torch.Tensor,  # (B, 8) raw measured state at the last context frame
        a_raw: torch.Tensor,   # (B, T, 8) raw q_des; frames T0..T-1 are executed
    ) -> dict:
        """Fully-autoregressive interleaved rollout: state branch and diffusion
        feed each other (deployment condition). Mirrors ``dynamics_forward``'s
        sliding-window sampler with the 16-d [q_des, s_hat] condition."""
        B, T0 = z0.shape[:2]
        T = a_raw.shape[1]
        xs_pred = rearrange(z0, "b t c h w -> t b c h w").to(self.dtype).clone()

        a_n = self._norm_q(a_raw)
        s_hat_n = self._norm_q(s0_raw)
        conds = [torch.cat([a_n[:, t], s_hat_n], dim=-1) for t in range(T0)]
        h: Optional[torch.Tensor] = None
        states_n = []

        timesteps = torch.linspace(
            self.noise_scheduler.timesteps - 1, 0, self.dyn_infer_steps + 1,
            device=self.device,
        )
        for i in range(T0, T):
            # --- state step: scene context = latest latent (NaN-guarded: a
            # diverged diffusion frame must not poison the state recursion) ---
            feat = self.state_branch.pool(torch.nan_to_num(xs_pred[-1].float()))
            s_hat_n, h = self.state_branch.step(s_hat_n, a_n[:, i], feat, h)
            states_n.append(s_hat_n)
            conds.append(torch.cat([a_n[:, i], s_hat_n], dim=-1))

            # --- scene step: one frame via the sliding-window sampler ---
            chunk = torch.randn(
                (1, B, *xs_pred.shape[2:]), device=self.device, dtype=self.dtype
            )
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)
            curr_end = xs_pred.shape[0]
            curr_start = max(0, curr_end - self.n_tokens)
            window = curr_end - curr_start
            clean_t = (
                torch.ones((window - 1,), device=self.device)
                * self.noise_scheduler.stabilization_level
            )
            cond_window = torch.stack(conds[curr_start:curr_end], dim=0).float()
            if self.mask_prev_action:
                cond_window = cond_window.clone()
                cond_window[:-1] = 0
            for step_i in range(self.dyn_infer_steps):
                t_lv = torch.cat([clean_t, timesteps[step_i].unsqueeze(0)], 0)
                s_lv = torch.cat([clean_t, timesteps[step_i + 1].unsqueeze(0)], 0)
                t_lv = torch.tile(t_lv[:, None], (1, B)).long()
                s_lv = torch.tile(s_lv[:, None], (1, B)).long()
                updated = self._forward(
                    self.dynamics, xs_pred[curr_start:], t_lv, s_lv,
                    external_cond=cond_window,
                )
                if self.last_frame_loss_only:
                    xs_pred[-1:] = updated[-1:]
                else:
                    xs_pred[curr_start:] = updated

        num_views = len(self.obs_keys)
        c_per_v = xs_pred.shape[2] // num_views
        for v in range(num_views):
            chunk_v = xs_pred[:, :, v * c_per_v : (v + 1) * c_per_v].clone()
            xs_pred[:, :, v * c_per_v : (v + 1) * c_per_v] = chunk_v / (
                torch.norm(chunk_v, dim=2, keepdim=True) + 1e-8
            )
        states = self._unnorm_q(torch.stack(states_n, dim=1))
        return {
            "z_seq": rearrange(xs_pred, "t b c h w -> b t c h w"),  # (B, T, C, H, W)
            "q": states,  # (B, T-T0, 8) raw rad, frames T0..T-1
        }

    # ------------------------------------------------------------ validation
    def validation_step(
        self, batch: dict, batch_idx: int, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        if self.training_stage != 2:
            return super().validation_step(batch, batch_idx, namespace)

        with torch.no_grad():
            obs = torch.cat(
                [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys],
                dim=2,
            ).float()
            b, T = obs.shape[:2]
            a_raw = batch["action"].float()
            s_raw = batch["qpos_right"].float()
            a_n = self._norm_q(a_raw)

            xs = rearrange(obs, "b t c h w -> (b t) c h w")
            z_gt = self.encoder_forward(xs)
            z_gt = rearrange(z_gt, "(b t) c h w -> b t c h w", b=b)
            z_gt_tb = rearrange(z_gt, "b t c h w -> t b c h w")

            # (1a) teacher-forced-scene state (clean latents, state AR over the
            # WHOLE horizon -- probes long-horizon stability far beyond the
            # 9-step training regime)
            s_tf_n = self._state_unroll(z_gt_tb[:-1], a_n, self._norm_q(s_raw[:, 0]))
            s_tf = self._unnorm_q(s_tf_n)
            state_rmse_tf = torch.sqrt(F.mse_loss(s_tf, s_raw[:, 1:]))

            # (1b) windowed TF state: re-seeded from GT every n_tokens frames --
            # the training-regime metric. Diverging (1a) with a healthy (1b)
            # means "long-horizon integration drifts", not "the head is bad".
            s_n_gt = self._norm_q(s_raw)
            win_preds = []
            for start in range(0, T - 1, self.n_tokens):
                end = min(start + self.n_tokens, T - 1)
                win_preds.append(
                    self._state_unroll(
                        z_gt_tb[start:end], a_n[:, start : end + 1], s_n_gt[:, start]
                    )
                )
            s_win = self._unnorm_q(torch.cat(win_preds, dim=1))
            state_rmse_tfwin = torch.sqrt(F.mse_loss(s_win, s_raw[:, 1:]))

            # (2) trunk torque, teacher-forced windows (baseline-comparable);
            #     replicates the parent's windowed replay with the 16-d condition.
            cond_tb = rearrange(
                torch.cat([a_n, self._norm_q(s_raw)], dim=-1), "b t d -> t b d"
            )
            outs = []
            for start in range(0, T, self.n_tokens):
                end = min(start + self.n_tokens, T)
                zc, cc = z_gt_tb[start:end], cond_tb[start:end]
                t_lv, s_lv = self._generate_noise_levels(zc, self.dyn_infer_steps)
                noisy_zt, _ = self.noise_scheduler.add_noise_to_t_s(zc, t_lv, s_lv)
                _ = self._forward(self.dynamics, noisy_zt, t_lv, s_lv, external_cond=cc)
                outs.append(self._pool_trunk_feat(t_lv))
            self._trunk_feat = None
            torque_pred = torch.cat(outs, dim=1)  # (B, T, 8)
            tmse = F.mse_loss(torque_pred, self._torque_target(batch))

            # (3) fully-autoregressive coupled rollout
            out = self.coupled_rollout(z_gt[:, :1], s_raw[:, 0], a_raw)
            # NaN-guard the latents BEFORE any metric/render: a diverged
            # rollout must degrade metrics, not poison FVD/videos (NaN there
            # can crash a rank and deadlock DDP).
            z_seq = torch.nan_to_num(out["z_seq"].float(), nan=0.0)
            q_ar = torch.nan_to_num(out["q"], nan=1e3)
            state_rmse_ar = torch.sqrt(F.mse_loss(q_ar, s_raw[:, 1:]))
            dyn_loss = F.mse_loss(z_seq[:, 1:], z_gt[:, 1:])

        # sentinel-clip all logged scalars: never propagate NaN/inf into
        # wandb/monitors (they also poison DDP metric reductions)
        _s = lambda x: torch.nan_to_num(x, nan=1e3, posinf=1e3)  # noqa: E731
        self.log(f"{namespace}/dyn_loss", _s(dyn_loss))
        self.validation_metrics.setdefault("dyn_loss", []).append(_s(dyn_loss))
        self.log(f"{namespace}/torque_mse", _s(tmse))
        self.log(f"{namespace}/torque_rmse_nm", _s(torch.sqrt(tmse)))
        self.log(f"{namespace}/state_rmse_tf_rad", _s(state_rmse_tf))
        self.log(f"{namespace}/state_rmse_tfwin_rad", _s(state_rmse_tfwin))
        self.log(f"{namespace}/state_rmse_ar_rad", _s(state_rmse_ar))

        if self.val_render:
            z_flat = rearrange(z_seq.to(self.dtype), "b t c h w -> (b t) c h w")
            xs_pred = render_img_cm(
                self, z_flat, tuple(obs.shape[-2:]), self.normalizer, num_views=self.num_views
            )
            xs_pred = rearrange(xs_pred, "(b t) c h w -> t b c h w", b=b)
            xs_gt = torch.cat([batch["obs"][k] for k in self.obs_keys], dim=2)
            xs_gt = rearrange(xs_gt, "b t c h w -> t b c h w")
            self.validation_step_outputs.append(
                (xs_pred.detach().cpu(), xs_gt.detach().cpu())
            )

        if batch_idx == 0 and self.trainer.is_global_zero and self.logger is not None:
            self._log_state_curves(
                namespace,
                s_raw[0, 1:].float().cpu().numpy(),
                s_win[0].float().cpu().numpy(),
                q_ar[0].float().cpu().numpy(),
            )
        return None

    def _log_state_curves(
        self, namespace: str, gt: np.ndarray, tf: np.ndarray, ar: np.ndarray
    ) -> None:
        """8-joint measured-q curves (gt vs windowed-TF vs autoregressive)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        T, D = gt.shape
        ncol, nrow = 4, (D + 3) // 4
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow), squeeze=False)
        for j in range(D):
            ax = axes[j // ncol][j % ncol]
            ax.plot(gt[:, j], lw=1.3, label="gt")
            ax.plot(tf[:, j], lw=1.0, ls="--", label="tf")
            ax.plot(ar[:, j], lw=1.0, ls=":", label="ar")
            r_tf = float(((tf[:, j] - gt[:, j]) ** 2).mean() ** 0.5)
            r_ar = float(((ar[:, j] - gt[:, j]) ** 2).mean() ** 0.5)
            ax.set_title(f"joint {j}  tf={r_tf:.3f} ar={r_ar:.3f} rad", fontsize=8)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.legend(fontsize=7)
        for j in range(D, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.tight_layout()
        self.logger.experiment.log({f"{namespace}/state_curves": wandb.Image(fig)})
        plt.close(fig)
