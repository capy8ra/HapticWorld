#!/usr/bin/env python
"""World-model "virtual follower" worker — STATE-conditioned stage-2 variant.

Same real-world teleop as ``wm_follower_worker.py`` (human moves the real RIGHT
leader; the world model is the virtual follower on screen; predicted joint torque
is published for the leader force-feedback loop), but for the **state-conditioned**
model ``LatentWorldModelState`` (ModelScope ``stage2_state``):

  * the dynamics FiLM condition is 16-d ``[q_des(8), q_measured(8)]`` (not 8-d);
  * ``q_measured`` is NOT read from a real follower — the model's autoregressive
    GRU ``state_branch`` predicts it, seeded once from the start pose. This is the
    ``coupled_rollout`` deployment path (state branch and diffusion feed each other),
    reproduced here one frame at a time as live leader angles arrive.

Differences vs the trunk worker (``wm_follower_worker.py``):
  * loads ``LatentWorldModelState`` and steps the coupled (scene+state) rollout;
  * the leader command is normalized with the model's dedicated ``qpos`` normalizer
    (``model._norm_q``), matching training — NOT the ``action`` normalizer;
  * torque uses the inherited trunk head on the 16-d condition;
  * ``--stage3-ckpt`` graft is kept identical. Use the STATE-MATCHED stage-3
    (ModelScope ``stage3/2026-07-21_22-54-36``, local ``world_ft_ckpts/stage3_state``;
    load_ae = stage-1 ``2026-07-16_14-12-06``, same as ``stage2_state``). Do NOT use
    the old ``world_ft_ckpts/stage3`` — that one was fine-tuned from the trunk's
    stage-1 (``2026-07-15_02-35-54``) and its latent space does not match.

Env: hapticworld:
    python scripts/inference/wm_follower_worker_state.py \
      --ckpt world_ft_ckpts/stage2_state/checkpoints/last.ckpt \
      --stage3-ckpt world_ft_ckpts/stage3_state/checkpoints/last.ckpt \
      --seed-episode data/world_ft_seed_0713/val/episode_0.hdf5
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend
from yixuan_utilities.draw_utils import center_crop

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model_state import (
    LatentWorldModelState,
)

# Reuse the async decoder + session recorder from the trunk worker (identical display
# path). Importing it also enforces the nova.redis_client requirement below.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_follower_worker import (  # noqa: E402
    _DECODE_LOCK,
    _ACTRecorder,
    _AsyncDecoder,
    _BackgroundSaver,
    _SessionRecorder,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "robot")))  # nova package
try:
    from nova.redis_client import RedisClient
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Could not import nova.redis_client -- is the HapticWorld repo layout intact "
        "(robot/nova next to worldsim/)?\n"
        f"(original error: {e})"
    )


def load_model(ckpt_path: str, device: str, stage3_ckpt: str | None = None) -> tuple[LatentWorldModelState, object]:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path} (need the run's .hydra/config.yaml)")
    if not OmegaConf.has_resolver("torch"):
        OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
    cfg = OmegaConf.load(cfg_path)
    load_dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.algorithm.load_ae = None
    model = LatentWorldModelState.load_from_checkpoint(
        ckpt_path, cfg=cfg.algorithm, map_location=device,
        dtype=load_dtype, strict=False, weights_only=False,
    )
    model.dynamics = model.dynamics.to(load_dtype)

    if stage3_ckpt is not None:
        sd = torch.load(stage3_ckpt, map_location=device, weights_only=False)
        sd = sd.get("state_dict", sd)
        dec_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
        if not dec_sd:
            raise ValueError(f"No decoder.* weights in stage-3 ckpt {stage3_ckpt}")
        missing, unexpected = model.decoder.load_state_dict(dec_sd, strict=False)
        model.decoder = model.decoder.to(load_dtype)
        print(f"[wm-state] grafted stage-3 decoder from {stage3_ckpt} "
              f"({len(dec_sd)} tensors; missing={len(missing)} unexpected={len(unexpected)})")

    model.eval()
    for m in model.modules():
        if hasattr(m, "cuda_backends"):
            m.cuda_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    return model, cfg


@torch.no_grad()
def seed_from_episode(
    model: LatentWorldModelState, seed_episode: str, res: int, device: str,
    amp_dtype: torch.dtype = torch.float32, amp_on: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return (z0 (1,1,C,H,W) latent, s0_raw (1,8), a0_raw (1,8), seed_gripper).

    s0 is the measured start state. Prefer ``obs/qpos_right`` (v4/v5 state datasets);
    fall back to ``action[0]`` (measured ~= commanded at the homed seed pose) with a
    warning if the episode predates the state converter.
    """
    obs_key = model.obs_keys[0]
    with h5py.File(seed_episode, "r") as f:
        frame0 = f["obs"]["images"][obs_key][0]        # (H,W,3) uint8
        action0 = f["action"][0].astype(np.float32)    # (8,) right q_des
        if "qpos_right" in f["obs"]:
            s0 = f["obs"]["qpos_right"][0].astype(np.float32)  # (8,) measured
        else:
            s0 = action0.copy()
            print(f"[wm-state] WARNING: {seed_episode} has no obs/qpos_right; "
                  f"seeding measured state from action[0] (homed-pose proxy).")
    img = center_crop(frame0, (res, res))
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    img_t = model.normalizer[obs_key].normalize(img_t)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
        z0 = model.encoder_forward(img_t).float()[:, None]  # (1,1,C,H,W) fp32 state
    s0_raw = torch.from_numpy(s0)[None].to(device)          # (1,8)
    a0_raw = torch.from_numpy(action0)[None].to(device)     # (1,8)
    return z0, s0_raw, a0_raw, float(action0[7])


class StateOnlineRollout:
    """Live, one-frame-at-a-time reproduction of ``LatentWorldModelState.coupled_rollout``.

    Maintains the raw latent context window (fed back un-normalized, exactly as the
    coupled rollout does), the GRU hidden state, the running predicted measured state
    ``s_hat`` and the per-frame 16-d conditions. Each ``step(q_des)`` advances one
    transition and returns the display latent (per-view unit-normalized copy) and the
    predicted total joint torque (raw N·m) for that frame.
    """

    def __init__(self, model: LatentWorldModelState, z0: torch.Tensor, s0_raw: torch.Tensor,
                 a0_raw: torch.Tensor, device: str, amp_dtype: torch.dtype, amp_on: bool,
                 n_ctx: int):
        self.m = model
        self.device = device
        self.amp_dtype = amp_dtype
        self.amp_on = amp_on
        self.n_tokens = int(n_ctx)
        self.num_views = int(model.num_views)
        self.dyn_steps = int(model.dyn_infer_steps)
        # context latents in coupled_rollout's (t, b, c, h, w) layout, RAW (un-normalized)
        self.xs = rearrange(z0, "b t c h w -> t b c h w").to(model.dtype).clone()
        with self._nocast():
            self.s_hat = model._norm_q(s0_raw).float()            # (B, 8) normalized
            a0_n = model._norm_q(a0_raw).float()                  # (B, 8) normalized
        self.conds = [torch.cat([a0_n, self.s_hat], dim=-1)]      # list of (B,16); aligned with xs
        self.h: torch.Tensor | None = None
        self.timesteps = torch.linspace(
            model.noise_scheduler.timesteps - 1, 0, self.dyn_steps + 1, device=device
        )

    def _nocast(self):
        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=dev, enabled=False)

    def _normalize_views(self, xs_tb: torch.Tensor) -> torch.Tensor:
        """Per-view unit-norm over the channel dim (what coupled_rollout applies to its
        output before decode/metrics; the decoder + torque head expect unit-norm latents)."""
        out = xs_tb.clone()
        c_per_v = out.shape[2] // self.num_views
        for v in range(self.num_views):
            sl = slice(v * c_per_v, (v + 1) * c_per_v)
            chunk = out[:, :, sl]
            out[:, :, sl] = chunk / (torch.norm(chunk, dim=2, keepdim=True) + 1e-8)
        return out

    @torch.no_grad()
    def step(self, q_des_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Advance one frame. Returns (display_latent (1,C,H,W), torque (1,8) N·m,
        s_hat_raw (8,) predicted measured state in rad)."""
        m = self.m
        B = self.xs.shape[1]

        # --- state branch (kept fp32: small residual GRU steps, long recursion) ---
        with self._nocast():
            a_n = m._norm_q(q_des_raw).float()                       # (B,8)
            feat = m.state_branch.pool(torch.nan_to_num(self.xs[-1].float()))
            self.s_hat, self.h = m.state_branch.step(self.s_hat, a_n, feat, self.h)
        cond_new = torch.cat([a_n, self.s_hat], dim=-1)              # (B,16)
        self.conds.append(cond_new)

        # --- scene step: one diffusion frame via the sliding-window sampler ---
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_on):
            chunk = torch.randn((1, B, *self.xs.shape[2:]), device=self.device, dtype=m.dtype)
            chunk = torch.clamp(chunk, -m.clip_noise, m.clip_noise)
            self.xs = torch.cat([self.xs, chunk], 0)
            curr_end = self.xs.shape[0]
            curr_start = max(0, curr_end - self.n_tokens)
            window = curr_end - curr_start
            clean_t = (
                torch.ones((window - 1,), device=self.device)
                * m.noise_scheduler.stabilization_level
            )
            cond_window = torch.stack(self.conds[curr_start:curr_end], dim=0).float()  # (win,B,16)
            if m.mask_prev_action:
                cond_window = cond_window.clone()
                cond_window[:-1] = 0
            for si in range(self.dyn_steps):
                t_lv = torch.cat([clean_t, self.timesteps[si].unsqueeze(0)], 0)
                s_lv = torch.cat([clean_t, self.timesteps[si + 1].unsqueeze(0)], 0)
                t_lv = torch.tile(t_lv[:, None], (1, B)).long()
                s_lv = torch.tile(s_lv[:, None], (1, B)).long()
                updated = m._forward(m.dynamics, self.xs[curr_start:], t_lv, s_lv,
                                     external_cond=cond_window)
                if m.last_frame_loss_only:
                    self.xs[-1:] = updated[-1:]
                else:
                    self.xs[curr_start:] = updated

        # display + torque use the per-view unit-normalized latents (like validation);
        # the RAW self.xs stays as context for the next step (coupled_rollout behavior).
        z_norm = self._normalize_views(self.xs)                     # (win?,B,C,H,W)
        z_disp = z_norm[-1]                                          # (B,C,H,W)
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_on):
            z_bt = rearrange(z_norm, "t b c h w -> b t c h w")
            cond_bt = rearrange(torch.stack(self.conds, dim=0), "t b d -> b t d")
            tau = m._val_torque_pred(z_bt, cond_bt)[:, -1]           # (B,8) raw N·m

        # bound memory: only the last n_tokens frames are ever used as context / torque window
        if self.xs.shape[0] > self.n_tokens:
            self.xs = self.xs[-self.n_tokens:]
            self.conds = self.conds[-self.n_tokens:]

        s_hat_raw = m._unnorm_q(self.s_hat)[0].float().cpu().numpy()
        return z_disp, tau, s_hat_raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="stage2_state checkpoint .ckpt (LatentWorldModelState)")
    ap.add_argument("--stage3-ckpt", default=None,
                    help="optional stage-3 decoder to graft. Use the state-matched one "
                         "(world_ft_ckpts/stage3_state, from stage-1 2026-07-16_14-12-06), NOT the old "
                         "world_ft_ckpts/stage3 (trunk stage-1, incompatible latent space).")
    ap.add_argument("--seed-episode", required=True, help="hdf5 for the fixed start pose/scene")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--leader-right", default="leader_right")
    ap.add_argument("--hz", type=float, default=10.0, help="sim step rate (training cadence = 10)")
    ap.add_argument("--n-context", type=int, default=None, help="latent context frames (default: model n_tokens)")
    ap.add_argument("--dec-infer-steps", type=int, default=3,
                    help="decoder (display) diffusion steps (default 3 = original codebase value)")
    ap.add_argument("--async-decode", action=argparse.BooleanOptionalAction, default=True,
                    help="decode the display on a background CUDA stream so the sim/torque loop keeps 10Hz")
    ap.add_argument("--record", action=argparse.BooleanOptionalAction, default=True,
                    help="save predicted frames -> rollout.mp4 and predicted torque -> torque.png + session.npz")
    ap.add_argument("--record-dir", default="wm_sessions_state", help="parent dir for per-session recordings")
    ap.add_argument("--act-record", action=argparse.BooleanOptionalAction, default=True,
                    help="also save each episode in the ACT training layout "
                         "(camera_head.mp4 + low_dim_npys/), decoded in the background at save")
    ap.add_argument("--act-record-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "sim_state_training"),
                    help="parent dir for ACT-format episodes (STATE-pipeline dataset -- kept "
                         "separate from the trunk-collected sim_training)")
    ap.add_argument("--act-save-torque", action=argparse.BooleanOptionalAction, default=True,
                    help="also save the predicted joint torque alongside the ACT data")
    ap.add_argument("--resume-delay-s", type=float, default=3.0,
                    help="after s (save), hold the seed frame for this many seconds (time to move "
                         "the leader back), then the sim goes LIVE; ENTER starts recording")
    ap.add_argument("--rehome-eps-deg", type=float, default=5.0,
                    help="WARN (never block) if the leader is more than this many degrees from the "
                         "seed start pose when the sim resumes after a save")
    ap.add_argument("--display-size", type=int, default=512)
    ap.add_argument("--publish-torque", action="store_true",
                    help="milestone 2: predict + publish total joint torque (raw N·m) to "
                         "virtual_follower_right_state:torque_pred_nm for the leader force-feedback loop")
    ap.add_argument("--virtual-follower", default="virtual_follower_right",
                    help="redis name for the published predicted torque (must match the leader script)")
    ap.add_argument("--display-every-n", type=int, default=1,
                    help="(--no-async-decode only) decode+show every N steps")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                    help="inference precision; float32 is the safe default (matches the validated coupled rollout)")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device, stage3_ckpt=args.stage3_ckpt)
    model.dec_infer_steps = args.dec_infer_steps
    res = int(cfg.dataset.resolution)
    n_ctx = args.n_context or int(getattr(model, "n_tokens", 10))
    async_decode = args.async_decode and device.startswith("cuda")
    amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    amp_on = args.dtype != "float32" and device.startswith("cuda")
    angle_key = f"{args.leader_right}_state:actual_angle_rad"
    torque_key = f"{args.virtual_follower}_state:torque_pred_nm"
    torque_ts_key = f"{args.virtual_follower}_state:timestamp"
    print(f"[wm-state] device={device} dtype={args.dtype} obs_keys={list(model.obs_keys)} res={res} "
          f"n_ctx={n_ctx} dyn_infer_steps={model.dyn_infer_steps} dec_infer_steps={model.dec_infer_steps} "
          f"publish_torque={args.publish_torque} async_decode={async_decode} record={args.record}")

    z0, s0_raw, a0_raw, seed_gripper = seed_from_episode(model, args.seed_episode, res, device, amp_dtype, amp_on)
    roller = StateOnlineRollout(model, z0, s0_raw, a0_raw, device, amp_dtype, amp_on, n_ctx)
    print(f"[wm-state] seeded from {args.seed_episode} (gripper held at {seed_gripper:+.3f}, "
          f"s0[:3]={s0_raw[0, :3].cpu().numpy().round(3)})")

    client = RedisClient(host=args.redis_host, port=args.redis_port)
    win = "world_model_follower_state (pred)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.display_size, args.display_size)

    def _decode_show(z_last: torch.Tensor, label: str) -> np.ndarray:
        with _DECODE_LOCK, torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
            img = render_img_cm(model, z_last, res, model.normalizer, num_views=model.num_views)
        img = img[0, :3].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (args.display_size, args.display_size), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, img)
        cv2.waitKey(1)
        return img

    last_frame = _decode_show(z0[:, -1], "seed")
    seed_frame = last_frame.copy()             # reused at every between-episode reset (no re-decode)
    seed_q7 = a0_raw[0, :7].float().cpu().numpy().astype(np.float64)
    decoder = None
    if async_decode:
        decoder = _AsyncDecoder(model, res, args.display_size, device, amp_dtype, amp_on)
        decoder.start()

    # MULTI-EPISODE flow (same as wm_follower_worker): ENTER = start recording; s = queue the
    # episode for BACKGROUND save, re-seed the coupled rollout, hold --resume-delay-s, then LIVE
    # (recording starts on the next ENTER; warns -- never blocks -- if the leader is far from the
    # seed pose); q = quit (queued saves are drained). Torque publishing pauses during the hold so
    # the force-feedback watchdog zeroes the reflected force and the leader moves freely.
    saver = _BackgroundSaver()
    recorder = None
    act_recorder = None
    recording = False
    episode_idx = 0
    resume_at = None

    def _start_recording():
        nonlocal recorder, act_recorder, recording
        if args.record:
            recorder = _SessionRecorder(args.record_dir, args.hz)
        if args.act_record:
            act_recorder = _ACTRecorder(args.act_record_dir, model, res, model.num_views,
                                        device, amp_dtype, amp_on, args.hz, args.act_save_torque)
        recording = True

    def _hud(frame):
        """REC/IDLE indicator + pending-saves counter (display-only copy)."""
        img = frame.copy()
        if recording:
            cv2.circle(img, (args.display_size - 30, 26), 10, (0, 0, 255), -1)
            cv2.putText(img, "REC", (args.display_size - 92, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.circle(img, (args.display_size - 30, 26), 10, (0, 200, 0), -1)
            cv2.putText(img, "IDLE", (args.display_size - 100, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        if saver.pending:
            cv2.putText(img, f"saving {saver.pending}", (8, args.display_size - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return img

    print("[wm-state] waiting for leader state on redis ...")
    print("[wm-state] focus the prediction window, then: ENTER = start episode | "
          "s = save (background) & reset | q = quit")

    dt_target = 1.0 / args.hz
    step = 0
    try:
        while True:
            t0 = time.time()
            ang = client.stream_get_batch({angle_key: np.ndarray})[angle_key]
            if ang is None:
                time.sleep(0.02)
                continue

            # post-save hold: freeze at the seed, then go LIVE (recording starts on ENTER)
            if resume_at is not None:
                remaining = resume_at - time.time()
                if remaining > 0:
                    frame = seed_frame.copy()
                    cv2.putText(frame, f"sim live in {remaining:.0f}s (ENTER = now, q = quit)",
                                (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    if saver.pending:
                        cv2.putText(frame, f"saving {saver.pending} episode(s) in background",
                                    (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.imshow(win, frame)
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord("q"):
                        print("[wm-state] q -> quitting.")
                        break
                    if key not in (13, 10):  # ENTER skips the hold
                        continue
                resume_at = None
                dmax_deg = float(np.rad2deg(np.max(np.abs(ang[:7] - seed_q7))))
                if dmax_deg > args.rehome_eps_deg:
                    print(f"[wm-state] WARNING: leader is {dmax_deg:.0f} deg from the start pose as "
                          f"the sim resumes -- let it settle near the seed before recording.")
                print(f"[wm-state] sim LIVE -- press ENTER to record episode {episode_idx + 1}.")

            # live q_des: 7 leader arm joints + held seed gripper (8th dim, not driven)
            q_des = np.concatenate([ang[:7].astype(np.float32), [seed_gripper]])
            q_des_t = torch.from_numpy(q_des)[None].to(device=device, dtype=torch.float32)

            z_disp, tau, s_hat_raw = roller.step(q_des_t)
            tau_np = tau[0].float().cpu().numpy().astype(np.float64)
            if args.publish_torque:
                client.stream_add_batch(
                    {torque_key: tau_np, torque_ts_key: time.time()},
                    {torque_key: 1, torque_ts_key: 1},
                )

            if decoder is not None:
                decoder.post(z_disp, f"step {step}")
                frame = decoder.latest_frame()
                if frame is not None:
                    last_frame = frame
                cv2.imshow(win, _hud(last_frame))
            elif step % args.display_every_n == 0:
                last_frame = _decode_show(z_disp, f"step {step}")
                cv2.imshow(win, _hud(last_frame))

            if recording:
                if recorder is not None:
                    recorder.add_frame(last_frame)
                    recorder.add_torque(tau_np, q_des)
                if act_recorder is not None:
                    # pair the display latent (what the frame decodes from) with this step's q_des
                    act_recorder.add(z_disp, q_des, tau_np)

            step += 1
            elapsed = time.time() - t0
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
            elif step % 20 == 0:
                print(f"[wm-state] step {step}: {1.0/elapsed:.1f} Hz (slower than {args.hz:.0f} Hz target)")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[wm-state] q -> DISCARDING the in-progress recording and quitting." if recording
                      else "[wm-state] q -> quitting.")
                break
            elif key == ord("s"):
                if recording:
                    saver.submit(act_recorder, recorder)
                    recorder = act_recorder = None
                    recording = False
                    episode_idx += 1
                    # instant re-seed of the coupled rollout (scene latents + GRU state + conds)
                    roller = StateOnlineRollout(model, z0, s0_raw, a0_raw, device,
                                                amp_dtype, amp_on, n_ctx)
                    step = 0
                    resume_at = time.time() + max(args.resume_delay_s, 0.0)
                    last_frame = seed_frame.copy()
                    cv2.imshow(win, _hud(last_frame))
                    print(f"[wm-state] s -> episode {episode_idx} SAVING IN BACKGROUND "
                          f"({saver.pending} pending). Sim reset to the seed; sim resumes in "
                          f"{max(args.resume_delay_s, 0.0):g}s -- move the leader back, then "
                          f"ENTER to record.")
                else:
                    print("[wm-state] s pressed but nothing recording (ENTER starts an episode).")
            elif key in (13, 10) and not recording:   # ENTER (CR/LF)
                _start_recording()
                print("[wm-state] ENTER -> recording STARTED. s = save & continue, "
                      "q = discard & quit.")
    except KeyboardInterrupt:
        if recording:
            print("\n[wm-state] interrupted -> queuing the in-progress recording for save.")
            saver.submit(act_recorder, recorder)
            recorder = act_recorder = None
            recording = False
        else:
            print("\n[wm-state] interrupted.")
    finally:
        if decoder is not None:
            decoder.stop()
        if recording:  # reached via q with an active recording -> discard it
            if recorder is not None:
                recorder.discard()
            if act_recorder is not None:
                act_recorder.discard()
        saver.drain()  # finish any queued background saves before exiting
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
