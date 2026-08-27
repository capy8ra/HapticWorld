#!/usr/bin/env python
"""World-model "virtual follower" worker — milestone 1: visual only, no force yet.

Reads the real RIGHT-leader joint angles from redis (published by the nova damiao
controller in robot/), drives the trained world model as a virtual follower, and
shows the decoded predicted camera view in a window. The human moves the real leader
and watches the simulated follower move on screen.

Seeding (fixed start pose):
  * the sim latent is seeded from frame 0 of a chosen validation episode,
  * the action context is seeded from that episode's action[0],
  * the real leader must be homed to the SAME pose first
    (robot/world_ft/leader_home_and_float.py --seed-episode <same hdf5>).

Gripper: joint 8 is NOT used. The live action's 8th dim is held at the seed
episode's action[0][7] (kept in-distribution, matches the static-gripper seed).

Env: runs in the WORLD_FT (iws) env (torch + interactive_world_sim). It needs
nova.redis_client, which only imports redis+numpy — make it importable with:
    python scripts/inference/wm_follower_worker.py ...   # from HapticWorld/worldsim

Usage:
  python scripts/inference/wm_follower_worker.py \
    --ckpt outputs/world_ft_stage_2/<ts>/checkpoints/last.ckpt \
    --seed-episode data/world_ft_v3/val/episode_0.hdf5
"""
import argparse
import os
import sys
import threading
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # sys.path bootstrap: worldsim root
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

# The diffusion decoder is NOT thread-safe: concurrent render_img_cm calls (display decoder +
# background episode saver) corrupt each other's outputs (measured: black frames, NaNs, garbage).
# Every decode, on any thread, must hold this lock. The dynamics/torque path needs no lock.
_DECODE_LOCK = threading.Lock()
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model_torque import (
    LatentWorldModelTorque,
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


def load_model(ckpt_path: str, device: str, stage3_ckpt: str | None = None) -> tuple[LatentWorldModelTorque, object]:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path} (need the run's .hydra/config.yaml)")
    # The training config uses custom OmegaConf resolvers (registered in main.py):
    #   ${torch:float} -> torch.float, ${eval:'...'} -> eval(...) with np available.
    # Register them here too, or resolving cfg.algorithm.dtype / num_views raises.
    if not OmegaConf.has_resolver("torch"):
        OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
    cfg = OmegaConf.load(cfg_path)
    load_dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.algorithm.load_ae = None
    model = LatentWorldModelTorque.load_from_checkpoint(
        ckpt_path, cfg=cfg.algorithm, map_location=device,
        dtype=load_dtype, strict=False, weights_only=False,
    )
    model.dynamics = model.dynamics.to(load_dtype)

    # Stage 3 fine-tunes ONLY the decoder (to be robust to the dynamics' latent errors),
    # so graft its decoder over the stage-2 model. Encoder + dynamics + torque head stay
    # from the stage-2 ckpt; the latent space is shared (both froze stage-1's encoder).
    if stage3_ckpt is not None:
        sd = torch.load(stage3_ckpt, map_location=device, weights_only=False)
        sd = sd.get("state_dict", sd)
        dec_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
        if not dec_sd:
            raise ValueError(f"No decoder.* weights in stage-3 ckpt {stage3_ckpt}")
        missing, unexpected = model.decoder.load_state_dict(dec_sd, strict=False)
        model.decoder = model.decoder.to(load_dtype)
        print(f"[wm] grafted stage-3 decoder from {stage3_ckpt} "
              f"({len(dec_sd)} tensors; missing={len(missing)} unexpected={len(unexpected)})")

    model.eval()
    # Enable all SDPA kernels: flash (fast, needs fp16/bf16) is used when eligible, else the
    # MATH/efficient kernels cover fp32.
    for m in model.modules():
        if hasattr(m, "cuda_backends"):
            m.cuda_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    return model, cfg


@torch.no_grad()
def seed_from_episode(
    model: LatentWorldModelTorque, seed_episode: str, res: int, device: str,
    amp_dtype: torch.dtype = torch.float32, amp_on: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (z_ctx (1,1,C,H,W), a_ctx (1,1,A) normalized, seed_gripper)."""
    obs_key = model.obs_keys[0]
    with h5py.File(seed_episode, "r") as f:
        frame0 = f["obs"]["images"][obs_key][0]      # (H,W,3) uint8
        action0 = f["action"][0].astype(np.float32)  # (8,) right q_des
    img = center_crop(frame0, res)
    img = cv2.resize(img, (res[1], res[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    img_t = model.normalizer[obs_key].normalize(img_t)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
        z0 = model.encoder_forward(img_t).float()[:, None]               # (1,1,C,H,W) fp32 state
    a0 = model.normalizer["action"].normalize(
        torch.from_numpy(action0)[None, None].to(device)
    ).clamp(-1.0, 1.0)                                                    # (1,1,A)
    return z0, a0, float(action0[7])


class _AsyncDecoder:
    """Render the latest latent on a background CUDA stream so the (slow, display-only)
    decode never blocks the dynamics/torque loop.

    ``post(latent, label)`` hands the newest latent to the worker thread (older un-decoded
    latents are dropped); the thread renders it to a display-ready BGR frame on its own CUDA
    stream (so the main thread's sync points don't wait on it). ``latest_frame()`` returns
    that frame for the MAIN thread to ``cv2.imshow`` — OpenCV GUI calls must stay on the main
    thread. The decode runs at whatever GPU time is left over; the sim keeps its cadence.
    """

    def __init__(self, model: LatentWorldModelTorque, res: int, display_size: int, device: str,
                 amp_dtype: torch.dtype = torch.float32, amp_on: bool = False):
        self._model = model
        self._res = res
        self._display_size = display_size
        self._amp_dtype = amp_dtype
        self._amp_on = amp_on
        self._lock = threading.Lock()
        self._pending: tuple | None = None   # (id, latent, event, label) newest posted
        self._posted_id = 0
        self._done_id = -1
        self._frame = None                   # latest display-ready BGR uint8 image
        self._stop = False
        self._cuda = str(device).startswith("cuda")
        self._stream = torch.cuda.Stream(device=device) if self._cuda else None
        self._thread = threading.Thread(target=self._run, name="wm-decode", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def post(self, latent: torch.Tensor, label: str) -> None:
        lat = latent.clone()                 # detach from the rolling z_ctx buffer
        ev = None
        if self._cuda:
            ev = torch.cuda.Event()
            ev.record()                      # on the default stream, after the clone
        with self._lock:
            self._posted_id += 1
            self._pending = (self._posted_id, lat, ev, label)

    def latest_frame(self):
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        m = self._model
        while not self._stop:
            with self._lock:
                item = self._pending
            if item is None or item[0] == self._done_id:
                time.sleep(0.003)
                continue
            pid, lat, ev, label = item
            with _DECODE_LOCK, torch.no_grad(), \
                    torch.autocast("cuda", dtype=self._amp_dtype, enabled=self._amp_on):
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        if ev is not None:
                            self._stream.wait_event(ev)   # ensure the clone finished
                        img = render_img_cm(m, lat, self._res, m.normalizer, num_views=m.num_views)
                        img = img[0, :3].permute(1, 2, 0).clamp(0, 1).float()
                    self._stream.synchronize()
                    img = img.cpu().numpy()
                else:
                    img = render_img_cm(m, lat, self._res, m.normalizer, num_views=m.num_views)
                    img = img[0, :3].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.resize(img, (self._display_size, self._display_size), interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            with self._lock:
                self._frame = img
                self._done_id = pid


class _SessionRecorder:
    """Save a live session like rollout_world_ft.py, minus ground truth (live teleop has none):
    the displayed predicted frames -> ``rollout.mp4`` (H.264, browser-playable, written
    incrementally via ffmpeg), and the predicted joint torque -> ``torque.png`` + ``session.npz``
    (also stores the leader action per frame). Files go under ``<record_dir>/<timestamp>/``.
    """

    def __init__(self, record_dir: str, fps: float):
        import datetime
        self.dir = Path(record_dir) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._proc = None
        self._t0 = time.time()
        self._t: list = []
        self._torque: list = []
        self._action: list = []
        print(f"[wm] recording session -> {self.dir}")

    def add_frame(self, bgr: np.ndarray) -> None:
        import subprocess
        if self._proc is None:
            h, w = bgr.shape[:2]
            self._proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", f"{self._fps:g}", "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-preset", "fast", "-movflags", "+faststart",
                 str(self.dir / "rollout.mp4")],
                stdin=subprocess.PIPE,
            )
        try:
            self._proc.stdin.write(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def add_torque(self, tau: np.ndarray, action: np.ndarray) -> None:
        self._t.append(time.time() - self._t0)
        self._torque.append(np.asarray(tau, dtype=np.float32))
        self._action.append(np.asarray(action, dtype=np.float32))

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.wait()
        t = np.asarray(self._t, dtype=np.float64)
        tq = np.stack(self._torque) if self._torque else np.zeros((0, 8), np.float32)
        ac = np.stack(self._action) if self._action else np.zeros((0, 8), np.float32)
        np.savez(self.dir / "session.npz", t=t, torque_pred_nm=tq, action=ac)
        if len(tq):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 4, figsize=(16, 6), squeeze=False)
            for j in range(tq.shape[1]):
                ax = axes[j // 4][j % 4]
                ax.plot(t, tq[:, j], lw=1.0)
                ax.set_title(f"joint {j}", fontsize=9)
                ax.tick_params(labelsize=6)
            fig.suptitle("live predicted joint torque (N·m) vs time (s)")
            fig.tight_layout()
            fig.savefig(self.dir / "torque.png", dpi=120)
            plt.close(fig)
        print(f"[wm] saved session ({len(t)} torque samples) -> {self.dir}")

    def discard(self) -> None:
        import shutil
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        shutil.rmtree(self.dir, ignore_errors=True)
        print(f"[wm] discarded session -> {self.dir}")


class _ACTRecorder:
    """Save a session in the ACT training layout (state-free / image-only policy):

        <dir>/<ts>/camera_head.mp4                                      clean predicted frames (fps=hz)
        <dir>/<ts>/low_dim_npys/
            follower_right_mit_command:q.npy          (N, 8)  action target (leader q7 + seed gripper)
            follower_right_mit_command:timestamp.npy  (N,)    = arange(N)/hz  (aligned with the mp4 pts)
            follower_controller_state:torque_pred_nm.npy (N, 8)  predicted joint torque   [if save_torque]
            follower_controller_state:timestamp.npy      (N,)                             [if save_torque]

    Latents (not decoded images) are stored during the run so the sim loop stays real-time; they are
    decoded to clean, exactly-paired frames at close (no display label, no async frame-dropping). This
    layout drops into the image-only ACT dataset (camera_head.mp4 + follower_right_mit_command:q).
    """

    def __init__(self, record_dir, model, res, num_views, device, amp_dtype, amp_on, hz, save_torque):
        import datetime
        self.dir = Path(record_dir) / datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        (self.dir / "low_dim_npys").mkdir(parents=True, exist_ok=True)
        self._model = model
        self._res = res
        self._num_views = num_views
        self._device = device
        self._amp_dtype = amp_dtype
        self._amp_on = amp_on
        self._hz = float(hz)
        self._save_torque = save_torque
        self._lat: list = []
        self._action: list = []
        self._torque: list = []
        print(f"[wm] ACT recording (image-only, save_torque={save_torque}) -> {self.dir}")

    def add(self, latent: torch.Tensor, action: np.ndarray, torque, real_tau=None) -> None:
        # store the latent the current frame is decoded from, paired with the action applied this step
        self._lat.append(latent.detach().to("cpu", copy=True))
        self._action.append(np.asarray(action, dtype=np.float32))
        if self._save_torque:
            t = torque if torque is not None else np.zeros(8, dtype=np.float32)
            self._torque.append(np.asarray(t, dtype=np.float32))
        # real follower's measured interaction torque (conducted rig): saved ALONGSIDE
        # the sim prediction for per-episode real-vs-sim force comparison
        if not hasattr(self, "_real_tau"):
            self._real_tau = []
        self._real_tau.append(None if real_tau is None
                              else np.asarray(real_tau, dtype=np.float32))

    @torch.no_grad()
    def close(self) -> None:
        import subprocess
        n = len(self._lat)
        if n == 0:
            print("[wm] ACT recording: nothing captured.")
            return
        ld = self.dir / "low_dim_npys"
        proc = None
        written = 0
        print(f"[wm] decoding {n} clean frames -> camera_head.mp4 (please wait, do not interrupt) ...")
        # own CUDA stream + tiny per-frame yield: close() now runs on the background saver
        # thread, and must not stall the live sim loop's 10 Hz cadence
        stream = (torch.cuda.Stream(device=self._device)
                  if str(self._device).startswith("cuda") else None)
        try:
            for lat in self._lat:
                with _DECODE_LOCK, torch.autocast("cuda", dtype=self._amp_dtype, enabled=self._amp_on):
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            img = render_img_cm(self._model, lat.to(self._device), self._res,
                                                self._model.normalizer, num_views=self._num_views)
                        stream.synchronize()
                    else:
                        img = render_img_cm(self._model, lat.to(self._device), self._res,
                                            self._model.normalizer, num_views=self._num_views)
                img = img[0, :3].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
                img = (img * 255).astype(np.uint8)          # clean RGB (res, res, 3)
                if proc is None:
                    h, w = img.shape[:2]
                    proc = subprocess.Popen(
                        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{w}x{h}", "-r", f"{self._hz:g}", "-i", "-", "-c:v", "libx264",
                         "-pix_fmt", "yuv420p", "-preset", "fast", "-movflags", "+faststart",
                         str(self.dir / "camera_head.mp4")],
                        stdin=subprocess.PIPE,
                    )
                proc.stdin.write(np.ascontiguousarray(img).tobytes())
                written += 1
                time.sleep(0.002)  # yield GIL/GPU to the live sim between frames
        finally:
            if proc is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait()
            # Save the low-dim data TRUNCATED to the frames actually written, so camera_head.mp4 and
            # the action npys always have the same length and stay aligned -- even if the decode was
            # cut short (e.g. Ctrl-C). Runs in `finally` so an interrupt mid-decode still saves.
            m = written
            if m > 0:
                ts = np.arange(m, dtype=np.float64) / self._hz
                np.save(ld / "follower_right_mit_command:q.npy", np.stack(self._action[:m]))
                np.save(ld / "follower_right_mit_command:timestamp.npy", ts)
                if self._save_torque and self._torque:
                    np.save(ld / "follower_controller_state:torque_pred_nm.npy", np.stack(self._torque[:m]))
                    np.save(ld / "follower_controller_state:timestamp.npy", ts)
                real = getattr(self, "_real_tau", [])[:m]
                if any(r is not None for r in real):
                    filled = [r if r is not None else np.zeros(8, dtype=np.float32)
                              for r in real]
                    np.save(ld / "follower_controller_state:tau_interaction_real_R.npy",
                            np.stack(filled))
                msg = f"[wm] saved ACT session ({m} frames, {m / self._hz:.1f}s) -> {self.dir}"
                if m != n:
                    msg += f"  [WARNING: decode was cut short at {m}/{n} frames]"
                print(msg)
            else:
                print("[wm] ACT session: no frames written.")

    def discard(self) -> None:
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)
        print(f"[wm] discarded ACT session -> {self.dir}")


class _BackgroundSaver:
    """Runs recorder ``.close()`` calls on a worker thread, so the slow part of saving an
    episode (the clean per-frame decode in _ACTRecorder.close) overlaps with recording the
    NEXT episode instead of blocking the operator."""

    def __init__(self):
        import queue
        self._q = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="wm-saver", daemon=True)
        self._thread.start()

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def submit(self, *recorders) -> None:
        with self._lock:
            self._pending += 1
        self._q.put([r for r in recorders if r is not None])

    def _run(self) -> None:
        while True:
            recs = self._q.get()
            if recs is None:
                return
            for r in recs:
                try:
                    r.close()
                except Exception as e:  # noqa: BLE001 -- one failed save must not kill the rest
                    print(f"[wm] background save ERROR ({type(r).__name__}): {e}")
            with self._lock:
                self._pending -= 1

    def drain(self) -> None:
        if self.pending:
            print(f"[wm] waiting for {self.pending} background save(s) to finish ...")
        self._q.put(None)
        self._thread.join()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Stage-2 (torque) checkpoint .ckpt (dynamics + torque head + encoder)")
    ap.add_argument("--stage3-ckpt", default=None,
                    help="optional Stage-3 checkpoint whose fine-tuned decoder is grafted over the "
                         "stage-2 decoder (sharper/robust decode of predicted latents)")
    ap.add_argument("--seed-episode", required=True, help="hdf5 to seed the fixed start pose/scene")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--leader-right", default="leader_right")
    ap.add_argument("--hz", type=float, default=10.0, help="sim step rate (training cadence = 10)")
    ap.add_argument("--rehome-eps-deg", type=float, default=5.0,
                    help="WARN (never block) if the leader is more than this many degrees from the "
                         "seed start pose when the sim resumes after a save")
    ap.add_argument("--resume-delay-s", type=float, default=3.0,
                    help="after s (save), hold the seed frame for this many seconds (time to move "
                         "the leader back), then the sim goes LIVE; press ENTER to start recording "
                         "the next episode (ENTER during the hold skips it)")
    ap.add_argument("--n-context", type=int, default=None, help="latent context frames (default: model n_tokens)")
    ap.add_argument("--dec-infer-steps", type=int, default=3,
                    help="decoder (display) diffusion steps (default 3 = the original codebase's value; "
                         "fewer -> visible speckle noise). With --async-decode the decode is off the "
                         "critical path, so 3 stays clean without slowing the sim.")
    ap.add_argument("--async-decode", action=argparse.BooleanOptionalAction, default=True,
                    help="decode the display in a background thread (own CUDA stream) so the "
                         "dynamics/torque loop keeps its 10Hz cadence while the image renders crisply. "
                         "Use --no-async-decode for the old in-loop decode (then tune --display-every-n).")
    ap.add_argument("--record", action=argparse.BooleanOptionalAction, default=True,
                    help="save each session's predicted frames -> rollout.mp4 and predicted joint "
                         "torque -> torque.png + session.npz (like the rollout, minus GT). Torque is "
                         "computed for recording even without --publish-torque. --no-record to disable.")
    ap.add_argument("--live-gripper", action="store_true",
                    help="drive the sim's 8th action dim from the LIVE leader trigger instead "
                         "of holding the seed value. Required for tasks where the gripper does "
                         "the work (microwave door: grasping the handle); wipe/box-flip had a "
                         "static gripper and should leave this off.")
    ap.add_argument("--settle-flow", action="store_true",
                    help="collection flow with an un-recorded settling phase, matching real "
                         "teleop collection: sim starts frozen at the seed; ENTER resets the sim "
                         "and goes LIVE (arm walks from the seed pose to wherever you have put "
                         "the leader, nothing recorded); ENTER again starts recording; s = save, "
                         "d = discard -- both return to frozen, ready for the next ENTER.")
    ap.add_argument("--record-dir", default="wm_sessions_no_torque", help="parent dir for per-session recordings")
    ap.add_argument("--display-size", type=int, default=1024)
    ap.add_argument("--publish-torque", action="store_true",
                    help="milestone 2: also predict joint torque each step and publish it (raw N·m, total "
                         "joint torque) to virtual_follower_right_state:torque_pred_nm for the leader "
                         "force-feedback loop. Gravity-comp subtraction happens on the nova/leader side.")
    ap.add_argument("--virtual-follower", default="virtual_follower_right",
                    help="redis name for the published predicted torque (must match the leader script)")
    ap.add_argument("--conducted", action="store_true",
                    help="rollout-rig mode (teleop_wm_rollout_follower.yaml): hold the sim frozen "
                         "at the seed until the conductor publishes wm_rollout:command. ` = start "
                         "rollout (auto-records), s = stop+save (robot re-homes), d = discard+reset "
                         "(robot keeps teleoping; ` re-arms). Keyboard q still quits.")
    ap.add_argument("--display-every-n", type=int, default=1,
                    help="(--no-async-decode only) decode+show every N steps so dynamics+torque still hold "
                         "10Hz with the in-loop decode. Ignored when --async-decode is on (the default).")
    ap.add_argument("--act-record", action=argparse.BooleanOptionalAction, default=True,
                    help="also save the session in the ACT training layout "
                         "(camera_head.mp4 + low_dim_npys/) for the state-free / image-only ACT.")
    ap.add_argument("--act-record-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "sim_training"),
                    help="parent dir for the ACT-format sessions (one subfolder per session)")
    ap.add_argument("--act-save-torque", action=argparse.BooleanOptionalAction, default=True,
                    help="also save the predicted joint torque alongside the ACT data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                    help="inference precision. bf16/fp16 use the tensor cores + flash attention "
                         "(much faster on Blackwell/A100), but can slightly change the diffusion output "
                         "-- eyeball the image and check torque. float32 is the safe default.")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device, stage3_ckpt=args.stage3_ckpt)
    model.dec_infer_steps = args.dec_infer_steps  # decode is display-only; see --dec-infer-steps
    res = ((int(cfg.dataset.res_h), int(cfg.dataset.res_w)) if "res_h" in cfg.dataset
           else (int(cfg.dataset.resolution),) * 2)  # square-era ckpts (wipe/microwave)
    n_ctx = args.n_context or int(getattr(model, "n_tokens", 10))
    async_decode = args.async_decode and device.startswith("cuda")
    # Half-precision compute via autocast: bf16/fp16 run matmul/conv/attention on the tensor cores
    # (+ flash attention) while the model + stored latents stay fp32. Much faster on Blackwell/A100.
    amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    amp_on = args.dtype != "float32" and device.startswith("cuda")
    angle_key = f"{args.leader_right}_state:actual_angle_rad"
    torque_key = f"{args.virtual_follower}_state:torque_pred_nm"
    torque_ts_key = f"{args.virtual_follower}_state:timestamp"
    print(f"[wm] device={device} dtype={args.dtype} obs_keys={list(model.obs_keys)} res={res} "
          f"n_ctx={n_ctx} dec_infer_steps={model.dec_infer_steps} publish_torque={args.publish_torque} "
          f"async_decode={async_decode} record={args.record}")

    z_ctx, a_ctx, seed_gripper = seed_from_episode(model, args.seed_episode, res, device, amp_dtype, amp_on)
    z0, a0 = z_ctx.clone(), a_ctx.clone()  # kept for instant re-seeding between episodes
    with h5py.File(args.seed_episode, "r") as f:
        seed_q7 = f["action"][0][:7].astype(np.float64)  # leader start pose (re-home check)
    print(f"[wm] seeded from {args.seed_episode} (gripper held at {seed_gripper:+.3f})")

    client = RedisClient(host=args.redis_host, port=args.redis_port)
    win = "world_model_follower (pred)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.display_size, args.display_size)

    # show the seeded frame once before any motion; returns the displayed BGR frame (for recording)
    def _show(z_last: torch.Tensor, label: str) -> np.ndarray:
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

    last_frame = _show(z_ctx[:, -1], "seed")   # show the seed frame immediately (main-thread decode)
    seed_frame = last_frame.copy()             # reused at every between-episode reset (no re-decode)
    decoder = None
    if async_decode:
        decoder = _AsyncDecoder(model, res, args.display_size, device, amp_dtype, amp_on)
        decoder.start()
    # MULTI-EPISODE flow: ENTER = start recording; s = queue the episode for BACKGROUND save,
    # reset the sim to the seed, hold for --resume-delay-s (move the leader back), then the sim
    # goes LIVE -- recording starts on the next ENTER (it warns, never blocks, if the leader is
    # far from the seed pose when the sim resumes); q = quit (discards an in-progress recording;
    # queued saves are drained). Torque publishing pauses during the hold, so the force-feedback
    # watchdog zeroes the reflected force and the leader moves freely.
    saver = _BackgroundSaver()
    recorder = None
    act_recorder = None
    recording = False
    episode_idx = 0
    resume_at = None  # end time of the post-save countdown (recording auto-starts then)

    def _start_recording():
        nonlocal recorder, act_recorder, recording
        if args.record:
            recorder = _SessionRecorder(args.record_dir, args.hz)
        if args.act_record:
            act_recorder = _ACTRecorder(args.act_record_dir, model, res, model.num_views,
                                        device, amp_dtype, amp_on, args.hz, args.act_save_torque)
        recording = True

    def _hud(frame):
        """REC/IDLE indicator + pending-saves counter, drawn on a COPY (display only --
        recorded frames stay clean)."""
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
    # torque is computed regardless so it is available the moment recording starts
    need_torque = args.publish_torque or args.record or (args.act_record and args.act_save_torque)
    print("[wm] waiting for leader state on redis ...")
    print("[wm] focus the prediction window, then: ENTER = start episode | "
          "s = save (background) & reset for the next | q = quit")

    dt_target = 1.0 / args.hz
    step = 0
    # frozen-at-seed state: conducted waits for ` over redis, settle-flow waits for ENTER here
    armed = args.conducted or args.settle_flow
    cmd_key, cmd_ts_key = "wm_rollout:command", "wm_rollout:timestamp"
    last_cmd_ts = None
    if args.conducted:
        print("[wm] CONDUCTED mode: ` = start rollout | s = stop+save | d = discard+reset "
              "(keys come from wm_rollout_conductor via redis; q here still quits)")
    try:
        while True:
            t0 = time.time()
            ang = client.stream_get_batch({angle_key: np.ndarray})[angle_key]
            if ang is None:
                time.sleep(0.02)
                continue

            if args.conducted:
                b = client.stream_get_batch({cmd_key: str, cmd_ts_key: float})
                cmd, cts = b[cmd_key], b[cmd_ts_key]
                if cts is not None and cts != last_cmd_ts:
                    last_cmd_ts = cts
                    if cmd == "start" and armed:
                        armed = False
                        resume_at = None
                        _start_recording()
                        print(f"[wm] ` -> rollout {episode_idx + 1} LIVE (recording).")
                    elif cmd == "stop" and not armed:
                        if recording:
                            saver.submit(act_recorder, recorder)
                            recorder = act_recorder = None
                            recording = False
                            episode_idx += 1
                            print(f"[wm] s -> rollout {episode_idx} saving in background; "
                                  f"sim reset to seed (robot re-homing).")
                        z_ctx, a_ctx = z0.clone(), a0.clone()
                        step = 0
                        armed = True
                        last_frame = seed_frame.copy()
                    elif cmd == "reset" and not armed:
                        recorder = act_recorder = None
                        recording = False
                        z_ctx, a_ctx = z0.clone(), a0.clone()
                        step = 0
                        armed = True
                        last_frame = seed_frame.copy()
                        print("[wm] d -> rollout DISCARDED; sim reset to seed, ` re-arms.")
                if armed:
                    frame = seed_frame.copy()
                    cv2.putText(frame, "ARMED - ` starts the rollout (q = quit)",
                                (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    cv2.imshow(win, _hud(frame))
                    if (cv2.waitKey(50) & 0xFF) == ord("q"):
                        print("[wm] q -> quitting.")
                        break
                    continue

            # --settle-flow IDLE: frozen at the seed, nothing stepping, no torque published (so
            # the force-feedback watchdog zeroes and the leader floats free while you position
            # it). ENTER re-seeds and goes LIVE: from here the sim arm walks from the seed pose
            # to wherever the leader is, UNRECORDED, so the episode itself can start clean.
            if args.settle_flow and armed:
                frame = seed_frame.copy()
                cv2.putText(frame, "IDLE - move the leader, ENTER = reset sim & go live",
                            (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                if saver.pending:
                    cv2.putText(frame, f"saving {saver.pending} episode(s) in background",
                                (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow(win, _hud(frame))
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    print("[wm] q -> quitting.")
                    break
                if key in (13, 10):
                    armed = False
                    z_ctx, a_ctx = z0.clone(), a0.clone()
                    step = 0
                    last_frame = seed_frame.copy()
                    print("[wm] ENTER -> sim RESET and LIVE (not recording). Let the arm reach "
                          "the leader pose, then ENTER again to start recording.")
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
                        print("[wm] q -> quitting.")
                        break
                    if key not in (13, 10):  # ENTER skips the hold
                        continue
                resume_at = None
                dmax_deg = float(np.rad2deg(np.max(np.abs(ang[:7] - seed_q7))))
                if dmax_deg > args.rehome_eps_deg:
                    print(f"[wm] WARNING: leader is {dmax_deg:.0f} deg from the start pose as the "
                          f"sim resumes -- let it settle near the seed before recording.")
                print(f"[wm] sim LIVE -- press ENTER to record episode {episode_idx + 1}.")

            # live action: 7 leader arm joints + gripper. --live-gripper drives the 8th dim
            # from the leader's trigger (microwave: the grasp IS the task); otherwise it is
            # held at the seed value, as wipe/box-flip's static-gripper data expects.
            grip_cmd = float(ang[7]) if args.live_gripper else seed_gripper
            action_live = np.concatenate([ang[:7].astype(np.float32), [grip_cmd]])
            a_new = model.normalizer["action"].normalize(
                torch.from_numpy(action_live)[None, None].to(device=device, dtype=model.dtype)
            ).clamp(-1.0, 1.0)

            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                action_seq = torch.cat([a_ctx, a_new], dim=1)           # (1, L+1, A)
                z_seq = model.dynamics_forward(z_ctx, action_seq)        # (1, L+1, C,H,W)
            z_new = z_seq[:, -1:].float()                               # keep latent state in fp32
            z_ctx = torch.cat([z_ctx, z_new], dim=1)[:, -n_ctx:]
            a_ctx = torch.cat([a_ctx, a_new], dim=1)[:, -n_ctx:]

            # predict the follower's (total) joint torque for the current frame; publish it (raw N·m)
            # + a timestamp for the leader force-feedback watchdog, and/or record it.
            tau_np = None
            if need_torque:
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                    tau_pred = model._val_torque_pred(z_ctx, a_ctx)[:, -1]   # (1, 8) raw N·m
                tau_np = tau_pred[0].float().cpu().numpy().astype(np.float64)
                if args.publish_torque:
                    client.stream_add_batch(
                        {torque_key: tau_np, torque_ts_key: time.time()},
                        {torque_key: 1, torque_ts_key: 1},
                    )

            if decoder is not None:
                decoder.post(z_ctx[:, -1], f"step {step}")   # hand latest latent to the decode thread
                frame = decoder.latest_frame()
                if frame is not None:
                    last_frame = frame
                cv2.imshow(win, _hud(last_frame))            # main-thread imshow (cheap)
            elif step % args.display_every_n == 0:
                last_frame = _show(z_ctx[:, -1], f"step {step}")
                cv2.imshow(win, _hud(last_frame))

            if recording:
                if recorder is not None:
                    recorder.add_frame(last_frame)
                    if tau_np is not None:
                        recorder.add_torque(tau_np, action_live)
                if act_recorder is not None:
                    # real follower's measured interaction torque, when the conducted
                    # rig is running (absent in leader-only rigs -> None, not saved)
                    real_tau = client.stream_get_batch(
                        {"follower_controller_state:tau_interaction_R": np.ndarray}
                    )["follower_controller_state:tau_interaction_R"]
                    # pair the latent the frame is decoded from with the action applied this step
                    act_recorder.add(z_ctx[:, -1], action_live, tau_np, real_tau)

            step += 1
            elapsed = time.time() - t0
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
            elif step % 20 == 0:
                print(f"[wm] step {step}: {1.0/elapsed:.1f} Hz (slower than {args.hz:.0f} Hz target)")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[wm] q -> DISCARDING the in-progress recording and quitting." if recording
                      else "[wm] q -> quitting.")
                break
            elif key == ord("d") and args.settle_flow:
                if recording:
                    recorder = act_recorder = None      # dropped, never queued for save
                    recording = False
                    z_ctx, a_ctx = z0.clone(), a0.clone()
                    step = 0
                    armed = True
                    last_frame = seed_frame.copy()
                    cv2.imshow(win, _hud(last_frame))
                    print("[wm] d -> episode DISCARDED. Sim frozen at the seed; move the leader "
                          "and press ENTER to reset for the next one.")
                else:
                    print("[wm] d pressed but nothing recording.")
            elif key == ord("s") and args.settle_flow:
                if recording:
                    saver.submit(act_recorder, recorder)
                    recorder = act_recorder = None
                    recording = False
                    episode_idx += 1
                    z_ctx, a_ctx = z0.clone(), a0.clone()
                    step = 0
                    armed = True
                    last_frame = seed_frame.copy()
                    cv2.imshow(win, _hud(last_frame))
                    print(f"[wm] s -> episode {episode_idx} SAVING IN BACKGROUND "
                          f"({saver.pending} pending). Sim frozen at the seed; move the leader "
                          "and press ENTER to reset for the next one.")
                else:
                    print("[wm] s pressed but nothing recording.")
            elif key == ord("s"):
                if recording:
                    saver.submit(act_recorder, recorder)
                    recorder = act_recorder = None
                    recording = False
                    episode_idx += 1
                    # instant re-seed; sim resumes after the hold, recording on next ENTER
                    z_ctx, a_ctx = z0.clone(), a0.clone()
                    step = 0
                    resume_at = time.time() + max(args.resume_delay_s, 0.0)
                    # auto re-home the floating leader (leader_home_and_float.py rigs;
                    # other leader drivers ignore this key)
                    client.stream_add_batch(
                        {"wm_leader:command": "rehome", "wm_leader:timestamp": time.time()},
                        {"wm_leader:command": 10, "wm_leader:timestamp": 10},
                    )
                    hint = (f"sim resumes in {max(args.resume_delay_s, 0.0):g}s -- leader "
                            f"re-homing itself; press ENTER to record when it settles")
                    last_frame = seed_frame.copy()
                    cv2.imshow(win, _hud(last_frame))
                    print(f"[wm] s -> episode {episode_idx} SAVING IN BACKGROUND "
                          f"({saver.pending} pending). Sim reset to the seed; {hint}.")
                else:
                    print("[wm] s pressed but nothing recording (ENTER starts an episode).")
            elif key in (13, 10) and not recording:   # ENTER (CR/LF)
                _start_recording()
                print("[wm] ENTER -> recording STARTED. Press s to save & continue, "
                      "q to discard & quit.")
    except KeyboardInterrupt:
        if recording:
            print("\n[wm] interrupted -> queuing the in-progress recording for save.")
            saver.submit(act_recorder, recorder)
            recorder = act_recorder = None
            recording = False
        else:
            print("\n[wm] interrupted.")
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
