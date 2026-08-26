"""Convert recorded follower interaction torque into a 6-D wrist/hand wrench.

Two versions are produced for every episode:

  * "raw" (version 1) -- uses tau_interaction as-is (quasi-static):
        tau_interaction = M(q) q_ddot + tau_friction - J^T W
        assume M q_ddot ~= 0 and friction ~= 0  =>  W ~= -pinv(J^T) tau_interaction

  * "inertiacomp" (version 2) -- subtracts the rigid-body inertia term M(q) q_ddot
    before solving (joint friction is intentionally NOT modelled, per request):
        W ~= -pinv(J^T) (tau_interaction - M(q) q_ddot)
    M(q) comes from pinocchio CRBA; q_ddot is obtained by differentiating the
    recorded joint velocity (actual_velocity_radps) and low-pass filtering.

For both, W = [Fx, Fy, Fz, Mx, My, Mz] is expressed in world-aligned axes at the
hand TCP frame (pinocchio LOCAL_WORLD_ALIGNED: rows 0:3 linear, 3:6 angular), and
only the 7 arm joints are used (the recorded 8-vector is
[shoulder x3, elbow x1, wrist x3, gripper x1]; the gripper column is dropped),
because a hand contact wrench couples into shoulder/elbow torques too.

Remaining caveats:
  * joint friction is NOT compensated -> a velocity-dependent bias remains
    (worst at low speed / stiction);
  * a grasped payload's weight and inertia are NOT in the model, so they are
    folded into the reported wrench (treat episodes as "bare hand" otherwise);
  * near kinematic singularities the J^T inverse is ill-conditioned (damped, and
    flagged via cond(JJt)).

Nothing outside world_ft is modified; nova / pinocchio are only imported.
"""

import os

import numpy as np
import pinocchio as pin
import typer
from scipy.signal import butter, filtfilt

from nova.robot.openarm1.models import OPENARM_MODEL_PATH

# qpos layout built by the teleop solvers: [L_arm(7), L_gripper(2), R_arm(7), R_gripper(2)]
ARM_QSLICE = {"L": slice(0, 7), "R": slice(9, 16)}
TCP_FRAME = {"L": "openarm_left_hand_tcp", "R": "openarm_right_hand_tcp"}
WRIST_FRAME = {"L": "L_wrist", "R": "R_wrist"}
STATE_KEY = {"L": "follower_left_state", "R": "follower_right_state"}
TAU_KEY = {"L": "follower_controller_state:tau_interaction_L",
           "R": "follower_controller_state:tau_interaction_R"}
VERSIONS = ("raw", "inertiacomp")


def _load(npy_dir: str, key: str) -> np.ndarray:
    arr = np.load(os.path.join(npy_dir, f"{key}.npy"), allow_pickle=True)
    if arr.dtype == object:  # list-of-arrays fallback
        arr = np.stack(arr)
    return arr


def _nearest_index(src_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp, index of the nearest source timestamp."""
    idx = np.searchsorted(src_ts, query_ts)
    idx = np.clip(idx, 1, len(src_ts) - 1)
    left, right = src_ts[idx - 1], src_ts[idx]
    idx -= (np.abs(query_ts - left) <= np.abs(query_ts - right)).astype(int)
    return np.clip(idx, 0, len(src_ts) - 1)


def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        return x
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        return x
    return filtfilt(b, a, x, axis=0)


def _solve_wrenches(model, data, side, qpos, tau_arm, qdd_arm, frame_id, damping, sign):
    """Per-timestep damped least-squares wrench, for both versions, in one pass."""
    cols = ARM_QSLICE[side]
    n = tau_arm.shape[0]
    W = {v: np.zeros((n, 6)) for v in VERSIONS}
    cond = np.zeros(n)
    eye6 = np.eye(6)
    qdd18 = np.zeros(model.nv)
    for t in range(n):
        J = pin.computeFrameJacobian(model, data, qpos[t], frame_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Ja = J[:, cols]                                    # 6 x 7
        op = Ja @ Ja.T + damping ** 2 * eye6               # damped (J J^T + lam^2 I)

        # version 1: raw tau_interaction
        W["raw"][t] = sign * np.linalg.solve(op, Ja @ tau_arm[t])

        # version 2: subtract rigid-body inertia term M(q) q_ddot
        M = pin.crba(model, data, qpos[t])
        M = M + M.T - np.diag(np.diag(M))                  # crba fills upper triangle
        qdd18[:] = 0.0
        qdd18[cols] = qdd_arm[t]
        tau_inertia = (M @ qdd18)[cols]
        W["inertiacomp"][t] = sign * np.linalg.solve(op, Ja @ (tau_arm[t] - tau_inertia))

        cond[t] = np.linalg.cond(Ja @ Ja.T)
    return W, cond


def process_episode(episode_dir: str, frame: str, convention: str,
                    cutoff_hz: float, accel_cutoff_hz: float, filter_order: int,
                    damping: float, save: bool, plot: bool):
    npy_dir = os.path.join(episode_dir, "low_dim_npys")
    if not os.path.exists(os.path.join(npy_dir, f"{TAU_KEY['L']}.npy")):
        print(f"  [skip] {episode_dir}: no tau_interaction recorded")
        return

    model = pin.buildModelFromMJCF(OPENARM_MODEL_PATH)
    data = model.createData()
    frame_name = (TCP_FRAME if frame == "hand_tcp" else WRIST_FRAME)
    frame_id = {s: model.getFrameId(frame_name[s]) for s in ("L", "R")}
    sign = -1.0 if convention == "env_on_robot" else 1.0

    tau_ts = _load(npy_dir, "follower_controller_state:timestamp").astype(float).ravel()
    fs = 1.0 / np.median(np.diff(tau_ts))

    results = {}
    for side in ("L", "R"):
        tau = _load(npy_dir, TAU_KEY[side]).astype(float)               # (N, 8)
        tau_arm = tau[:, :7]
        ang = _load(npy_dir, f"{STATE_KEY[side]}:actual_angle_rad").astype(float)      # (M, 8)
        vel = _load(npy_dir, f"{STATE_KEY[side]}:actual_velocity_radps").astype(float)  # (M, 8)
        ang_ts = _load(npy_dir, f"{STATE_KEY[side]}:timestamp").astype(float)
        ang_ts = ang_ts.mean(axis=1) if ang_ts.ndim > 1 else ang_ts.ravel()

        # align follower joint angles/velocities to the tau_interaction sample times
        sel = _nearest_index(ang_ts, tau_ts)
        q_arm7 = ang[sel, :7]
        qpos = np.zeros((len(tau_ts), model.nq))
        qpos[:, ARM_QSLICE[side]] = q_arm7

        # joint acceleration: differentiate measured velocity, then low-pass.
        # the recorded timestamps are noisy / non-monotonic, so differentiate on a
        # uniform grid (dt = 1/fs) rather than against the raw timestamps.
        qd_arm7 = vel[sel, :7]
        qdd_arm7 = np.gradient(qd_arm7, axis=0) * fs
        qdd_arm7 = _lowpass(qdd_arm7, fs, accel_cutoff_hz, filter_order)

        W, cond = _solve_wrenches(model, data, side, qpos, tau_arm, qdd_arm7,
                                  frame_id[side], damping, sign)
        W = {v: _lowpass(W[v], fs, cutoff_hz, filter_order) for v in VERSIONS}
        results[side] = {"W": W, "cond": cond}

        if save:
            for v in VERSIONS:
                suffix = "" if v == "raw" else "_inertiacomp"
                out_key = f"follower_controller_state:wrench_{frame}{suffix}_{side}"
                np.save(os.path.join(npy_dir, f"{out_key}.npy"), W[v])

        line = f"  {side}: cond(JJt) p95={np.percentile(cond, 95):10.1f}"
        for v in VERSIONS:
            F = np.linalg.norm(W[v][:, :3], axis=1)
            Mn = np.linalg.norm(W[v][:, 3:], axis=1)
            line += f"  | {v:11s} |F|mean={F.mean():5.2f} peak={F.max():6.2f}N  |M|peak={Mn.max():5.2f}Nm"
        print(line)

    print(f"  fs~{fs:.0f}Hz  frame={frame}  convention={convention}  "
          f"wrench_cutoff={cutoff_hz}Hz  accel_cutoff={accel_cutoff_hz}Hz"
          + ("  [saved]" if save else ""))

    if plot:
        _plot(episode_dir, tau_ts, results, frame)


def _plot(episode_dir, tau_ts, results, frame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot skipped] matplotlib unavailable: {e}")
        return
    t = tau_ts - tau_ts[0]
    for side, res in results.items():
        W = res["W"]
        fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for v in VERSIONS:
            axs[0].plot(t, np.linalg.norm(W[v][:, :3], axis=1), label=f"|F| {v}")
            axs[1].plot(t, np.linalg.norm(W[v][:, 3:], axis=1), label=f"|M| {v}")
        axs[0].set_ylabel("|force| [N]"); axs[0].legend(loc="upper right")
        axs[1].set_ylabel("|torque| [Nm]"); axs[1].set_xlabel("time [s]")
        axs[1].legend(loc="upper right")
        axs[0].set_title(f"{os.path.basename(episode_dir)}  arm {side}  ({frame}, world-aligned)")
        out = os.path.join(episode_dir, f"wrench_{frame}_{side}.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print(f"  [plot] {out}")


def main(
    data_dir: str = "../data/collecting",
    episode: str = "",          # specific episode folder name; empty = all episodes
    frame: str = "hand_tcp",    # hand_tcp | wrist
    convention: str = "env_on_robot",   # env_on_robot | robot_on_env
    cutoff_hz: float = 10.0,    # low-pass cutoff on the output wrench; <=0 disables
    accel_cutoff_hz: float = 8.0,   # low-pass cutoff on differentiated q_ddot
    filter_order: int = 4,
    damping: float = 1e-2,      # damped least-squares regularization (near singularities)
    save: bool = True,
    plot: bool = False,
):
    assert frame in ("hand_tcp", "wrist")
    assert convention in ("env_on_robot", "robot_on_env")

    if episode:
        episodes = [episode]
    else:
        episodes = sorted(d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d)))
    for ep in episodes:
        ep_dir = os.path.join(data_dir, ep)
        print(f"[{ep}]")
        process_episode(ep_dir, frame, convention, cutoff_hz, accel_cutoff_hz,
                        filter_order, damping, save, plot)


if __name__ == "__main__":
    typer.run(main)
