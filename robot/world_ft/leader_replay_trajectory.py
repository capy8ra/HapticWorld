"""Drive the RIGHT leader along a RECORDED episode trajectory (active position control).

Hardware-in-the-loop counterpart to the direct redis replay
(world_ft: scripts/inference/replay_leader_to_redis.py). Instead of writing the recorded
action straight onto ``leader_right_state:actual_angle_rad``, this commands the physical
leader to FOLLOW the recorded trajectory; the damiao controller then publishes the
leader's MEASURED angle, which the world-model worker reads -- exercising the full real
path (recorded command -> real leader motors -> measured angle -> simulator).

Phases:
  1. home   : smooth, speed-limited ramp from the current pose to ``action[0][:7]``
              (reuses ``leader_home_and_float``'s homing; gains ramp from 0).
  2. hold   : hold ``action[0]`` and WAIT FOR ENTER. Start the worker (step 2) first,
              then press Enter here -- that is the "start moving" moment.
  3. replay : play ``action[1..T-1]`` at ``--hz``, linearly interpolated up to the control
              rate, with the teleop ``ARM_KP/KD`` + gravity-comp ``tau`` + velocity feedforward.
  4. hold   : hold the final pose until Ctrl-C (then relax to gravity-comp float).

Env: hapticworld (nova + pinocchio + h5py). Needs redis (6379) and the right-leader damiao
controller up on can1 -- start them with:
    process-compose -f world_ft/teleop_wm_leader_only.yaml
Use the SAME episode here and as the worker's --seed-episode.
"""
import os
import sys
import time

import numpy as np
import typer

# Same directory as this script when run as `python world_ft/leader_replay_trajectory.py`,
# so the homing gains/helpers are reused rather than duplicated.
from leader_home_and_float import (
    ARM_KP,
    ARM_KD,
    _command,
    _gravity_comp_R,
    _read_leader,
    _smoothstep,
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # robot/ (nova package root)
from nova.redis_client import RedisClient
from nova.robot.openarm1.solver import OpenArm1Solver
from nova.tools.precise_sleep import FrequencyRegulator

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def _load_traj(seed_episode: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is required to read the seed episode trajectory")
    with h5py.File(seed_episode, "r") as f:
        action = f["action"][()].astype(float)  # (T, 8) right q_des per frame
    return action


def main(
    seed_episode: str = typer.Option(..., help="hdf5 whose action[t] trajectory the leader replays"),
    redis_host: str = "localhost",
    redis_port: int = 6379,
    leader_right: str = "leader_right",
    hz: float = 10.0,
    freq_hz: float = 500.0,
    homing_time_s: float = 3.0,
    max_vel_deg_s: float = 30.0,
    eps_deg: float = 1.5,
    gain_ramp_s: float = 0.5,
    max_step_deg: float = 20.0,
):
    client = RedisClient(host=redis_host, port=redis_port)
    solver = OpenArm1Solver()
    reg = FrequencyRegulator(frequency=freq_hz)

    action = _load_traj(seed_episode)
    traj = action[:, :7]  # (T, 7) arm waypoints; gripper (col 7) left limp
    T = traj.shape[0]

    # Safety pre-scan: reject a glitchy recorded frame that would command a violent jump.
    step_deg = np.rad2deg(np.abs(np.diff(traj, axis=0))).max(axis=1)  # (T-1,)
    worst_i = int(step_deg.argmax())
    worst = float(step_deg[worst_i])
    print(f"[replay] {seed_episode}")
    print(f"[replay] T={T} (~{T/hz:.1f}s @ {hz:.0f}Hz)  target[0](deg)={np.rad2deg(traj[0]).round(1).tolist()}")
    print(f"[replay] worst per-frame joint step = {worst:.1f} deg at frame {worst_i} (~{worst*hz:.0f} deg/s)")
    if worst > max_step_deg:
        raise SystemExit(
            f"[replay] ABORT: frame {worst_i} jumps {worst:.1f} deg (> --max-step-deg {max_step_deg}). "
            "Recorded trajectory looks glitchy/too fast; inspect it or raise --max-step-deg."
        )

    # Wait for a first valid leader reading.
    while True:
        s = _read_leader(client, leader_right)
        if s is not None:
            break
        reg.sleep()
    q0_arm = s[0][:7]

    # ---- Phase 1: home to action[0] (smooth, speed-limited, gains ramped from 0) ----
    target0 = traj[0]
    max_disp = float(np.max(np.abs(target0 - q0_arm)))
    duration = max(homing_time_s, max_disp / max(np.deg2rad(max_vel_deg_s), 1e-6))
    print(f"[replay] homing {np.rad2deg(max_disp):.1f} deg over {duration:.1f}s ...")
    t_start = time.time()
    while True:
        s = _read_leader(client, leader_right)
        if s is None:
            reg.sleep()
            continue
        q, v = s
        t = time.time() - t_start
        s_frac, ds_frac = _smoothstep(t / duration)
        q_des = q0_arm + s_frac * (target0 - q0_arm)
        dq_des = (ds_frac / duration) * (target0 - q0_arm)
        ramp = min(1.0, t / max(gain_ramp_s, 1e-6))
        tau = _gravity_comp_R(solver, q[:7], v[:7])
        _command(client, leader_right, ramp * ARM_KP, ramp * ARM_KD, q_des, dq_des, tau, gripper_q=q[7])
        err = float(np.max(np.abs(q[:7] - target0)))
        if t >= duration and err <= np.deg2rad(eps_deg):
            print(f"[replay] homed (max err {np.rad2deg(err):.2f} deg)")
            break
        reg.sleep()

    # ---- Phase 2: hold action[0], wait for the worker ----
    while True:
        s = _read_leader(client, leader_right)
        if s is not None:
            break
        reg.sleep()
    q = s[0]
    tau = _gravity_comp_R(solver, q[:7], np.zeros(7))
    _command(client, leader_right, ARM_KP, ARM_KD, target0, np.zeros(7), tau, gripper_q=q[7])
    input("[replay] leader homed & holding at start. Start the worker (step 2), then press Enter to play ...")

    try:
        # ---- Phase 3: replay the recorded trajectory ----
        print("[replay] playing trajectory ...")
        t0 = time.time()
        last_print = -1.0
        while True:
            s = _read_leader(client, leader_right)
            if s is None:
                reg.sleep()
                continue
            q, v = s
            tau = _gravity_comp_R(solver, q[:7], v[:7])
            tf = (time.time() - t0) * hz
            i = int(tf)
            if i >= T - 1:
                break
            frac = tf - i
            q_des = traj[i] + frac * (traj[i + 1] - traj[i])  # interpolate between 10Hz waypoints
            dq_des = (traj[i + 1] - traj[i]) * hz             # waypoint velocity feedforward (rad/s)
            _command(client, leader_right, ARM_KP, ARM_KD, q_des, dq_des, tau, gripper_q=q[7])
            if time.time() - last_print >= 1.0:
                print(f"[replay] t={tf/hz:5.1f}s  frame {i:4d}/{T-1}")
                last_print = time.time()
            reg.sleep()

        # ---- Phase 4: hold the final pose ----
        print("[replay] trajectory done -- holding final pose (Ctrl-C to release) ...")
        q_final = traj[T - 1]
        while True:
            s = _read_leader(client, leader_right)
            if s is None:
                reg.sleep()
                continue
            q, v = s
            tau = _gravity_comp_R(solver, q[:7], v[:7])
            _command(client, leader_right, ARM_KP, ARM_KD, q_final, np.zeros(7), tau, gripper_q=q[7])
            reg.sleep()
    finally:
        # Safe shutdown: relax to gravity-comp float (limp) so the arm is not left actively driven.
        s = _read_leader(client, leader_right)
        if s is not None:
            q, v = s
            tau = _gravity_comp_R(solver, q[:7], v[:7])
            z7 = np.zeros(7)
            _command(client, leader_right, z7, z7, z7, z7, tau, gripper_q=q[7])
        print("[replay] released to gravity-comp float.")


if __name__ == "__main__":
    typer.run(main)
