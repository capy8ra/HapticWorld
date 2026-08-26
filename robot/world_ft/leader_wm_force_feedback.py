"""Milestone 2: reflect the world model's predicted interaction torque to the RIGHT leader.

Companion to ``scripts/inference/wm_follower_worker.py --publish-torque`` (worldsim).
The worker predicts the (virtual) follower's TOTAL joint torque and publishes it raw to
``virtual_follower_right_state:torque_pred_nm`` (+ ``:timestamp``). Here, on the nova side
(which has pinocchio / OpenArm1Solver), we isolate the interaction torque and reflect it to
the human's leader as force feedback:

    tau_interaction = torque_pred[:7] - gravity_comp(leader_q)          # isolate contact torque
    reflected       = ramp * k * LPF(tau_interaction), rate-limited, CLAMPED to +/- 8 N.m
    tau_cmd         = gravity_comp(leader_q) - reflected                # sign per teleop_leader_solver
    command leader with kp=kd=0 (backdrivable), gripper limp

Safety (the world model is open-loop and can drift, so bad torque is possible):
  * per-joint reflected-torque CLAMP at +/- ``--clamp-nm`` (default 8, the documented cap);
  * staleness WATCHDOG: if the WM torque is older than ``--stale-ms``, the reflected term is
    driven to 0 (slewed by the rate limit) so a hung/stale worker cannot keep pushing;
  * gain RAMP from 0 over ``--ramp-s`` once the worker is detected;
  * LPF + rate limit so the 10 Hz WM torque does not step-kick the ~1 kHz leader loop.

Phases: home to action[0] -> HOLD at action[0] until the worker starts publishing torque
        (auto-detected, no keypress) -> float + reflect (ramp in) -> on Ctrl-C, relax to
        gravity-comp float. Holding at the seed pose keeps the sim consistent; you can move
        the arm as soon as force engages (when the sim starts).

Env: hapticworld (nova + pinocchio + h5py). Needs redis (6379) + the right-leader damiao
controller on can1 (process-compose -f world_ft/teleop_wm_leader_only.yaml).
Use the SAME episode here and as the worker's --seed-episode. Keep a hand on the e-stop.
"""
import time

import numpy as np
import typer

# same directory when run as `python world_ft/leader_wm_force_feedback.py`
from leader_home_and_float import (
    ARM_KP,
    ARM_KD,
    _command,
    _gravity_comp_R,
    _read_leader,
    _smoothstep,
)
from nova.redis_client import RedisClient
from nova.robot.openarm1.solver import OpenArm1Solver
from nova.tools.precise_sleep import FrequencyRegulator

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def _load_target_q7(seed_episode: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is required to read the seed episode start pose")
    with h5py.File(seed_episode, "r") as f:
        return f["action"][0][:7].astype(float)


def _home(client, solver, reg, leader_right, target_q7, homing_time_s, max_vel_deg_s, eps_deg, gain_ramp_s):
    """Smooth, speed-limited ramp from the current pose to target_q7 (gains ramp from 0)."""
    while True:
        s = _read_leader(client, leader_right)
        if s is not None:
            break
        reg.sleep()
    q0_arm = s[0][:7]
    max_disp = float(np.max(np.abs(target_q7 - q0_arm)))
    duration = max(homing_time_s, max_disp / max(np.deg2rad(max_vel_deg_s), 1e-6))
    print(f"[ff] homing {np.rad2deg(max_disp):.1f} deg over {duration:.1f}s ...")
    t_start = time.time()
    while True:
        s = _read_leader(client, leader_right)
        if s is None:
            reg.sleep()
            continue
        q, v = s
        t = time.time() - t_start
        s_frac, ds_frac = _smoothstep(t / duration)
        q_des = q0_arm + s_frac * (target_q7 - q0_arm)
        dq_des = (ds_frac / duration) * (target_q7 - q0_arm)
        ramp = min(1.0, t / max(gain_ramp_s, 1e-6))
        tau = _gravity_comp_R(solver, q[:7], v[:7])
        _command(client, leader_right, ramp * ARM_KP, ramp * ARM_KD, q_des, dq_des, tau, gripper_q=q[7])
        if t >= duration and float(np.max(np.abs(q[:7] - target_q7))) <= np.deg2rad(eps_deg):
            print("[ff] homed.")
            return


def main(
    seed_episode: str = typer.Option(..., help="hdf5 whose action[0][:7] is the start pose (SAME as the worker)"),
    redis_host: str = "localhost",
    redis_port: int = 6379,
    leader_right: str = "leader_right",
    virtual_follower: str = "virtual_follower_right",
    # reflection
    k: float = 0.3,             # target reflection gain (conservative first bring-up)
    ramp_s: float = 3.0,          # ramp the gain 0 -> k over this long once the worker is detected
    clamp_nm: float = 8.0,        # per-joint reflected-torque safety cap (the documented +/-8 N.m)
    lpf_tau_s: float = 0.05,      # 1-pole low-pass time constant on the reflected torque
    rate_nm_s: float = 40.0,      # max slew of the reflected torque (N.m/s)
    stale_ms: float = 300.0,      # WM torque older than this -> drive reflection to 0
    auto_start_timeout_s: float = 120.0,  # if no fresh WM torque by now, float anyway (0 = wait forever)
    # loop / homing
    freq_hz: float = 1000.0,
    homing_time_s: float = 3.0,
    max_vel_deg_s: float = 30.0,
    eps_deg: float = 1.5,
    gain_ramp_s: float = 0.5,
):
    client = RedisClient(host=redis_host, port=redis_port)
    solver = OpenArm1Solver()
    reg = FrequencyRegulator(frequency=freq_hz)

    target_q7 = _load_target_q7(seed_episode)
    torque_key = f"{virtual_follower}_state:torque_pred_nm"
    torque_ts_key = f"{virtual_follower}_state:timestamp"
    print(f"[ff] seed={seed_episode}  target[0](deg)={np.rad2deg(target_q7).round(1).tolist()}")
    print(f"[ff] k={k} clamp=+/-{clamp_nm} N.m  lpf_tau={lpf_tau_s}s  rate={rate_nm_s} N.m/s  stale={stale_ms:.0f}ms")

    # ---- Phase 1: home to action[0] ----
    _home(client, solver, reg, leader_right, target_q7, homing_time_s, max_vel_deg_s, eps_deg, gain_ramp_s)

    stale_s = stale_ms / 1000.0

    # ---- Phase 2: HOLD at action[0] until the worker (simulation) is up ----
    # Keep holding exactly at the seed pose (position gains, do NOT float) so the worker seeds
    # consistently. No keypress: auto-detect the worker by its first FRESH published torque,
    # then switch to float + reflect so you can move the arm as soon as the sim is running.
    print("[ff] homed & HOLDING at start pose. Start the worker (--publish-torque); force feedback "
          "engages automatically when it starts publishing (do not move the arm until then).")
    t_wait = time.time()
    while True:
        s = _read_leader(client, leader_right)
        if s is None:
            reg.sleep()
            continue
        q, v = s
        grav = _gravity_comp_R(solver, q[:7], v[:7])
        _command(client, leader_right, ARM_KP, ARM_KD, target_q7, np.zeros(7), grav, gripper_q=q[7])
        ts = client.stream_get_batch({torque_ts_key: float})[torque_ts_key]
        if ts is not None and (time.time() - ts) <= stale_s:
            print("[ff] worker detected (fresh WM torque) -- switching to float + reflect.")
            break
        if auto_start_timeout_s > 0 and (time.time() - t_wait) > auto_start_timeout_s:
            print(f"[ff] WARNING: no fresh WM torque after {auto_start_timeout_s:.0f}s -- floating anyway.")
            break
        reg.sleep()

    lpf = np.zeros(7)          # low-pass state of the interaction torque
    reflected = np.zeros(7)    # rate-limited reflected torque actually applied
    dt = 1.0 / freq_hz
    t_enter = time.time()      # gain ramp starts now (worker detected)
    last_print = 0.0
    try:
        print("[ff] reflecting -- ramping gain in. Ctrl-C to release.")
        while True:
            s = _read_leader(client, leader_right)
            if s is None:
                reg.sleep()
                continue
            q, v = s
            grav = _gravity_comp_R(solver, q[:7], v[:7])

            batch = client.stream_get_batch({torque_key: np.ndarray, torque_ts_key: float})
            tau_pred = batch[torque_key]
            ts = batch[torque_ts_key]
            fresh = tau_pred is not None and ts is not None and (time.time() - ts) <= stale_s

            if fresh:
                tau_int = tau_pred[:7].astype(float) - grav            # isolate interaction torque
                lpf += (dt / (lpf_tau_s + dt)) * (tau_int - lpf)       # 1-pole LPF
                k_now = k * min(1.0, (time.time() - t_enter) / max(ramp_s, 1e-6))
                reflected_des = np.clip(k_now * lpf, -clamp_nm, clamp_nm)
                tau_int_max = float(np.max(np.abs(lpf)))               # for the observe-stage print
            else:
                reflected_des = np.zeros(7)                            # watchdog: slew to zero
                tau_int_max = float("nan")

            step_max = rate_nm_s * dt
            reflected += np.clip(reflected_des - reflected, -step_max, step_max)
            reflected = np.clip(reflected, -clamp_nm, clamp_nm)        # final safety cap

            tau_cmd = grav - reflected                                # sign per teleop_leader_solver
            zeros7 = np.zeros(7)
            _command(client, leader_right, zeros7, zeros7, zeros7, zeros7, tau_cmd, gripper_q=q[7])

            now = time.time()
            if now - last_print >= 1.0:
                tag = "" if fresh else " [STALE->0]"
                print(f"[ff] |tau_int|max={tau_int_max:5.2f}  |reflected|max={np.max(np.abs(reflected)):5.2f} N.m  "
                      f"k_now={k * min(1.0,(now-t_enter)/max(ramp_s,1e-6)):.2f}{tag}")
                last_print = now
            reg.sleep()
    finally:
        # safe release: gravity-comp float (limp), no reflected torque
        s = _read_leader(client, leader_right)
        if s is not None:
            q, v = s
            grav = _gravity_comp_R(solver, q[:7], v[:7])
            z7 = np.zeros(7)
            _command(client, leader_right, z7, z7, z7, z7, grav, gripper_q=q[7])
        print("[ff] released to gravity-comp float.")


if __name__ == "__main__":
    typer.run(main)
