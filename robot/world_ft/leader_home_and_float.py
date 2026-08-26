"""Home the OpenArm RIGHT leader to a fixed start pose, then hold it in
gravity-compensation float.

This is the leader-side companion to the world-model teleop (milestone 1: visual
only, no force feedback yet).

Phase 1 -- homing:
    From wherever the human roughly placed the leader, smoothly drive the 7
    right-arm joints (ease-in/out, speed-limited, gains ramped up from 0) to the
    seed pose = ``action[0][:7]`` of a chosen validation episode. The gripper
    (joint 8) is NOT driven.

Phase 2 -- float:
    Command MIT with kp=kd=0 and tau = gravity compensation, so the arm floats
    and the human can move it freely while watching the world simulator. (This is
    where reflected torque will later be added for milestone 2.)

Runs in the hapticworld env (needs nova + pinocchio + CAN). Talks to the same redis
(6379) as the leader damiao controller. The world-model worker
(world_ft/scripts/inference/wm_follower_worker.py) must be seeded from the SAME
episode so the sim scene and the leader pose start consistent.
"""
import time

import numpy as np
import typer

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

from nova.redis_client import RedisClient
from nova.robot.openarm1.solver import OpenArm1Solver
from nova.tools.precise_sleep import FrequencyRegulator

# Arm gains (7 right-arm joints). Mirror the follower teleop gains; gripper -> 0.
ARM_KP = np.array([240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0])
ARM_KD = np.array([3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2])
TAU_CLAMP_NM = 20.0  # safety clamp on commanded feed-forward torque (per joint)

# sim_qpos layout used by OpenArm1Solver: [L_arm(0:7), L_grip(7:9), R_arm(9:16), R_grip(16:18)]
R_ARM_SLICE = slice(9, 16)
# get_gravity_compensation_torques returns 16 = [L_arm7, L_grip1, R_arm7, R_grip1];
# the right arm's 7 joints are positions 8:15 in that vector.
GRAV_R_SLICE = slice(8, 15)


def _smoothstep(u: float) -> tuple[float, float]:
    """Ease-in/out position fraction s(u) and its d/du, for u in [0,1]."""
    u = min(max(u, 0.0), 1.0)
    s = 3 * u**2 - 2 * u**3
    ds = 6 * u - 6 * u**2
    return s, ds


def _read_leader(client: RedisClient, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    b = client.stream_get_batch(
        {
            f"{name}_state:actual_angle_rad": np.ndarray,
            f"{name}_state:actual_velocity_radps": np.ndarray,
        }
    )
    ang = b[f"{name}_state:actual_angle_rad"]
    vel = b[f"{name}_state:actual_velocity_radps"]
    if ang is None or vel is None:
        return None
    return ang.astype(float), vel.astype(float)


def _gravity_comp_R(solver: OpenArm1Solver, arm_q7: np.ndarray, arm_v7: np.ndarray) -> np.ndarray:
    """Right-arm gravity(+Coriolis) compensation torque, 7-vector, in N*m."""
    nq = solver.model.nq
    q = np.zeros(nq)
    v = np.zeros(nq)
    q[R_ARM_SLICE] = arm_q7
    v[R_ARM_SLICE] = arm_v7
    solver.forward_kinematics(q)
    grav16 = solver.get_gravity_compensation_torques(q, v, np.zeros(nq))
    return np.asarray(grav16[GRAV_R_SLICE], dtype=float)


def _command(client: RedisClient, name: str, kp7, kd7, q7, dq7, tau7, gripper_q: float) -> None:
    """Send an 8-DoF MIT command (7 arm joints + a passive gripper column)."""
    tau7 = np.clip(tau7, -TAU_CLAMP_NM, TAU_CLAMP_NM)
    batch = {
        f"{name}_mit_command:kp": np.concatenate([kp7, [0.0]]).astype(np.float64),
        f"{name}_mit_command:kd": np.concatenate([kd7, [0.0]]).astype(np.float64),
        f"{name}_mit_command:q": np.concatenate([q7, [gripper_q]]).astype(np.float64),
        f"{name}_mit_command:dq": np.concatenate([dq7, [0.0]]).astype(np.float64),
        f"{name}_mit_command:tau": np.concatenate([tau7, [0.0]]).astype(np.float64),
    }
    client.stream_add_batch(batch, {k: 1 for k in batch})


def _load_target_q7(seed_episode: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is required to read the seed episode target pose")
    with h5py.File(seed_episode, "r") as f:
        action0 = f["action"][0].astype(float)  # (8,) right q_des at frame 0
    return action0[:7]


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    leader_right: str = "leader_right",
    seed_episode: str = typer.Option(..., help="hdf5 whose action[0][:7] is the start pose"),
    homing_time_s: float = 3.0,
    max_vel_deg_s: float = 30.0,
    eps_deg: float = 1.5,
    gain_ramp_s: float = 0.5,
    freq_hz: float = 500.0,
):
    client = RedisClient(host=redis_host, port=redis_port)
    solver = OpenArm1Solver()
    reg = FrequencyRegulator(frequency=freq_hz)

    target_q7 = _load_target_q7(seed_episode)
    print(f"[home] seed={seed_episode}")
    print(f"[home] target(deg)={np.rad2deg(target_q7).round(1).tolist()}")

    # wait for a first valid leader reading
    while True:
        s = _read_leader(client, leader_right)
        if s is not None:
            break
        reg.sleep()
    q0, _ = s
    q0_arm = q0[:7]

    def home_from(q0_arm: np.ndarray) -> None:
        """Phase 1: smooth speed-limited move to target_q7 with ramped gains."""
        max_disp = float(np.max(np.abs(target_q7 - q0_arm)))
        max_vel = np.deg2rad(max_vel_deg_s)
        duration = max(homing_time_s, max_disp / max(max_vel, 1e-6))
        print(f"[home] moving {np.rad2deg(max_disp):.1f} deg over {duration:.1f}s ...")
        t_start = time.time()
        while True:
            s = _read_leader(client, leader_right)
            if s is None:
                reg.sleep()
                continue
            q, v = s
            q_arm, v_arm = q[:7], v[:7]

            t = time.time() - t_start
            s_frac, ds_frac = _smoothstep(t / duration)
            q_des = q0_arm + s_frac * (target_q7 - q0_arm)
            dq_des = (ds_frac / duration) * (target_q7 - q0_arm)

            ramp = min(1.0, t / max(gain_ramp_s, 1e-6))
            kp = ramp * ARM_KP
            kd = ramp * ARM_KD
            tau = _gravity_comp_R(solver, q_arm, v_arm)
            _command(client, leader_right, kp, kd, q_des, dq_des, tau, gripper_q=q[7])

            err = float(np.max(np.abs(q_arm - target_q7)))
            if t >= duration and err <= np.deg2rad(eps_deg):
                print(f"[home] leader homed (max err {np.rad2deg(err):.2f} deg)")
                return
            reg.sleep()

    home_from(q0_arm)

    # ---- Phase 2: gravity-comp float; re-home on wm_leader:command (published by
    # the WM worker's 's' = episode saved), so the arm returns to the seed pose
    # automatically between episodes ----
    print("leader float ready (auto re-home on the worker's 's')")
    last_cmd_ts = None
    tick = 0
    while True:
        s = _read_leader(client, leader_right)
        if s is None:
            reg.sleep()
            continue
        q, v = s
        tau = _gravity_comp_R(solver, q[:7], v[:7])
        zeros7 = np.zeros(7)
        _command(client, leader_right, zeros7, zeros7, zeros7, zeros7, tau, gripper_q=q[7])

        tick += 1
        if tick % 25 == 0:  # ~20 Hz command poll, cheap on the 500 Hz loop
            b = client.stream_get_batch({"wm_leader:command": str,
                                         "wm_leader:timestamp": float})
            cmd, cts = b["wm_leader:command"], b["wm_leader:timestamp"]
            if cts is not None and cts != last_cmd_ts:
                last_cmd_ts = cts
                if cmd == "rehome":
                    print("[home] re-home command from the WM worker")
                    home_from(q[:7])
                    print("[home] floating again")
        reg.sleep()


if __name__ == "__main__":
    typer.run(main)
