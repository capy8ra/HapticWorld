"""Right-only teleop follower solver for the world_ft sim collection.

Makes the REAL right follower physically mirror the right leader (direct joint copy + gravity comp),
so moving the leader moves the follower -- exactly like nova/robot/openarm1/teleop_follower_solver.py
but for the right arm only (the sim rig has no left arm). Run it alongside the leader controller +
the real right-follower controller (see teleop_wm_leader_follower.yaml).
"""
import os
import sys
import time

import numpy as np
import typer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # robot/ (nova package root)
from nova.redis_client import RedisClient
from nova.tools.precise_sleep import FrequencyRegulator
from nova.robot.openarm1.solver import OpenArm1Solver

FOLLOWER_KP = np.array([240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0])
FOLLOWER_KD = np.array([3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    follower_right: str = "follower_right",
    leader_right: str = "leader_right",
    frequency_hz: float = 2000,
):
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=frequency_hz)
    solver = OpenArm1Solver()

    iter_idx = 0
    while True:
        data_batch = client.stream_get_batch({
            f"{follower_right}_state:actual_angle_rad": np.ndarray,
            f"{follower_right}_state:actual_velocity_radps": np.ndarray,
            f"{leader_right}_state:actual_angle_rad": np.ndarray,
            f"{leader_right}_state:actual_velocity_radps": np.ndarray,
        })
        qpos_R = data_batch[f"{follower_right}_state:actual_angle_rad"]
        qvel_R = data_batch[f"{follower_right}_state:actual_velocity_radps"]
        leader_qR = data_batch[f"{leader_right}_state:actual_angle_rad"]
        leader_vR = data_batch[f"{leader_right}_state:actual_velocity_radps"]
        if any(v is None for v in (qpos_R, qvel_R, leader_qR, leader_vR)):
            reg.sleep()
            continue

        # gravity/Coriolis compensation for the right arm (the solver models both arms; the left is
        # held at 0 -- independent chains, so the right-arm gravity torque is unaffected)
        sim_qpos = np.concatenate([np.zeros(9), np.concatenate([qpos_R[:7], [0, 0]])])
        sim_qvel = np.concatenate([np.zeros(9), np.concatenate([qvel_R[:7], [0, 0]])])
        solver.forward_kinematics(sim_qpos)
        tau = solver.get_gravity_compensation_torques(sim_qpos, sim_qvel, np.zeros_like(sim_qpos))
        desired_tau_R = np.concatenate([tau[8:15], [0.0]])

        # command the follower to track the leader's joint angles (+ velocity feedforward)
        command_batch = {
            f"{follower_right}_mit_command:timestamp": time.time(),
            f"{follower_right}_mit_command:kp": FOLLOWER_KP,
            f"{follower_right}_mit_command:kd": FOLLOWER_KD,
            f"{follower_right}_mit_command:q": leader_qR,
            f"{follower_right}_mit_command:dq": leader_vR,
            f"{follower_right}_mit_command:tau": desired_tau_R,
        }
        command_batch_maxlen = {k: int(frequency_hz * 5) for k in command_batch}
        client.stream_add_batch(command_batch, command_batch_maxlen)

        reg.sleep(verbose=True, verbose_interval=1, verbose_prefix="teleop follower solver (right)")
        iter_idx += 1
        if iter_idx == 1:
            print("teleop follower solver ready")


if __name__ == "__main__":
    typer.run(main)
