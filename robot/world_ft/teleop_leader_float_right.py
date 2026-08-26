"""Right-only leader gravity-comp float (kp=kd=0, tau = gravity comp, gripper limp).

The leader is backdrivable and STAYS wherever you put it -- no homing phase. Runs as part
of capture_wm_seeds.yaml so posing the follower (which mirrors the leader) needs no extra
terminal; the same job collect_data.yaml's leader_solver does, minus the bimanual streams
and force feedback (right arm only, nothing to reflect during seed capture).
"""
import time

import numpy as np
import typer

from nova.redis_client import RedisClient
from nova.tools.precise_sleep import FrequencyRegulator
from nova.robot.openarm1.solver import OpenArm1Solver


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    leader_right: str = "leader_right",
    frequency_hz: float = 1000,
):
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=frequency_hz)
    solver = OpenArm1Solver()
    q_zeros = np.zeros(8)

    iter_idx = 0
    while True:
        data_batch = client.stream_get_batch({
            f"{leader_right}_state:actual_angle_rad": np.ndarray,
            f"{leader_right}_state:actual_velocity_radps": np.ndarray,
        })
        qpos_R = data_batch[f"{leader_right}_state:actual_angle_rad"]
        qvel_R = data_batch[f"{leader_right}_state:actual_velocity_radps"]
        if any(v is None for v in (qpos_R, qvel_R)):
            reg.sleep()
            continue

        # gravity/Coriolis for the right arm (bimanual model, left held at 0 -- independent chains)
        sim_qpos = np.concatenate([np.zeros(9), np.concatenate([qpos_R[:7], [0, 0]])])
        sim_qvel = np.concatenate([np.zeros(9), np.concatenate([qvel_R[:7], [0, 0]])])
        solver.forward_kinematics(sim_qpos)
        tau = solver.get_gravity_compensation_torques(sim_qpos, sim_qvel, np.zeros_like(sim_qpos))
        tau_R = np.concatenate([tau[8:15], [0.0]])

        command_batch = {
            f"{leader_right}_mit_command:timestamp": time.time(),
            f"{leader_right}_mit_command:kp": q_zeros,
            f"{leader_right}_mit_command:kd": q_zeros,
            f"{leader_right}_mit_command:q": qpos_R,
            f"{leader_right}_mit_command:dq": q_zeros,
            f"{leader_right}_mit_command:tau": tau_R,
        }
        command_batch_maxlen = {k: int(frequency_hz * 5) for k in command_batch}
        client.stream_add_batch(command_batch, command_batch_maxlen)

        reg.sleep(verbose=True, verbose_interval=1, verbose_prefix="leader float (right)")
        iter_idx += 1
        if iter_idx == 1:
            print("leader float ready")


if __name__ == "__main__":
    typer.run(main)
