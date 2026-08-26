import time

import can
import numpy as np

import typer
from nova.redis_client import RedisClient
from nova.tools.precise_sleep import FrequencyRegulator

from nova.robot.openarm1.solver import OpenArm1Solver


def tau_friction_compensation(qvel, coeff, max_comp):
    tau_friction = np.clip(coeff * qvel, -max_comp, max_comp)
    return tau_friction

def tau_stiction_compensation(qvel, sti_threshold_min, sti_threshold_max, sti_comp):
    tau_stiction = np.where((sti_threshold_min < np.abs(qvel)) & (np.abs(qvel) < sti_threshold_max), np.sign(qvel) * sti_comp, 0)
    return tau_stiction

FRICTION_COEFF    = np.array([  0.5,   0.5,  2.0,  2.0, 0.05, 0.05, 0.05, 1.00])
FRICTION_MAX_COMP = np.array([ 0.09,  0.09, 0.8, 1.2,  0.1,  0.1,  0.1, 0.18])
STICTION_THRESHOLD_MIN = np.deg2rad([1, 1, 1, 1, 1, 1, 1, 1]) * 1
STICTION_THRESHOLD_MAX = np.deg2rad([1, 1, 1, 1, 1, 1, 1, 1]) * 10
STICTION_COMP = np.array([0.2, 0.2, 0.5, 0.5, 0.0, 0.0, 0.0, 0.2]) # 0.6, 0.6, 0.6, 0.6


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    follower_left: str = "follower_left",
    follower_right: str = "follower_right",
    leader_left: str = "leader_left",
    leader_right: str = "leader_right",
):
    
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=2000)
    solver = OpenArm1Solver()
    q_zeros = np.zeros(8)
    
    iter_idx = 0
    while True:
        data_batch = client.stream_get_batch(
            {
                f"{leader_left}_state:actual_angle_rad": np.ndarray,
                f"{leader_left}_state:actual_velocity_radps": np.ndarray,
                f"{leader_left}_state:actual_torque_nm": np.ndarray,

                f"{leader_right}_state:actual_angle_rad": np.ndarray,
                f"{leader_right}_state:actual_velocity_radps": np.ndarray,
                f"{leader_right}_state:actual_torque_nm": np.ndarray,
            }
        )
        
        qpos_L = data_batch[f"{leader_left}_state:actual_angle_rad"]
        qpos_R = data_batch[f"{leader_right}_state:actual_angle_rad"]
        sim_qpos_L = np.concatenate([qpos_L[:7], [0, 0]])
        sim_qpos_R = np.concatenate([qpos_R[:7], [0, 0]])
        sim_qpos = np.concatenate([sim_qpos_L, sim_qpos_R])

        qvel_L = data_batch[f"{leader_left}_state:actual_velocity_radps"]
        qvel_R = data_batch[f"{leader_right}_state:actual_velocity_radps"]
        sim_qvel_L = np.concatenate([qvel_L[:7], [0, 0]])
        sim_qvel_R = np.concatenate([qvel_R[:7], [0, 0]])
        sim_qvel = np.concatenate([sim_qvel_L, sim_qvel_R])
        
        solver.forward_kinematics(sim_qpos)
        
        desired_qacc = np.zeros_like(sim_qpos)
        tau_gravity_comp = solver.get_gravity_compensation_torques(sim_qpos, sim_qvel, desired_qacc)
        
        tau_friction_comp_L = tau_friction_compensation(qvel_L, FRICTION_COEFF, FRICTION_MAX_COMP)
        tau_friction_comp_R = tau_friction_compensation(qvel_R, FRICTION_COEFF, FRICTION_MAX_COMP)
        tau_stiction_comp_L = tau_stiction_compensation(qvel_L, STICTION_THRESHOLD, STICTION_COMP)
        tau_stiction_comp_R = tau_stiction_compensation(qvel_R, STICTION_THRESHOLD, STICTION_COMP)
        # print(f"tau_friction_comp_L: {tau_friction_comp_L[:2]}")
        print(tau_stiction_comp_L[-1])

        desired_tau_L = np.concatenate([tau_gravity_comp[0:7], [0.0]])  + tau_friction_comp_L + tau_stiction_comp_L
        desired_tau_R = np.concatenate([tau_gravity_comp[8:15], [0.0]]) + tau_friction_comp_R + tau_stiction_comp_R

        command_batch = {
            f"{leader_left}_mit_command:kp" : q_zeros,
            f"{leader_left}_mit_command:kd" : q_zeros,
            f"{leader_left}_mit_command:q"  : q_zeros,
            f"{leader_left}_mit_command:dq" : q_zeros,
            f"{leader_left}_mit_command:tau": desired_tau_L,

            f"{leader_right}_mit_command:kp" : q_zeros,
            f"{leader_right}_mit_command:kd" : q_zeros,
            f"{leader_right}_mit_command:q"  : q_zeros,
            f"{leader_right}_mit_command:dq" : q_zeros,
            f"{leader_right}_mit_command:tau": desired_tau_R,
        }
        command_batch_maxlen = {key: 1 for key in command_batch.keys()}  # Only keep the latest command
        client.stream_add_batch(command_batch, command_batch_maxlen)

        reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=f"teleop solver")
        iter_idx += 1
        
        if iter_idx == 1:
            print("teleop solver ready")


if __name__ == "__main__":
    typer.run(main)