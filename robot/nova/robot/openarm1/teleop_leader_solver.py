import time

import can
import numpy as np

import typer
from nova.redis_client import RedisClient
from nova.tools.precise_sleep import FrequencyRegulator

from nova.robot.openarm1.solver import OpenArm1Solver
from nova.robot.openarm1.teleop_leader_tuning import (
    tau_friction_compensation, FRICTION_COEFF, FRICTION_MAX_COMP, 
    tau_stiction_compensation, STICTION_THRESHOLD_MIN, STICTION_THRESHOLD_MAX, STICTION_COMP
)

torque_feedback_tolerance = np.deg2rad(1)
torque_feedback_coeff = 0.9
tanh_sharpness = 10.0

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
                f"{follower_left}_state:actual_angle_rad": np.ndarray,
                f"{follower_left}_state:actual_velocity_radps": np.ndarray,
                f"{follower_left}_state:actual_torque_nm": np.ndarray,
                f"{follower_left}_mit_command:tau": np.ndarray,

                f"{follower_right}_state:actual_angle_rad": np.ndarray,
                f"{follower_right}_state:actual_velocity_radps": np.ndarray,
                f"{follower_right}_state:actual_torque_nm": np.ndarray,
                f"{follower_right}_mit_command:tau": np.ndarray,

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
        tau = solver.get_gravity_compensation_torques(sim_qpos, sim_qvel, desired_qacc)

        follower_tau_interaction_L = torque_feedback_coeff * (data_batch[f"{follower_left}_state:actual_torque_nm"] - data_batch[f"{follower_left}_mit_command:tau"])
        position_error_L = data_batch[f"{leader_left}_state:actual_angle_rad"] - data_batch[f"{follower_left}_state:actual_angle_rad"]
        # follower_tau_interaction_L *= (np.abs(position_error_L) > torque_feedback_tolerance)
        scale = 0.5 * (np.tanh(tanh_sharpness * (np.abs(position_error_L) - torque_feedback_tolerance)) + 1)
        follower_tau_interaction_L *= scale

        follower_tau_interaction_R = torque_feedback_coeff * (data_batch[f"{follower_right}_state:actual_torque_nm"] - data_batch[f"{follower_right}_mit_command:tau"])
        position_error_R = data_batch[f"{leader_right}_state:actual_angle_rad"] - data_batch[f"{follower_right}_state:actual_angle_rad"]
        # follower_tau_interaction_R *= (np.abs(position_error_R) > torque_feedback_tolerance)
        scale = 0.5 * (np.tanh(tanh_sharpness * (np.abs(position_error_R) - torque_feedback_tolerance)) + 1)
        follower_tau_interaction_R *= scale

        # to_print = np.rad2deg(qvel_L[2:4]).tolist()
        # for x in to_print:
        #     print(f"{x:8.4f}", end=" ")
        # print()

        tau_friction_comp_L = tau_friction_compensation(qvel_L, FRICTION_COEFF, FRICTION_MAX_COMP)
        tau_friction_comp_R = tau_friction_compensation(qvel_R, FRICTION_COEFF, FRICTION_MAX_COMP)
        print(f"tau_fric_comp_L: {tau_friction_comp_L[3]:8.4f}")
        
        tau_stiction_comp_L = tau_stiction_compensation(qvel_L, STICTION_THRESHOLD_MIN, STICTION_THRESHOLD_MAX, STICTION_COMP)
        tau_stiction_comp_R = tau_stiction_compensation(qvel_R, STICTION_THRESHOLD_MIN, STICTION_THRESHOLD_MAX, STICTION_COMP)

        desired_tau_L = np.concatenate([tau[0:7], [0.0]])  - follower_tau_interaction_L + tau_friction_comp_L + tau_stiction_comp_L
        desired_tau_R = np.concatenate([tau[8:15], [0.0]]) - follower_tau_interaction_R + tau_friction_comp_R + tau_stiction_comp_R
        # observed_tau_L = data_batch[f"{leader_left}_state:actual_torque_nm"] - desired_tau_L
        # observed_tau_R = data_batch[f"{leader_right}_state:actual_torque_nm"] - desired_tau_R

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