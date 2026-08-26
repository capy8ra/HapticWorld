import time

import can
import numpy as np

import typer
from nova.redis_client import RedisClient
from nova.tools.precise_sleep import FrequencyRegulator

from nova.robot.openarm1.solver import OpenArm1Solver

FOLLOWER_KP = np.array([240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0])
FOLLOWER_KD = np.array([3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])

def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    follower_left: str = "follower_left",
    follower_right: str = "follower_right",
    leader_left: str = "leader_left",
    leader_right: str = "leader_right",
    frequency_hz: float = 2000,
):
    
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=frequency_hz)
    solver = OpenArm1Solver()
    iter_idx = 0
    while True:
        data_batch = client.stream_get_batch(
            {
                f"{follower_left}_state:actual_angle_rad": np.ndarray,
                f"{follower_left}_state:actual_velocity_radps": np.ndarray,
                f"{follower_left}_state:actual_torque_nm": np.ndarray,

                f"{follower_right}_state:actual_angle_rad": np.ndarray,
                f"{follower_right}_state:actual_velocity_radps": np.ndarray,
                f"{follower_right}_state:actual_torque_nm": np.ndarray,
                
                f"{leader_left}_state:actual_angle_rad": np.ndarray,
                f"{leader_left}_state:actual_velocity_radps": np.ndarray,
                f"{leader_left}_state:actual_torque_nm": np.ndarray,

                f"{leader_right}_state:actual_angle_rad": np.ndarray,
                f"{leader_right}_state:actual_velocity_radps": np.ndarray,
                f"{leader_right}_state:actual_torque_nm": np.ndarray,
            }
        )
        qpos_L = data_batch[f"{follower_left}_state:actual_angle_rad"]
        qpos_R = data_batch[f"{follower_right}_state:actual_angle_rad"]
        sim_qpos_L = np.concatenate([qpos_L[:7], [0, 0]])
        sim_qpos_R = np.concatenate([qpos_R[:7], [0, 0]])
        sim_qpos = np.concatenate([sim_qpos_L, sim_qpos_R])
        
        qvel_L = data_batch[f"{follower_left}_state:actual_velocity_radps"]
        qvel_R = data_batch[f"{follower_right}_state:actual_velocity_radps"]
        sim_qvel_L = np.concatenate([qvel_L[:7], [0, 0]])
        sim_qvel_R = np.concatenate([qvel_R[:7], [0, 0]])
        sim_qvel = np.concatenate([sim_qvel_L, sim_qvel_R])
        
        solver.forward_kinematics(sim_qpos)
        
        desired_qacc = np.zeros_like(sim_qpos)
        tau = solver.get_gravity_compensation_torques(sim_qpos, sim_qvel, desired_qacc)
        desired_tau_L = np.concatenate([tau[0:7], [0.0]])
        desired_tau_R = np.concatenate([tau[8:15], [0.0]])
        
        desired_timestamp = time.time()
        desired_qpos_L = data_batch[f"{leader_left}_state:actual_angle_rad"]
        desired_qpos_R = data_batch[f"{leader_right}_state:actual_angle_rad"]
        desired_qvel_L = data_batch[f"{leader_left}_state:actual_velocity_radps"]
        desired_qvel_R = data_batch[f"{leader_right}_state:actual_velocity_radps"]

        command_batch = {
            
            f"{follower_left}_mit_command:timestamp": desired_timestamp,
            f"{follower_left}_mit_command:kp" : FOLLOWER_KP,
            f"{follower_left}_mit_command:kd" : FOLLOWER_KD,
            f"{follower_left}_mit_command:q"  : desired_qpos_L,
            f"{follower_left}_mit_command:dq" : desired_qvel_L,
            f"{follower_left}_mit_command:tau": desired_tau_L,

            f"{follower_right}_mit_command:timestamp": desired_timestamp,
            f"{follower_right}_mit_command:kp" : FOLLOWER_KP,
            f"{follower_right}_mit_command:kd" : FOLLOWER_KD,
            f"{follower_right}_mit_command:q"  : desired_qpos_R,
            f"{follower_right}_mit_command:dq" : desired_qvel_R,
            f"{follower_right}_mit_command:tau": desired_tau_R,
        }
        command_batch_maxlen = {key: int(frequency_hz * 5) for key in command_batch.keys()}  # Litian HACK: 5 seconds
        client.stream_add_batch(command_batch, command_batch_maxlen)

        reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=f"teleop solver")
        iter_idx += 1
        
        if iter_idx == 1:
            print("teleop solver ready")


if __name__ == "__main__":
    typer.run(main)