import time

import can
import numpy as np

import typer
from nova.redis_client import RedisClient
from nova.tools.precise_sleep import FrequencyRegulator

from nova.sim.mujoco_env import MujocoBaseEnv, mujoco, mink
from dm_control import mjcf
from nova.robot.openarm1.models import OPENARM_MODEL_PATH

class OpenArm1PlayEnv(MujocoBaseEnv):
    
    def construct_model(self):
        root = self.get_chessboard_floor()
        self.add_coordinate_frame_mocap(root, "world_origin", size=0.05)
        self.add_coordinate_frame_mocap(root, "L_ee", size=0.05)
        self.add_coordinate_frame_mocap(root, "R_ee", size=0.05)
        self.add_coordinate_frame_mocap(root, "L_shoulder", size=0.05)
        self.add_coordinate_frame_mocap(root, "R_shoulder", size=0.05)

        self.robot_model = mjcf.from_path(OPENARM_MODEL_PATH)
        root.attach(self.robot_model)

        model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())
        configuration = mink.Configuration(model)
        return model, configuration
    
    def get_R_shoulder_pose(self):
        return self.get_body_pose("openarm/openarm_right_link3")

    def get_L_shoulder_pose(self):
        return self.get_body_pose("openarm/openarm_left_link3")

    def get_R_ee_pose(self):
        return self.get_body_pose("openarm/openarm_right_hand")

    def get_L_ee_pose(self):
        return self.get_body_pose("openarm/openarm_left_hand")

def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    left: str = "follower_left",
    right: str = "follower_right"
):
    
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=30)
    sim = OpenArm1PlayEnv()
    sim_viewer = sim.launch_viewer()
    iter_idx = 0
    while True:
        data_batch = client.stream_get_batch(
            {
                f"{left}_state:actual_angle_rad": np.ndarray,
                # f"{left}_state:actual_velocity_radps": np.ndarray,
                # f"{left}_state:actual_torque_nm": np.ndarray,

                f"{right}_state:actual_angle_rad": np.ndarray,
                # f"{right}_state:actual_velocity_radps": np.ndarray,
                # f"{right}_state:actual_torque_nm": np.ndarray,
            }
        )
        qpos_L = data_batch[f"{left}_state:actual_angle_rad"]
        qpos_R = data_batch[f"{right}_state:actual_angle_rad"]
        sim_qpos_L = np.concatenate([qpos_L[:7], [0, 0]])
        sim_qpos_R = np.concatenate([qpos_R[:7], [0, 0]])
        sim.set_qpos(np.concatenate([sim_qpos_L, sim_qpos_R]))
        sim.update()
        sim.set_mocap_pose("L_ee", sim.get_L_ee_pose())
        sim.set_mocap_pose("R_ee", sim.get_R_ee_pose())
        sim.set_mocap_pose("L_shoulder", sim.get_L_shoulder_pose())
        sim.set_mocap_pose("R_shoulder", sim.get_R_shoulder_pose())
        sim_viewer.sync()
        reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=f"teleop solver")
        iter_idx += 1
        
        if iter_idx == 1:
            print("teleop vis ready")


if __name__ == "__main__":
    typer.run(main)
