import numpy as np
from nova.damiao_chain import DamiaoChain, MotorType
from nova.tools.precise_sleep import FrequencyRegulator

from nova.sim.mujoco_env import MujocoBaseEnv, mujoco, mink
from dm_control import mjcf
from nova.robot.openarm1.models import OPENARM_MODEL_PATH

import pinocchio as pin
from nova.robot.openarm1.solver import OpenArm1Solver

LEFT_FOLLOWER = 'can0'
RIGHT_FOLLOWER = 'can1'
LEFT_LEADER = 'can2'
RIGHT_LEADER = 'can3'

class OpenArm1PlayEnv(MujocoBaseEnv):
    
    def construct_model(self):
        root = self.get_chessboard_floor()
        self.add_coordinate_frame_mocap(root, "world_origin", size=0.05)
        self.add_coordinate_frame_mocap(root, "ee_L", size=0.05)
        self.add_coordinate_frame_mocap(root, "ee_R", size=0.05)

        self.robot_model = mjcf.from_path(OPENARM_MODEL_PATH)
        root.attach(self.robot_model)

        model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())
        configuration = mink.Configuration(model)
        return model, configuration
    
    def get_qpos(self):
        return self.configuration.data.qpos.copy()
    
    def get_qvel(self):
        return self.configuration.data.qvel.copy()
    
    def get_qacc(self):
        return self.configuration.data.qacc.copy()
    
    def set_qpos(self, qpos):
        self.configuration.data.qpos[:] = qpos

if __name__ == "__main__":
    q_zeros = np.zeros(8)
    reg = FrequencyRegulator(frequency=1000)
    use_sim = True
    if use_sim:
        sim = OpenArm1PlayEnv()
        sim_viewer = sim.launch_viewer()
    robot_solver = OpenArm1Solver()
    with DamiaoChain(
        motor_ids=[1, 2, 3, 4, 5, 6, 7, 8], 
        motor_types=[
            MotorType.DM8009, MotorType.DM8009, 
            MotorType.DM4340, MotorType.DM4340, 
            MotorType.DM4310, MotorType.DM4310, MotorType.DM4310, MotorType.DM4310
        ],
        can_channel=LEFT_LEADER, 
        can_fd=True,
        ) as arm_L, \
        DamiaoChain(
        motor_ids=[1, 2, 3, 4, 5, 6, 7, 8], 
        motor_types=[
            MotorType.DM8009, MotorType.DM8009, 
            MotorType.DM4340, MotorType.DM4340, 
            MotorType.DM4310, MotorType.DM4310, MotorType.DM4310, MotorType.DM4310
        ],
        can_channel=RIGHT_LEADER, 
        can_fd=True,
        ) as arm_R:
        
        data_L = arm_L.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=q_zeros)
        data_R = arm_R.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=q_zeros)
        while True:
            real_qpos = np.concatenate([
                data_L["actual_angle_rad"][:7], 
                [0.0] * 2, 
                data_R["actual_angle_rad"][:7], 
                [0.0] * 2
            ])
            real_qvel = np.concatenate([
                data_L["actual_velocity_radps"][:7], 
                [0.0] * 2, 
                data_R["actual_velocity_radps"][:7], 
                [0.0] * 2
            ])
            
            robot_solver.forward_kinematics(real_qpos)
            tau = robot_solver.get_gravity_compensation_torques(real_qpos, qvel=real_qvel, qacc=np.zeros_like(real_qpos))
            # print(tau.shape)
            
            tau_L = np.concatenate([tau[0:7], [0.0]])
            tau_R = np.concatenate([tau[8:15], [0.0]])
            # tau_L[:3] = 0
            # tau_R[:3] = 0
            data_L = arm_L.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=tau_L * 1)
            data_R = arm_R.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=tau_R * 1)
            
            # data_L = arm_L.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=q_zeros)
            # data_R = arm_R.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=q_zeros)

            if use_sim:
                sim.set_qpos(real_qpos)
                sim.update()
                sim_viewer.sync()
            reg.sleep(verbose=True)
