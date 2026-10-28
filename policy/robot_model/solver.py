import numpy as np
import pinocchio as pin
from policy.robot_model import OPENARM_MODEL_PATH


class OpenArm1Solver:
    
    q_slice_arm = np.concatenate([np.arange(7), np.arange(10, 17)])         # 14
    q_slice_arm_gripper = np.concatenate([np.arange(8), np.arange(9, 17)])  # 16
    
    ee_L_name = "openarm_left_hand"
    ee_R_name = "openarm_right_hand"

    def __init__(self):
        self.model = pin.buildModelFromMJCF(OPENARM_MODEL_PATH)
        # self.model = pin.buildModelFromUrdf("/home/jason/openarm/openarm/kinematics/models/openarm.urdf")
        self.data = self.model.createData()
        self.ee_L_id = self.model.getFrameId(self.ee_L_name)
        self.ee_R_id = self.model.getFrameId(self.ee_R_name)

    def forward_kinematics(self, qpos):
        pin.framesForwardKinematics(self.model, self.data, qpos)
    
    def rnea(self, qpos, qvel, qacc):
        tau = pin.rnea(self.model, self.data, qpos, qvel, qacc)
        return tau

    def get_ee_L_pose(self):
        return self.data.oMf[self.ee_L_id].homogeneous

    def get_ee_R_pose(self):
        return self.data.oMf[self.ee_R_id].homogeneous

    def get_gravity_compensation_torques(self, qpos, qvel, qacc):
        tau_gravity = self.rnea(qpos, qvel, qacc)[self.q_slice_arm_gripper]
        return tau_gravity

    # def osc_impedance(self, qpos, desired_ee_L_pose, desired_ee_R_pose, Kp=1.0, Kd=0.1):
    #     current_ee_L_pose = self.get_ee_L_pose()
    #     current_ee_R_pose = self.get_ee_R_pose()

    #     # Compute the error
    #     error_L_pos = desired_ee_L_pose[:3, 3] - current_ee_L_pose[:3, 3]
    #     error_R_pos = desired_ee_R_pose[:3, 3] - current_ee_R_pose[:3, 3]
    #     # print("error_L_pos:", error_L_pos)
    #     error_L_rot = pin.log3(current_ee_L_pose[:3, :3].T @ desired_ee_L_pose[:3, :3])
    #     error_R_rot = pin.log3(current_ee_R_pose[:3, :3].T @ desired_ee_R_pose[:3, :3])
        
    #     # Combine position and orientation errors
    #     error_L = np.concatenate([error_L_pos, error_L_rot])
    #     error_R = np.concatenate([error_R_pos, error_R_rot])
        
    #     # Compute desired wrench
    #     desired_wrench_L = Kp * error_L - Kd * np.zeros(6)  # Assuming zero velocity for simplicity
    #     desired_wrench_R = Kp * error_R - Kd * np.zeros(6)  # Assuming zero velocity for simplicity

    #     # Compute jacobian
    #     J_L = pin.computeFrameJacobian(self.model, self.data, qpos, self.ee_L_id)
    #     J_R = pin.computeFrameJacobian(self.model, self.data, qpos, self.ee_R_id)

    #     # Compute impedance control torques
    #     tau_L = J_L.T @ desired_wrench_L
    #     tau_R = J_R.T @ desired_wrench_R
        
    #     # Combine torques for both arms
    #     return (tau_L + tau_R)[self.q_slice_arm_gripper]