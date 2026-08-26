import numpy as np
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from nova.tools.numpy3dtransform import mat_to_pose, pose_to_mat
from nova.tools.shared_memory.spacemouse import Spacemouse


def get_new_spacemouse_pose(
    sm: Spacemouse,
    pose_curr,
    dt: float,
    max_pos_speed=1.0,
    max_rot_speed=3.0,
    rotated=False,
    no_pos=False,
    xy_transforms_only=False,
    no_rotation=False
):
    assert pose_curr.shape == (4, 4)
    sm_state = sm.get_motion_state_transformed()
    dpos = sm_state[:3] * (max_pos_speed * dt)
    drot_euler = sm_state[3:] * (max_rot_speed * dt)

    if xy_transforms_only:
        drot_euler[:2] = 0.0
        dpos[2] = 0.0

    if no_rotation:
        drot_euler = np.zeros(3)

    if rotated:
        dpos *= np.array([-1, -1, 1])
        drot_euler *= np.array([-1, -1, 1])

    drot = st.Rotation.from_euler("xyz", drot_euler)
    curr_pose_aa = mat_to_pose(pose_curr)
    new_pose_aa = curr_pose_aa.copy()
    if no_pos:
        dpos = np.zeros(3)
    new_pose_aa[:3] += dpos
    new_pose_aa[3:] = (drot * st.Rotation.from_rotvec(curr_pose_aa[3:])).as_rotvec()
    return pose_to_mat(new_pose_aa)


def get_new_spacemouse_pose_se2(
    sm: Spacemouse, xyt_curr, dt: float, max_pos_speed=1.0, max_rot_speed=3.0, rotated=False
):
    assert xyt_curr.shape == (3,)
    sm_state = sm.get_motion_state_transformed()
    dpos = sm_state[:3] * (max_pos_speed * dt)
    drot_euler = sm_state[3:] * (max_rot_speed * dt)
    if rotated:
        dpos *= np.array([-1, -1, 1])
        drot_euler *= np.array([-1, -1, 1])

    theta_z = drot_euler[-1]
    d_xyt = np.r_[dpos[:2], [theta_z]]
    new_pose = xyt_curr + d_xyt
    return new_pose
