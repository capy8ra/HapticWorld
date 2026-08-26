import numpy as np


def pose_xyt_to_se2(pose_xyt):
    x, y, theta = pose_xyt
    return np.array([
        [np.cos(theta), -np.sin(theta),  x],
        [np.sin(theta),  np.cos(theta),  y],
        [0,              0,              1]
    ])


def pose_se2_to_xyt(se2):
    R = se2[:2, :2]
    t = se2[:2, 2]
    theta = np.arctan2(R[1, 0], R[0, 0])
    return np.array([t[0], t[1], theta])


def pose_se2_to_se3(se2):
    se3 = np.eye(4)
    se3[:2, :2] = se2[:2, :2]
    se3[:2, 3] = se2[:2, 2]
    return se3

