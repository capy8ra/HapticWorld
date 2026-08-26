"""Force-viz composites for box-flip WM (simulator) rollouts, from the per-frame
tau_interaction replayed by replay_wm_tau_boxflip.py.

Box-flip twin of the wipe task's render_wm_force_viz.py. One difference that matters:
wiping presses DOWN, so the wipe viz keys contact off -Fz. Box flipping pushes the box
HORIZONTALLY into the wall, so contact lives on +Fy -- measured over the WM teleop demos,
the approach phase reaches only ~5 N mean on +Fy while the push/flip phase runs 15-23 N,
whereas -Fz peaks at ~8 N in contact and carries no usable signal.

Contact threshold 12 N is calibrated on the WM torque head's own training data (the 75
force-feedback teleop demos in data/sim_training): it sits above the 95th-percentile
approach-phase force (11.4 N) and below the weakest per-episode contact peak of all 75
(15.9 N). Shading therefore means "pushing into the box harder than any demo ever did
while still approaching".

Run with the hapticworld env python (needs pinocchio + the rig model):
  python world_ft/learning/render_wm_force_viz_boxflip.py            # everything
  python .../render_wm_force_viz_boxflip.py it_sim__rollout_2 i_sim__rollout_0  # a subset
"""
import glob
import os
import sys

import cv2
import numpy as np
import pinocchio as pin

_HW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HW, "robot"))
import render_force_video as rf  # noqa: E402
from nova.robot.openarm1.solver import OpenArm1Solver  # noqa: E402

OUT = os.path.join(_HW, "data", "force_viz", "boxflip_wm")
TAUD = os.path.join(OUT, "tau")
IWS = os.path.join(_HW, "worldsim")


class WMDat:
    """EpisodeData-compatible object built from replayed (q, tau) at 10 Hz."""

    def __init__(self, tau_npz, solver):
        d = np.load(tau_npz)
        q, tau = d["q"][:, :7], d["tau_interaction"][:, :7]
        self.cap = cv2.VideoCapture(os.path.join(IWS, str(d["video"])))
        self.n = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(q))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0
        model, data, fid = solver.model, solver.data, solver.ee_R_id
        qf = np.zeros(model.nq)
        F = np.empty((self.n, 3))
        for i in range(self.n):
            qf[rf.R_ARM_Q] = q[i]
            pin.computeJointJacobians(model, data, qf)
            J = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)[:, rf.R_ARM_Q]
            F[i] = np.linalg.lstsq(J.T, tau[i], rcond=None)[0][:3]
        self.F = np.stack([rf.ema(F[:, i]) for i in range(3)], 1)
        self.contact = None  # set after the threshold is applied
        self.t = np.arange(self.n) / self.fps

    def frame(self, i):
        return rf.EpisodeData.frame(self, i)[30:]  # crop the baked-in debug label


# box flipping presses +Fy (into the box/wall), not -Fz like wiping
rf.TITLE = "Box flipping - end-effector force from joint torques"
rf.FORCE_AXIS = 1
rf.PRESS_SIGN = 1.0
rf.AXIS_LABEL = "Fy (N)"
rf.TL_LO, rf.TL_HI = -20.0, 25.0     # shared by both timelines, so Fy and Fz compare directly
rf.TL_TICKS = (0, 20, -20)
rf.SECOND_AXIS = 2                   # stack Fz (the lift) under Fy (the push)
rf.SECOND_AXIS_LABEL = "Fz (N)"
rf.THRESH_N = 12.0

os.makedirs(OUT, exist_ok=True)
solver = OpenArm1Solver()
paths = sorted(glob.glob(os.path.join(TAUD, "*_tau.npz")),
               key=lambda p: (os.path.basename(p).split("__")[0],
                              int(os.path.basename(p).split("_")[-2])))
if len(sys.argv) > 1:  # optional stem filter: <policy>__rollout_<idx>
    want = set(sys.argv[1:])
    paths = [p for p in paths if os.path.basename(p)[:-8] in want]
    missing = want - {os.path.basename(p)[:-8] for p in paths}
    if missing:
        raise SystemExit(f"no replayed tau for: {sorted(missing)}")
if not paths:
    raise SystemExit(f"no replayed tau in {TAUD} -- run replay_wm_tau_boxflip.py first")
print(f"contact threshold = {rf.THRESH_N} N on +Fy (demo approach ceiling 11.4 N, "
      f"weakest demo contact peak 15.9 N)")
for p in paths:
    stem = os.path.basename(p)[:-8]  # <policy>__rollout_<idx>
    dat = WMDat(p, solver)
    dat.contact = rf.press_mag(dat.F) > rf.THRESH_N
    policy, ridx = stem.split("__")
    print(f"{stem}: {dat.n} frames, max push {dat.F[:, 1].max():.2f} N, "
          f"contact {dat.contact.mean() * 100:.0f}% of frames")
    rf.render_composite(None, f"policy: {policy} - WM (simulator) rollout {ridx.split('_')[1]}",
                        solver, out_name=f"{stem}_composite.mp4", out_dir=OUT, dat=dat)
print("all done")
