"""Force-viz composites for the 2026-07-24 REAL-ROBOT box-flip eval rollouts
(4 policies x 3 episodes) -> data/force_viz/boxflip_eval/.

Box-flip twin of the wipe task's render_eval_force_viz.py, and the real-domain counterpart
of render_wm_force_viz_boxflip.py. Same layout, same +Fy contact axis, and deliberately the
SAME 12 N threshold as the simulator composites: it was calibrated on the WM teleop demos
(approach ceiling 11.4 N, weakest demo contact peak 15.9 N), independently of any eval
episode, and it separates the real evals cleanly -- policies that commit to a push reach
16-25 N, policies that never touch the box stay under 7.6 N. One threshold, both domains.

Videos are remuxed to 0-based PTS first: the recorder writes epoch PTS, and EpisodeData
reads the true epoch t0 from the ORIGINAL file to align the low-dim streams, so the remux
is display-only.

Run with the hapticworld env python:
  python world_ft/learning/render_eval_force_viz_boxflip.py
"""
import os
import subprocess
import sys

_HW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HW, "robot"))
import render_force_video as rf  # noqa: E402
from nova.robot.openarm1.solver import OpenArm1Solver  # noqa: E402

SN = os.path.join(_HW, "robot", "world_ft")
EVAL = os.path.join(SN, "data/eval_rollouts")
OUT = os.path.join(SN, "data/force_viz/boxflip_eval")
TMP = os.path.join(OUT, "_remux")
POLICIES = ["i_sim", "it_sim", "it_real", "it_sim_wo_t"]

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

os.makedirs(TMP, exist_ok=True)
solver = OpenArm1Solver()
print(f"contact threshold = {rf.THRESH_N} N on +Fy (same as the simulator composites)")
for pol in POLICIES:
    for ts in sorted(os.listdir(os.path.join(EVAL, pol))):
        ep_dir = os.path.join(EVAL, pol, ts)
        remux = os.path.join(TMP, f"{pol}__{ts}.mp4")
        if not os.path.exists(remux):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i",
                            os.path.join(ep_dir, "camera_head.mp4"), "-c", "copy",
                            "-avoid_negative_ts", "make_zero",
                            "-video_track_timescale", "90000", remux], check=True)
        dat = rf.EpisodeData(ep_dir, solver, video=remux)
        print(f"{pol}/{ts}: {dat.n} frames, max push {dat.F[:, 1].max():.2f} N, "
              f"contact {dat.contact.mean() * 100:.0f}% of frames")
        rf.render_composite(ep_dir, f"policy: {pol} - real-robot eval rollout", solver,
                            out_name=f"{pol}__{ts}_composite.mp4", video=remux,
                            out_dir=OUT, dat=dat)
print("all done")
