"""Render submission-ready force-showcase videos for wipe episodes.

Outputs (to data/force_viz/paper/):
  <ep>_composite.mp4        camera view + large 3D force-arrow view
                            + Fz timeline w/ contact shading
  contrast_press_vs_hover.mp4   split screen: press episode vs hover episode,
                            each with gauge + Fz trace; one caption

Force = (J^T)^+ tau_interaction with the rig's own pinocchio model (see
force_viz_prep.py); contact labels from FK EE height (within 5 mm of the
episode's own contact floor). Run with the hapticworld env python:

  python world_ft/learning/render_force_video.py
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import pinocchio as pin

_HW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_HW, "robot"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # robot/ (nova package root)
from nova.robot.openarm1.solver import OpenArm1Solver

ROOT = os.path.join(_HW, "data", "real_training")
VIZ = os.path.join(_HW, "data", "force_viz")
OUT = os.path.join(VIZ, "paper")
R_ARM_Q = slice(9, 16)

# palette (light surface; BGR for cv2)
SURF = (251, 252, 252)
INK = (11, 11, 11)
INK2 = (78, 81, 82)
GRID = (211, 215, 216)
BLUE = (214, 120, 42)      # #2a78d6
ORANGE = (52, 104, 235)    # #eb6834
ORANGE_SOFT = (224, 236, 252)

COMPOSITE_EPS = [("2026_08_04_15_44_41_PDT", "press episode (ink erased)"),
                 ("2026_08_04_17_29_54_PDT", "hover episode (ink survives)")]
CONTRAST = ("2026_08_04_15_44_41_PDT", "2026_08_04_17_29_54_PDT")
THRESH_N = 9.0            # display threshold on -Fz for the CONTACT chip
# 9 N sits above the hover episode's displayed max (7.4 N) and below the
# press episode's wiping plateau (p95 21.6 N) -> hover shows zero contact
GAUGE_MAX_N = 25.0
TAU_MAX_NM = 6.0

# Which force component reads "pressing", and how the timeline is framed. Defaults
# are the wipe task's: press = -Fz (down onto the whiteboard). Other tasks push
# along a different axis -- box flipping presses +Fy into the box/wall -- so a
# driver script overrides these before calling render_composite (see
# render_wm_force_viz_boxflip.py).
TITLE = "Whiteboard wiping - end-effector force from joint torques"
FORCE_AXIS = 2            # index into F: 0=x, 1=y, 2=z
PRESS_SIGN = -1.0         # press magnitude = PRESS_SIGN * F[FORCE_AXIS]
AXIS_LABEL = "Fz (N)"
TL_LO, TL_HI = -GAUGE_MAX_N, 6.0    # timeline y-range on the signed component
TL_TICKS = (0, -20)                 # labelled gridline values on that axis
# Optional SECOND timeline stacked under the first (same time axis, same y-scale, same
# contact shading -- only the plotted component differs). None = single timeline, the
# original wipe layout. Box flipping sets this to 2 to show Fz (the lift) under Fy (the
# push into the box).
SECOND_AXIS = None
SECOND_AXIS_LABEL = ""
# Plot an arbitrary per-frame series on the timeline instead of a component of F -- set to
# the name of an attribute the driver puts on the dat object (e.g. "grip" for gripper
# torque in N.m). The microwave door task uses this: its contact is a grip, not a press.
TL_SERIES_ATTR = None
SHOW_NOTE = True          # the "shaded = press force > N" caption; off when nothing is shaded
NOTE_TEXT = None          # override that caption (e.g. a grip-torque rule); None = default


def press_mag(F):
    """Signed press magnitude (positive = pressing) for one frame or the whole array."""
    return PRESS_SIGN * np.asarray(F)[..., FORCE_AXIS]


def interp_cols(ref_t, src_t, src_v):
    out = np.empty((len(ref_t), src_v.shape[1]))
    for d in range(src_v.shape[1]):
        td = src_t[:, d] if src_t.ndim == 2 else src_t
        o = np.argsort(td, kind="stable")
        out[:, d] = np.interp(ref_t, td[o], src_v[o, d])
    return out


def ema(x, a=0.35):
    y = np.copy(x)
    for i in range(1, len(y)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y


class EpisodeData:
    """Per-video-frame force/tau/contact arrays, EMA-smoothed for display."""

    def __init__(self, ep, solver, video=None):
        ep_dir = ep if os.sep in ep else os.path.join(ROOT, ep)
        video = video or os.path.join(VIZ, os.path.basename(ep_dir) + ".mp4")
        self.cap = cv2.VideoCapture(video)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        ld = os.path.join(ep_dir, "low_dim_npys")
        L = lambda n_: np.load(os.path.join(ld, n_ + ".npy"))  # noqa: E731
        # frame times on the low-dim epoch clock: first controller ts aligns with
        # video start only approximately; use the raw video PTS epoch instead
        import av
        with av.open(os.path.join(ep_dir, "camera_head.mp4")) as c:
            s = c.streams.video[0]
            t0 = float(next(c.decode(s)).pts * s.time_base)
        ts = t0 + np.arange(self.n) / self.fps

        fq = interp_cols(ts, L("follower_right_state:timestamp"),
                         L("follower_right_state:actual_angle_rad"))[:, :7]
        tau = interp_cols(ts, L("follower_controller_state:timestamp"),
                          L("follower_controller_state:tau_interaction_R"))[:, :7]
        model, data, fid = solver.model, solver.data, solver.ee_R_id
        qf = np.zeros(model.nq)
        F = np.empty((self.n, 3))
        z = np.empty(self.n)
        for i in range(self.n):
            qf[R_ARM_Q] = fq[i]
            pin.framesForwardKinematics(model, data, qf)
            z[i] = data.oMf[fid].translation[2]
            pin.computeJointJacobians(model, data, qf)
            J = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)[:, R_ARM_Q]
            F[i] = np.linalg.lstsq(J.T, tau[i], rcond=None)[0][:3]
        self.F = np.stack([ema(F[:, i]) for i in range(3)], 1)
        self.tau = np.abs(np.stack([ema(tau[:, i]) for i in range(7)], 1))
        self.contact = press_mag(self.F) > THRESH_N
        self.t = np.arange(self.n) / self.fps
        self.ts = ts          # frame times on the low-dim epoch clock, for extra streams

    def frame(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        return fr if ok else np.zeros((480, 640, 3), np.uint8)


# ---------- drawing helpers ----------
def text(img, s, xy, scale, color, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def rrect(img, x0, y0, x1, y1, color, filled=True):
    t = -1 if filled else 2
    cv2.rectangle(img, (x0, y0), (x1, y1), color, t, cv2.LINE_AA)


def draw_gauge(img, x, y, w, h, fcomp, label=True):
    """Vertical press-force gauge: fills with the press component; threshold; CONTACT chip."""
    press = max(0.0, PRESS_SIGN * fcomp)
    on = press > THRESH_N
    rrect(img, x, y, x + w, y + h, (240, 240, 238))
    fill = int(h * min(press / GAUGE_MAX_N, 1.0))
    rrect(img, x, y + h - fill, x + w, y + h, ORANGE if on else (170, 170, 168))
    yt = y + h - int(h * THRESH_N / GAUGE_MAX_N)
    cv2.line(img, (x - 6, yt), (x + w + 6, yt), INK2, 2, cv2.LINE_AA)
    for nv in (0, 10, 20):
        yy = y + h - int(h * nv / GAUGE_MAX_N)
        cv2.line(img, (x + w + 4, yy), (x + w + 10, yy), GRID, 2)
        text(img, f"{nv}", (x + w + 14, yy + 6), 0.55, INK2)
    if label:
        text(img, "press force", (x - 8, y + h + 30), 0.62, INK)
        text(img, f"{'-' if PRESS_SIGN < 0 else '+'}{AXIS_LABEL}", (x - 8, y + h + 56), 0.55, INK2)
    chip_c = ORANGE if on else (235, 235, 233)
    rrect(img, x - 10, y - 46, x + w + 44, y - 12, chip_c)
    text(img, "CONTACT" if on else "no contact", (x - 2, y - 21), 0.6,
         (255, 255, 255) if on else INK2, 2 if on else 1)


def draw_joint_bars(img, x, y, w, h, tau):
    bw = w // 7
    text(img, "joint |tau_interaction| (Nm)", (x, y - 12), 0.6, INK)
    for j in range(7):
        bx = x + j * bw
        rrect(img, bx + 6, y, bx + bw - 6, y + h, (240, 240, 238))
        fill = int(h * min(tau[j] / TAU_MAX_NM, 1.0))
        rrect(img, bx + 6, y + h - fill, bx + bw - 6, y + h, BLUE)
        text(img, f"J{j+1}", (bx + bw // 2 - 12, y + h + 24), 0.55, INK2)
    text(img, "wrist", (x + 6 * bw + 2, y + h + 46), 0.5, INK2)


AZ, EL, DIST, FOV = -0.75, 0.42, 2.6, 700


def _proj(p, cx, cy):
    ca, sa, ce, se = np.cos(AZ), np.sin(AZ), np.cos(EL), np.sin(EL)
    x = ca * p[0] + sa * p[1]
    yy = -sa * p[0] + ca * p[1]
    yz = ce * yy + se * p[2]
    zz = -se * yy + ce * p[2]
    d = DIST + yz
    return int(cx + FOV * x / d), int(cy - FOV * zz / d)


def draw_arrow_inset(img, cx, cy, F):
    for i in range(-2, 3):
        for a, b in [([i * .2, -.4, 0], [i * .2, .4, 0]), ([-.4, i * .2, 0], [.4, i * .2, 0])]:
            cv2.line(img, _proj(a, cx, cy), _proj(b, cx, cy), GRID, 2, cv2.LINE_AA)
    for ax, lbl in [([.45, 0, 0], "x"), ([0, .45, 0], "y"), ([0, 0, .45], "z")]:
        cv2.line(img, _proj([0, 0, 0], cx, cy), _proj(ax, cx, cy), INK2, 2, cv2.LINE_AA)
        text(img, lbl, _proj([a * 1.25 for a in ax], cx, cy), 0.7, INK2)
    v = [c / 25.0 for c in F]
    tip = _proj(v, cx, cy)
    cv2.line(img, _proj([0, 0, 0], cx, cy), tip, BLUE, 7, cv2.LINE_AA)
    cv2.circle(img, tip, 10, BLUE, -1, cv2.LINE_AA)


def draw_timeline(img, x, y, w, h, dat, i, note=True, axis=None, label=None, secs=True,
                  series=None):
    fz = series if series is not None else dat.F[:, FORCE_AXIS if axis is None else axis]
    lo, hi = TL_LO, TL_HI
    ty = lambda v: y + int(h * (hi - v) / (hi - lo))  # noqa: E731
    tx = lambda k: x + int(w * k / max(dat.n - 1, 1))  # noqa: E731
    rrect(img, x, y, x + w, y + h, (246, 246, 244))
    on = False
    for k in range(dat.n):
        if dat.contact[k] and not on:
            k0, on = k, True
        if on and (not dat.contact[k] or k == dat.n - 1):
            rrect(img, tx(k0), y, tx(k), y + h, ORANGE_SOFT)
            on = False
    cv2.line(img, (x, ty(0)), (x + w, ty(0)), GRID, 2)
    pts = np.array([[tx(k), ty(fz[k])] for k in range(dat.n)], np.int32)
    cv2.polylines(img, [pts], False, BLUE, 3, cv2.LINE_AA)
    cv2.line(img, (tx(i), y), (tx(i), y + h), INK, 2)
    text(img, AXIS_LABEL if label is None else label, (x + 8, y + 24), 0.55, INK2)
    for tv in TL_TICKS:
        text(img, f"{tv:g}", (x - 22 - 20 * (len(f"{tv:g}") > 1), ty(tv) + 6), 0.5, INK2)
    if secs:
        for sec in range(0, int(dat.t[-1]) + 1, 5):
            k = int(sec * dat.fps)
            if k < dat.n:
                text(img, f"{sec}s", (tx(k) - 12, y + h + 24), 0.5, INK2)
    if note:
        # NOTE_TEXT lets a task state its own shading rule (the microwave door shades on
        # GRIP TORQUE in N.m, not press force in N); default is the wipe/box-flip wording
        text(img, NOTE_TEXT or f"shaded = press force > {THRESH_N:g} N",
             (x + w - 340, y - 10), 0.55, INK2)


def h264(path_tmp, path_out, fps):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path_tmp,
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-r", str(fps), path_out], check=True)
    os.remove(path_tmp)


def render_composite(ep, subtitle, solver, out_name=None, video=None, out_dir=None, dat=None):
    dat = dat or EpisodeData(ep, solver, video)
    W, H = 1920, 1080
    fr0 = dat.frame(0)
    vh = 720 if SECOND_AXIS is None else 660   # shorter panel to fit the second timeline
    vw_px = int(round(vh * fr0.shape[1] / fr0.shape[0]))
    x1 = 40 + vw_px
    cx = (x1 + W - 40) // 2
    dst = out_dir or OUT
    tmp = os.path.join(dst, "_tmp.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), dat.fps, (W, H))
    for i in range(dat.n):
        img = np.full((H, W, 3), SURF, np.uint8)
        text(img, TITLE, (40, 50), 0.95, INK, 2)
        text(img, subtitle, (40, 86), 0.7, INK2)
        fr = cv2.resize(dat.frame(i), (vw_px, vh))
        img[110:110 + vh, 40:x1] = fr
        cv2.rectangle(img, (40, 110), (x1, 110 + vh), GRID, 2)
        draw_arrow_inset(img, cx, 420, dat.F[i])
        text(img, "end-effector force (world frame)", (cx - 220, 80 + vh), 0.7, INK2)
        if SECOND_AXIS is None:
            draw_timeline(img, 90, 890, W - 180, 140, dat, i, note=SHOW_NOTE,
                          series=None if TL_SERIES_ATTR is None else getattr(dat, TL_SERIES_ATTR))
        else:
            draw_timeline(img, 90, 820, W - 180, 100, dat, i, secs=False)
            draw_timeline(img, 90, 950, W - 180, 100, dat, i, note=False,
                          axis=SECOND_AXIS, label=SECOND_AXIS_LABEL)
        vw.write(img)
    vw.release()
    out = os.path.join(dst, out_name or f"{os.path.basename(ep)}_composite.mp4")
    h264(tmp, out, dat.fps)
    print("wrote", out)


def render_contrast(ep_press, ep_hover, solver):
    dp, dh = EpisodeData(ep_press, solver), EpisodeData(ep_hover, solver)
    n = min(dp.n, dh.n)
    fps = dp.fps
    W, H = 1920, 1080
    tmp = os.path.join(OUT, "_tmp.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        img = np.full((H, W, 3), SURF, np.uint8)
        text(img, "Same task, same drawing - only force reveals the difference", (40, 52), 0.95, INK, 2)
        for col, (dat, name) in enumerate([(dp, "PRESS - ink erased"),
                                           (dh, "HOVER - ink survives")]):
            x0 = 40 + col * 940
            fr = cv2.resize(dat.frame(i), (800, 600))
            img[130:730, x0:x0 + 800] = fr
            cv2.rectangle(img, (x0, 130), (x0 + 800, 730), GRID, 2)
            on = press_mag(dat.F[i]) > THRESH_N
            rrect(img, x0, 92, x0 + 380, 124, ORANGE if on else (238, 238, 236))
            text(img, name, (x0 + 10, 115), 0.62,
                 (255, 255, 255) if on else INK, 2 if on else 1)
            draw_gauge(img, x0 + 806, 200, 56, 360, dat.F[i, FORCE_AXIS], label=False)
            dat_n, dat.n = dat.n, n  # timeline over the shared window
            draw_timeline(img, x0 + 50, 790, 750, 120, dat, i, note=False)
            dat.n = dat_n
        vw.write(img)
    vw.release()
    out = os.path.join(OUT, "contrast_press_vs_hover.mp4")
    h264(tmp, out, fps)
    print("wrote", out)


def main():
    os.makedirs(OUT, exist_ok=True)
    solver = OpenArm1Solver()
    for ep, sub in COMPOSITE_EPS:
        render_composite(ep, sub, solver)
    render_contrast(*CONTRAST, solver)


if __name__ == "__main__":
    main()
