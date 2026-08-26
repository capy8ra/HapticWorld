"""Synchronized scrubber: head-camera video (left) + 3-D force vector (right).

The mp4 frames and the recorded wrench share the same absolute wall clock:
  * each video frame's PTS * time_base == camera epoch timestamp (set by
    video_recorder.py),
  * the wrench shares follower_controller_state:timestamp (epoch seconds).
So syncing is just nearest-neighbour matching in epoch time -- the video does not
start at t=0, but its PTS already carry absolute time, so no extra offset is needed.

Layout:
  left  : current video frame
  right : 3-D arrow of the current force [Fx, Fy, Fz] (world-aligned, hand TCP)
  bottom: Fx/Fy/Fz context strip with a moving cursor, a draggable time Slider,
          and a Play/Pause button. Click on the strip to jump to a time.

Run `compute_wrist_wrench.py` first so the wrench .npy files exist.
Nothing outside world_ft is modified; nova is not even needed here.
"""

import os
import sys

import numpy as np
import av
import cv2
import typer

# pick backend before importing pyplot: Agg for --snapshot (headless), else TkAgg
_SNAPSHOT_MODE = any(a.startswith("--snapshot") for a in sys.argv)
import matplotlib
matplotlib.use("Agg" if _SNAPSHOT_MODE else "TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)


def _nearest(sorted_ts: np.ndarray, q: float) -> int:
    i = int(np.searchsorted(sorted_ts, q))
    i = min(max(i, 1), len(sorted_ts) - 1)
    return i - int(abs(q - sorted_ts[i - 1]) <= abs(q - sorted_ts[i]))


def _load_video(mp4_path: str, downscale: float):
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    tb = float(stream.time_base)
    frames, epochs = [], []
    for f in container.decode(stream):
        if f.pts is None:
            continue
        img = f.to_ndarray(format="rgb24")
        if downscale != 1.0:
            img = cv2.resize(img, None, fx=downscale, fy=downscale,
                             interpolation=cv2.INTER_AREA)
        frames.append(img)
        epochs.append(f.pts * tb)
    container.close()
    return frames, np.asarray(epochs, dtype=float)


def _load_wrench(npy_dir: str, frame: str, version: str, arm: str):
    suffix = "" if version == "raw" else "_inertiacomp"
    wkey = f"follower_controller_state:wrench_{frame}{suffix}_{arm}"
    wpath = os.path.join(npy_dir, f"{wkey}.npy")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            f"{wpath} not found -- run compute_wrist_wrench.py first "
            f"(frame={frame}, version={version}, arm={arm}).")
    wrench = np.load(wpath)                                   # (N, 6)
    ts = np.load(os.path.join(npy_dir, "follower_controller_state:timestamp.npy")
                 ).astype(float).ravel()
    # recorded timestamps are slightly non-monotonic -> sort for nearest lookup
    order = np.argsort(ts)
    return ts[order], wrench[order]


def main(
    episode: str = typer.Option(..., help="episode folder name under data_dir"),
    data_dir: str = "../data/collecting",
    arm: str = "R",                 # R | L
    version: str = "raw",           # raw | inertiacomp
    frame: str = "hand_tcp",        # hand_tcp | wrist
    downscale: float = 1.0,         # <1.0 shrinks cached frames to save memory
    trail_sec: float = 1.0,         # fading trail of recent force tips (0 disables)
    snapshot: float = None,         # if set: render this time [s], save PNG, exit
):
    assert arm in ("L", "R") and version in ("raw", "inertiacomp")
    ep_dir = os.path.join(data_dir, episode)
    npy_dir = os.path.join(ep_dir, "low_dim_npys")

    print("loading video frames ...")
    frames, frame_epoch = _load_video(os.path.join(ep_dir, "camera_head.mp4"), downscale)
    force_ts, wrench = _load_wrench(npy_dir, frame, version, arm)
    force = wrench[:, :3]
    print(f"  {len(frames)} frames, {len(force)} force samples")

    # shared relative timeline (t=0 at the earlier of the two streams' starts)
    t0 = min(frame_epoch[0], force_ts[0])
    T = max(frame_epoch[-1], force_ts[-1]) - t0
    video_rel = frame_epoch - t0
    force_rel = force_ts - t0
    fmax = float(np.abs(force).max()) * 1.1 + 1e-6

    # ---- figure ----
    fig = plt.figure(figsize=(14, 8))
    ax_video = fig.add_axes([0.03, 0.30, 0.45, 0.64])
    ax3d = fig.add_axes([0.54, 0.30, 0.43, 0.64], projection="3d")
    ax_ctx = fig.add_axes([0.08, 0.15, 0.86, 0.11])
    ax_slider = fig.add_axes([0.08, 0.07, 0.86, 0.03])
    ax_btn = fig.add_axes([0.08, 0.015, 0.07, 0.04])

    ax_video.axis("off")
    im = ax_video.imshow(frames[0])
    vid_title = ax_video.set_title("")

    # 3-D force arrow
    for setlim in (ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim):
        setlim(-fmax, fmax)
    ax3d.set_xlabel("Fx [N]"); ax3d.set_ylabel("Fy [N]"); ax3d.set_zlabel("Fz [N]")
    ax3d.set_box_aspect((1, 1, 1))
    quiver = ax3d.quiver(0, 0, 0, 0, 0, 0, color="crimson", linewidth=2.5,
                         arrow_length_ratio=0.1)
    (tip,) = ax3d.plot([0], [0], [0], "o", color="crimson", ms=5)
    (trail,) = ax3d.plot([], [], [], "-", color="orange", lw=1.0, alpha=0.6)
    f3d_title = ax3d.set_title("")

    # context strip: Fx/Fy/Fz over time + moving cursor
    for k, c in zip(range(3), ("tab:red", "tab:green", "tab:blue")):
        ax_ctx.plot(force_rel, force[:, k], c, lw=0.7, label="FxFyFz"[2 * k:2 * k + 2])
    ax_ctx.set_xlim(0, T); ax_ctx.set_ylabel("force [N]"); ax_ctx.set_xlabel("time [s]")
    ax_ctx.legend(loc="upper right", ncol=3, fontsize=8)
    cursor = ax_ctx.axvline(0, color="k", lw=1.0)

    slider = Slider(ax_slider, "time [s]", 0.0, T, valinit=0.0)
    btn = Button(ax_btn, "Play")

    state = {"quiver": quiver, "playing": False}

    def update(t):
        vi = _nearest(video_rel, t)
        im.set_data(frames[vi])
        vid_title.set_text(f"video  t={t:6.2f}s  frame {vi}/{len(frames) - 1}")

        fi = _nearest(force_rel, t)
        F = force[fi]
        state["quiver"].remove()
        state["quiver"] = ax3d.quiver(0, 0, 0, F[0], F[1], F[2], color="crimson",
                                      linewidth=2.5, arrow_length_ratio=0.1)
        tip.set_data([F[0]], [F[1]]); tip.set_3d_properties([F[2]])
        if trail_sec > 0:
            m = (force_rel >= t - trail_sec) & (force_rel <= t)
            trail.set_data(force[m, 0], force[m, 1]); trail.set_3d_properties(force[m, 2])
        f3d_title.set_text(f"|F|={np.linalg.norm(F):5.2f} N  "
                           f"[{F[0]:+5.1f} {F[1]:+5.1f} {F[2]:+5.1f}]  ({version}, arm {arm})")
        cursor.set_xdata([t, t])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_click(event):
        if event.inaxes is ax_ctx and event.xdata is not None:
            slider.set_val(float(np.clip(event.xdata, 0, T)))
    fig.canvas.mpl_connect("button_press_event", on_click)

    if snapshot is not None:
        update(float(np.clip(snapshot, 0, T)))
        out = os.path.join(ep_dir, f"wrench_view_{arm}_{version}_t{snapshot:.1f}.png")
        fig.savefig(out, dpi=110)
        print(f"snapshot saved: {out}")
        return

    timer = fig.canvas.new_timer(interval=33)

    def tick():
        nt = slider.val + 0.05 * T / max(T, 1)  # ~ realtime-ish step
        nt = slider.val + (33 / 1000.0)
        slider.set_val(0.0 if nt > T else nt)

    timer.add_callback(tick)

    def toggle(_):
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")
        timer.start() if state["playing"] else timer.stop()
    btn.on_clicked(toggle)

    update(0.0)
    plt.show()


if __name__ == "__main__":
    typer.run(main)
