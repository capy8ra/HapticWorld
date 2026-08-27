"""Capture live seed frames for the in-simulator policy rollout (rollout_policy_in_wm.py).

Each ENTER press snapshots the current RealSense frame + the follower's actual joint pose
into a minimal seed hdf5 (same schema as the world_ft converter episodes: one frame of
``obs/images/camera_1_color`` + one row of ``action``). Move the box between presses to
give each simulation rollout a different initial scene.

A WM seed needs BOTH pieces: the frame (encoded to the start latent) and the arm pose at
that instant (seeds the WM's action context + the rollout's homing/gravity reference), so
the arm must be VISIBLE and STATIONARY in frame when you press ENTER. Vary BOTH the box
and the arm pose per episode and the rollouts need no synthetic start-pose randomness.

Recommended flow (see capture_wm_seeds.yaml): teleop-pose the follower via the leader --
  1) process-compose -f world_ft/capture_wm_seeds.yaml
     (camera + right teleop; the leader floats with gravity comp, the follower mirrors it)
  2) this tool: move leader (follower mirrors + holds) + box, ENTER, repeat x20.
The arm pose is read from follower_right_mit_command:q when teleop is running (commanded q
= the WM's action semantics), falling back to the measured angle for a limp/hand-placed arm.

Then roll out with (from HapticWorld/worldsim):
    python scripts/inference/rollout_policy_in_wm.py        # default seed glob = these
Keys (preview window focused): ENTER = capture | q = quit early
"""
import sys
import os

import cv2
import h5py
import numpy as np
import typer
from redis.exceptions import ConnectionError as RedisConnectionError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # robot/ (nova package root)
from nova.redis_client import RedisClient

OBS_KEY = "camera_1_color"  # the WM's obs_keys[0] (stage2 .hydra config)


def write_seed(path, frame_rgb, q8):
    """Minimal seed episode: 1 frame + 1 action row (all seed_from_episode reads)."""
    with h5py.File(path, "w") as f:
        f.create_dataset(f"obs/images/{OBS_KEY}", data=frame_rgb[None].astype(np.uint8))
        f.create_dataset("action", data=np.asarray(q8, np.float32)[None])


def main(
    out_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "wm_seed_live"),
    n: int = 20,
    perception_redis_port: int = 6380,
    control_redis_port: int = 6379,
    follower_right: str = "follower_right",
):
    perception = RedisClient(host="localhost", port=perception_redis_port)
    control = RedisClient(host="localhost", port=control_redis_port)
    os.makedirs(out_dir, exist_ok=True)
    # continue numbering after any existing captures
    existing = [int(e.split("_")[1].split(".")[0]) for e in os.listdir(out_dir)
                if e.startswith("episode_") and e.endswith(".hdf5")]
    idx = max(existing) + 1 if existing else 0
    captured = 0
    print(f"Capturing {n} seeds -> {out_dir} (starting at episode_{idx}). "
          f"Place the box, focus the preview window, ENTER to capture, q to quit.")

    win = "capture_wm_seeds (ENTER=capture, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    warned = False
    while captured < n:
        try:
            img_bgr = perception.stream_get_batch(
                {"camera_head:image": np.ndarray})["camera_head:image"]
            lowdim = control.stream_get_batch({
                f"{follower_right}_mit_command:q": np.ndarray,
                f"{follower_right}_state:actual_angle_rad": np.ndarray,
            })
        except RedisConnectionError:
            if not warned:
                print("redis not reachable -- start the capture stack first:\n"
                      "    process-compose -f world_ft/capture_wm_seeds.yaml\n"
                      "retrying every second ...")
                warned = True
            cv2.waitKey(1000)
            continue
        warned = False
        # prefer the COMMANDED q (teleop running: matches the WM's action semantics exactly);
        # fall back to the measured angle for a limp, hand-placed arm
        if lowdim[f"{follower_right}_mit_command:q"] is not None:
            q8, q_src = lowdim[f"{follower_right}_mit_command:q"], "commanded"
        else:
            q8, q_src = lowdim[f"{follower_right}_state:actual_angle_rad"], "measured"
        if img_bgr is None:
            print("waiting for camera_head:image ...")
            cv2.waitKey(200)
            continue

        preview = img_bgr.copy()
        h, w = preview.shape[:2]
        s = min(h, w)  # the WM only sees the center square crop -- draw it as a guide
        x0, y0 = (w - s) // 2, (h - s) // 2
        cv2.rectangle(preview, (x0, y0), (x0 + s, y0 + s), (0, 255, 0), 1)
        status = (f"{captured}/{n} captured | arm pose: "
                  + (f"OK ({q_src})" if q8 is not None else "MISSING (no follower command/state on redis!)"))
        cv2.putText(preview, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if q8 is not None else (0, 0, 255), 2)
        cv2.imshow(win, preview)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print(f"quit after {captured} captures.")
            break
        if key in (13, 10):  # ENTER
            if q8 is None:
                print("cannot capture: no follower command or state on redis "
                      "(is the follower MIT controller / teleop solver running?)")
                continue
            path = os.path.join(out_dir, f"episode_{idx}.hdf5")
            write_seed(path, img_bgr[..., ::-1], q8)  # camera publishes BGR -> store RGB
            print(f"captured {path}  q({q_src}, deg)={np.rad2deg(q8).round(1).tolist()}")
            idx += 1
            captured += 1
    cv2.destroyAllWindows()
    print(f"done: {captured} seeds in {out_dir}")


if __name__ == "__main__":
    typer.run(main)
