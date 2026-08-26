import os
import shutil

import cv2
import numpy as np
import time
import typer
from nova.redis_client import RedisClient
from nova.tools.data_collection_path import get_data_path
from nova.tools.keycounter import KeystrokeCounter, Key, KeyCode
from nova.tools.time import get_time_hmstz


def discard_episode(output_dir: str, recording_start_time: float, wait_s: float = 5.0) -> None:
    """Delete the episode folder of the recording that started at recording_start_time.

    Called after the recorders report idle (video already closed); waits up to wait_s for
    the low-dim save (low_dim_npys/) to land so a slow write is never deleted mid-save and
    then recreated by it."""
    episode_dir = os.path.dirname(get_data_path("x", output_dir, recording_start_time))
    t0 = time.time()
    while not os.path.isdir(os.path.join(episode_dir, "low_dim_npys")):
        if time.time() - t0 > wait_s:
            print(f"discard: low_dim save not confirmed in {episode_dir} after {wait_s}s "
                  "-- NOT deleting, remove it manually")
            return
        time.sleep(0.1)
    time.sleep(0.3)  # grace for the last npy writes
    shutil.rmtree(episode_dir)
    print(f"DISCARDED episode {episode_dir}")


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    """Resize an image to target_h while preserving aspect ratio."""
    h, w = image.shape[:2]
    if h == target_h:
        return image
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(image, (new_w, target_h))


def render_status_panel(height: int, episode_number: int, is_recording: bool, elapsed_seconds: float) -> np.ndarray:
    """Render the right-hand info panel: episode number, REC indicator, and timer."""
    width = 300
    s = height / 480.0  # scale factor so the panel adapts to the image resolution
    panel = np.full((height, width, 3), 40, dtype=np.uint8)  # dark gray background
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (240, 240, 240)
    gray = (150, 150, 150)
    red = (0, 0, 255)
    green = (0, 200, 0)

    def put(text, x, y_frac, scale, color, thickness):
        cv2.putText(
            panel, text, (int(x * s), int(height * y_frac)),
            font, scale * s, color, max(1, int(round(thickness * s))), cv2.LINE_AA
        )

    # episode number
    put("EPISODE", 20, 0.10, 0.7, gray, 2)
    put(f"{episode_number:03d}", 20, 0.26, 2.4, white, 5)

    # recording indicator (red dot + REC while recording, green dot + IDLE otherwise)
    circle_center = (int(40 * s), int(height * 0.43))
    circle_radius = max(6, int(18 * s))
    if is_recording:
        cv2.circle(panel, circle_center, circle_radius, red, -1)
        put("REC", 70, 0.46, 1.1, red, 3)
    else:
        cv2.circle(panel, circle_center, circle_radius, green, -1)
        put("IDLE", 70, 0.46, 1.1, green, 3)

    # recording time length (mm:ss.d)
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds - 60 * minutes
    put("TIME", 20, 0.66, 0.7, gray, 2)
    put(f"{minutes:02d}:{seconds:04.1f}", 20, 0.80, 1.6, white, 4)

    return panel


def build_display(video_images, recording_camera_names, episode_number, is_recording, elapsed_seconds) -> np.ndarray:
    """Compose the unified window: camera image(s) on the left, info panel on the right."""
    imgs = [video_images[f"{name}:image"] for name in recording_camera_names if video_images.get(f"{name}:image") is not None]

    if len(imgs) == 0:
        # no camera frame yet: show a placeholder so the window (and status) still renders
        cameras = np.full((480, 640, 3), 30, dtype=np.uint8)
        cv2.putText(cameras, "waiting for camera...", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    else:
        ref_h = imgs[0].shape[0]
        cameras = np.hstack([_resize_to_height(im, ref_h) for im in imgs])

    panel = render_status_panel(cameras.shape[0], episode_number, is_recording, elapsed_seconds)
    return np.hstack([cameras, panel])


def main(
    control_redis_host: str = "localhost",
    control_redis_port: int = 6379,
    perception_redis_host: str = "localhost",
    perception_redis_port: int = 6380,
    recording_camera_names: str = "camera_top,camera_bottom",
    start_recording_key: str = "`",
    stop_recording_key: str = "s",
    discard_recording_key: str = "d",
    output_dir: str = "",  # recorders' episode dir; REQUIRED for the discard key to delete
    start_episode: int = 1,
    display_scale: float = 3.0,  # window size multiplier (window stays drag-resizable)
):

    control_redis = RedisClient(host=control_redis_host, port=control_redis_port)
    perception_redis = RedisClient(host=perception_redis_host, port=perception_redis_port)
    recording_camera_names = recording_camera_names.split(",")

    data_batch_maxlen = {
        f"collect_data:command": 100,
        f"collect_data:timestamp": 100
    }

    # UI state (session-local episode counter, driven by the recorders' actual state)
    episode_number = start_episode
    ui_recording = False           # displayed recording state, latched off the recorders
    recording_start_time = None    # reference for the elapsed timer
    pending_start_time = None       # timestamp of a start command awaiting the recorders
    discard_requested = False      # current recording should be deleted once saved

    print(f"keys: {start_recording_key} = record | {stop_recording_key} = stop+save | "
          f"{discard_recording_key} = stop+DISCARD"
          + ("" if output_dir else " (discard disabled: no --output-dir)"))

    window_name = "collect_data"
    iter_idx = 0
    with KeystrokeCounter() as counter:
        while True:
            # get camera states and images
            video_recorder_states_query = {f"video_recorder:{name}:state": str for name in recording_camera_names}
            video_stream_image_query = {f"{name}:image": np.ndarray for name in recording_camera_names}
            all_queries = video_recorder_states_query | video_stream_image_query

            # get data from redis in batch
            input_data = perception_redis.stream_get_batch(all_queries)

            # recorder states, camera stream images
            video_recorder_states = {key: value for key, value in input_data.items() if key in video_recorder_states_query}
            video_images = {key: value for key, value in input_data.items() if key in video_stream_image_query}

            # derive the authoritative recording state from the recorders themselves
            states = list(video_recorder_states.values())
            states_valid = len(states) > 0 and all(state is not None for state in states)
            any_recording = states_valid and any(state == "recording" for state in states)
            all_idle = states_valid and all(state == "idle" for state in states)

            # update recording state based on camera states
            for press_events in counter.get_press_events():
                if press_events == KeyCode(char=start_recording_key):
                    # if all recorders are idle, command all recorders to start recording
                    if all(state == "idle" for state in states):
                        data_batch = {
                            f"collect_data:command": "start_recording",
                            f"collect_data:timestamp": time.time()
                        }

                        perception_redis.stream_add_batch(data_batch, data_batch_maxlen)
                        control_redis.stream_add_batch(data_batch, data_batch_maxlen)
                        pending_start_time = data_batch["collect_data:timestamp"]
                        print(f"collect data start recording at {get_time_hmstz(data_batch['collect_data:timestamp'])}")
                    # refuse to start again if any recorder is still recording
                    else:
                        print("already recording, or waiting all recorder to finish recording the current session")
                elif press_events == KeyCode(char=stop_recording_key):
                    # command all recorders to stop recording
                    data_batch = {
                        f"collect_data:command": "stop_recording",
                        f"collect_data:timestamp": time.time()
                    }
                    perception_redis.stream_add_batch(data_batch, data_batch_maxlen)
                    control_redis.stream_add_batch(data_batch, data_batch_maxlen)
                    print(f"collect data stop recording at {get_time_hmstz(data_batch['collect_data:timestamp'])}")
                elif press_events == KeyCode(char=discard_recording_key):
                    # stop like the stop key, but delete the episode once the recorders
                    # finish saving it (the recorders have no abort -- they always save)
                    if not (ui_recording or any_recording):
                        print("not recording -- nothing to discard")
                    elif not output_dir:
                        print("discard key pressed but --output-dir is not set -- "
                              "stopping WITHOUT deleting")
                    else:
                        discard_requested = True
                    if ui_recording or any_recording:
                        data_batch = {
                            f"collect_data:command": "stop_recording",
                            f"collect_data:timestamp": time.time()
                        }
                        perception_redis.stream_add_batch(data_batch, data_batch_maxlen)
                        control_redis.stream_add_batch(data_batch, data_batch_maxlen)
                        print(f"collect data stop recording (discard={discard_requested}) "
                              f"at {get_time_hmstz(data_batch['collect_data:timestamp'])}")
                else:
                    print("unmapped key:", press_events)

            # latch the displayed recording state off the recorders, count a completed episode
            # on the recording -> idle transition (only leave "recording" on a confirmed idle,
            # so a transient missing reading never miscounts)
            if not ui_recording and any_recording:
                ui_recording = True
                recording_start_time = pending_start_time if pending_start_time is not None else time.time()
                pending_start_time = None
            elif ui_recording and all_idle:
                ui_recording = False
                ended_start_time = recording_start_time
                recording_start_time = None
                if discard_requested:
                    discard_requested = False
                    if output_dir and ended_start_time is not None:
                        discard_episode(output_dir, ended_start_time)
                    # discarded episodes do not advance the episode counter
                else:
                    episode_number += 1

            # elapsed recording time length
            if ui_recording and recording_start_time is not None:
                elapsed_seconds = max(0.0, time.time() - recording_start_time)
            else:
                elapsed_seconds = 0.0

            # render the unified window (image + episode number + recording indicator + timer)
            display = build_display(video_images, recording_camera_names, episode_number, ui_recording, elapsed_seconds)
            if display_scale != 1.0:
                display = cv2.resize(display, None, fx=display_scale, fy=display_scale,
                                     interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, display)
            cv2.waitKey(1)

            if iter_idx == 0:
                print("collect data ready")
            iter_idx += 1
            time.sleep(0.1)


if __name__ == "__main__":
    typer.run(main)
