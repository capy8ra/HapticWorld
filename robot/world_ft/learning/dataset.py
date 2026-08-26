"""Box-flip dataset for the world_ft project (OpenArm, single head camera).

Adapted from nova/learning/mobile_openarm_graphics_card/dataset.py with these changes:
  * stream names have NO `openarm_` prefix (world_ft recording config), e.g.
    `follower_right_state:actual_angle_rad`, `follower_controller_state:tau_interaction_R`.
  * single camera: `camera_head` (Intel RealSense, 640x480) -> plain center-crop + resize
    (NOT the jieruiweitong fisheye crop).
  * no mobile base.
  * `arms` selects which arm(s) to model. The box-flip task is right-arm only (the left
    arm is static), so the default is ("right",). Use ("left", "right") for bimanual.
  * `use_torque` toggles the interaction torque in the observation (the point of `ft`):
        per arm  obs = [actual_angle(8) (+ tau_interaction(8) if use_torque)]
        per arm  action = [commanded q(8)]

Right-arm + torque (default): obs joint_state = 16-d, action = 8-d.
"""
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from nova.learning.video_mp4_dataset import VideoMp4Dataset
# reuse the shared low-dim loader (already handles both 1-D and 2-D timestamp arrays)
from nova.learning.mobile_openarm_graphics_card.lowdim_dataset import LowDimDataset

# arm name -> (stream infix, tau_interaction suffix)
_ARM_INFO = {
    "left": ("follower_left", "L"),
    "right": ("follower_right", "R"),
}


def get_timestamps(timestamp, horizon=16, duration=2.0, backward=False):
    """Generate `horizon` timestamps relative to `timestamp`, always ascending.

    backward=False (action): [t, t+dt, ..., t+(horizon-1)*dt]
    backward=True  (obs):    [t-(horizon-1)*dt, ..., t-dt, t]
    dt = duration / horizon
    """
    dt = duration / horizon
    if backward:
        return [timestamp - (horizon - 1 - i) * dt for i in range(horizon)]
    return [timestamp + i * dt for i in range(horizon)]


def center_crop_resize(frames, size=224, crop="center"):
    """(T, H, W, 3) -> (T, size, size, 3). Crop to a square then resize.

    Suitable for the RealSense head camera (640x480); no fisheye de-warp.

    crop="center" (default, every task before the microwave): keeps the middle square,
    dropping 80 px from each side of a 640x480 frame.
    crop="right": keeps the RIGHTMOST square (cols 160-640), dropping the left 160 px.
    The microwave rig has both the appliance and the arm in the right of the frame, so
    a center crop cuts through the microwave's control panel and the right edge of the
    gripper; the discarded left strip is empty floor. TRAIN AND DEPLOY MUST USE THE SAME
    MODE — the eval worker reads it from the checkpoint's info.yaml.
    """
    T, H, W, _ = frames.shape
    s = min(H, W)
    top = (H - s) // 2
    if crop == "right":
        left = W - s
    elif crop == "center":
        left = (W - s) // 2
    else:
        raise ValueError(f"unknown crop mode {crop!r} (expected 'center' or 'right')")
    cropped = frames[:, top:top + s, left:left + s]
    return np.stack([cv2.resize(f, (size, size)) for f in cropped], axis=0)


class OpenArmDataset(Dataset):
    def __init__(
        self,
        data_root,
        camera_names=("camera_head",),
        arms=("right",),
        action_horizon=20,
        action_duration=2.0,
        obs_horizon=1,
        obs_duration=1.0,
        image_size=224,
        use_angle=True,
        use_torque=True,
        image_only=False,
        blur_aug_res=0,
        blur_aug_p=0.5,
        ink_enhance=False,
        ink_keypoints_k=0,
        crop="center",
    ):
        super().__init__()
        # Fail fast on stray non-directory entries: VideoMp4Dataset treats every
        # os.listdir entry as a trajectory folder and will crash on files.
        stray = sorted(
            e for e in os.listdir(data_root)
            if not os.path.isdir(os.path.join(data_root, e))
        )
        if stray:
            raise ValueError(
                f"{data_root} contains non-directory entries {stray}. "
                f"Move/remove them before training (each entry is treated as a trajectory)."
            )
        # fail fast on incomplete/aborted sessions (e.g. an interrupted collection missing its video
        # or action npy) so training doesn't later crash with a cryptic video/interp error
        incomplete = []
        for d in sorted(e for e in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, e))):
            p = os.path.join(data_root, d)
            has_video = all(os.path.isfile(os.path.join(p, f"{c}.mp4")) for c in camera_names)
            has_action = all(os.path.isfile(
                os.path.join(p, "low_dim_npys", f"{_ARM_INFO[a][0]}_mit_command:q.npy")) for a in arms)
            if not (has_video and has_action):
                incomplete.append(d)
        if incomplete:
            raise ValueError(
                f"{data_root} has incomplete session(s) {incomplete} (missing a camera video or the "
                f"action npy -- likely an interrupted collection). Remove them before training.")

        self.data_root = data_root
        self.camera_names = list(camera_names)
        self.arms = list(arms)
        self.action_horizon = action_horizon
        self.action_duration = action_duration
        self.obs_horizon = obs_horizon
        self.obs_duration = obs_duration
        self.image_size = image_size
        self.crop = crop  # "center" (all earlier tasks) | "right" (microwave rig)
        self.use_angle = use_angle    # include joint angle in the state
        self.use_torque = use_torque  # include torque in the state
        self.image_only = image_only  # if True, no joint_state at all (state-free / image-only ACT)
        # sharpness augmentation: with prob blur_aug_p, downsample the image to blur_aug_res and
        # back (the exact deploy-time --match-blur-res op). Makes a real-camera-trained policy
        # sharpness-invariant so it also runs on soft WM renders (in-simulator eval). 0 = off.
        self.blur_aug_res = blur_aug_res
        self.blur_aug_p = blur_aug_p
        self.ink_enhance = ink_enhance
        if ink_enhance:
            from ink_detect import enhance_ink  # deferred: only the wipe task needs it
            self._enhance_ink = enhance_ink

        self.ink_keypoints_k = ink_keypoints_k
        self.reference_camera_idx = 0  # first camera provides the reference timeline

        self.video_dataset = VideoMp4Dataset(folder_path=data_root, camera_names=self.camera_names)
        self.lowdim_dataset = LowDimDataset(folder_path=data_root)

        # episode-constant drawing-shape feature: k uniform-arclength points along the
        # stroke detected in each episode's FIRST frame, appended (2k-d, normalized
        # image coords) to every joint_state obs of that episode. Cached per episode
        # dir (sibling of camera_head.mp4 -- NOT in low_dim_npys, which must keep an
        # identical stream set across episodes).
        if ink_keypoints_k:
            assert not image_only, "ink keypoints ride in joint_state (not image_only)"
            from ink_detect import stroke_keypoints
            self._ink_kp = {}
            n_zero = 0
            for tp in self.video_dataset.traj_paths:
                cache = os.path.join(tp, f"ink_keypoints_k{ink_keypoints_k}.npy")
                if os.path.exists(cache):
                    kp = np.load(cache)
                else:
                    cap = cv2.VideoCapture(os.path.join(tp, "camera_head.mp4"))
                    ok, frame = cap.read()
                    cap.release()
                    kp = (stroke_keypoints(frame, k=ink_keypoints_k, bgr=True)
                          if ok else np.zeros((ink_keypoints_k, 2), np.float32))
                    np.save(cache, kp)
                n_zero += int(not kp.any())
                self._ink_kp[tp] = kp.reshape(-1).astype(np.float32)
            print(f"ink keypoints: {ink_keypoints_k} pts ({2 * ink_keypoints_k}-d) per "
                  f"episode, cached; {n_zero} episodes with NO detected stroke"
                  + (" <-- check them!" if n_zero else ""))

        # resolve which torque stream to use: real data has per-arm interaction torque
        # (follower_controller_state:tau_interaction_{L,R}); world-model sim data has a single
        # predicted-torque stream (follower_controller_state:torque_pred_nm). Both are the 8-d
        # "torque" obs, used in the same slot.
        self._torque_kind = None
        if self.use_torque and not self.image_only:
            names = self.lowdim_dataset.data_names or []
            if any(n.startswith("follower_controller_state:tau_interaction") for n in names):
                self._torque_kind = "tau_interaction"
            elif "follower_controller_state:torque_pred_nm" in names:
                self._torque_kind = "torque_pred_nm"
            else:
                raise ValueError(f"use_torque=True but no torque stream found in {data_root} "
                                 f"(need tau_interaction_* or torque_pred_nm).")

        self.reference_timestamps = []
        for traj in self.video_dataset.traj_paths:
            ref_cam = self.camera_names[self.reference_camera_idx]
            self.reference_timestamps.append(self.video_dataset.get_video_timestamps(traj, ref_cam))

        self.length = sum(len(ts) for ts in self.reference_timestamps)
        print(f"dataset initialized with {len(self.video_dataset.traj_paths)} trajectories and "
              f"{self.length} samples in total (arms={self.arms}, image_only={self.image_only}, "
              f"use_angle={self.use_angle}, use_torque={self.use_torque}, torque={self._torque_kind}, "
              f"blur_aug={self.blur_aug_res or 'off'}@p={self.blur_aug_p if self.blur_aug_res else 0}).")

    def __len__(self):
        return self.length

    def _seek(self, traj_path, data_name, timestamp_name, timestamps):
        return self.lowdim_dataset.seek_and_decode(traj_path, data_name, timestamp_name, timestamps)

    def __getitem__(self, idx):
        # map flat idx -> (trajectory, reference timestamp)
        for i, timestamps in enumerate(self.reference_timestamps):
            if idx < len(timestamps):
                traj_path = self.video_dataset.traj_paths[i]
                timestamp = timestamps[idx]
                break
            idx -= len(timestamps)

        data = {"obs": {}, "action": {}}

        # ---- image observations ----
        camera_timestamps = get_timestamps(
            timestamp, horizon=self.obs_horizon, duration=self.obs_duration, backward=True)
        camera_names_to_frames, _ = self.video_dataset.seek_and_decode_multiple_timestamps(
            traj_path, camera_timestamps)
        for cam_name in camera_names_to_frames:
            frames = camera_names_to_frames[cam_name]  # (T, H, W, 3) RGB, full resolution
            if self.ink_enhance:
                # paint the detected drawing saturated red BEFORE the downscale, so a
                # ~1-px stroke survives as a fat unmissable curve (see ink_detect.py)
                frames = np.stack([self._enhance_ink(f, bgr=False) for f in frames], axis=0)
            frames = center_crop_resize(frames, size=self.image_size,
                                        crop=self.crop)  # (T, size, size, 3)
            if self.blur_aug_res and np.random.rand() < self.blur_aug_p:
                r = self.blur_aug_res
                frames = np.stack([cv2.resize(cv2.resize(f, (r, r)),
                                              (self.image_size, self.image_size))
                                   for f in frames], axis=0)
            camera_names_to_frames[cam_name] = \
                np.moveaxis(frames, -1, 1).astype(np.float32) / 255.0  # (T, 3, size, size)
        data["obs"].update(camera_names_to_frames)

        # ---- low-dim observations: per arm [actual_angle (if use_angle)] [torque (if use_torque)] ----
        # skipped entirely for the image-only model (sim data has no follower state anyway)
        if not self.image_only:
            joint_timestamps = get_timestamps(
                timestamp, horizon=self.obs_horizon, duration=self.obs_duration, backward=True)
            obs_parts = []
            for arm in self.arms:
                infix, suffix = _ARM_INFO[arm]
                if self.use_angle:
                    obs_parts.append(self._seek(
                        traj_path, f"{infix}_state:actual_angle_rad",
                        f"{infix}_state:timestamp", joint_timestamps))
                if self.use_torque:
                    tq_name = (f"follower_controller_state:tau_interaction_{suffix}"
                               if self._torque_kind == "tau_interaction"
                               else "follower_controller_state:torque_pred_nm")
                    obs_parts.append(self._seek(
                        traj_path, tq_name, "follower_controller_state:timestamp", joint_timestamps))
            data["obs"]["joint_state"] = np.concatenate(obs_parts, axis=1).astype(np.float32)
            if self.ink_keypoints_k:
                # episode-constant drawing shape, same vector at every obs step
                kp = np.tile(self._ink_kp[traj_path], (self.obs_horizon, 1))
                data["obs"]["joint_state"] = np.concatenate(
                    [data["obs"]["joint_state"], kp], axis=1)

        # ---- action target: per arm commanded joint positions ----
        action_timestamps = get_timestamps(
            timestamp, horizon=self.action_horizon, duration=self.action_duration, backward=False)
        action_parts = []
        for arm in self.arms:
            infix, _ = _ARM_INFO[arm]
            action_parts.append(self._seek(
                traj_path, f"{infix}_mit_command:q",
                f"{infix}_mit_command:timestamp", action_timestamps))
        data["action"] = np.concatenate(action_parts, axis=1).astype(np.float32)
        return data


if __name__ == "__main__":
    dataset = OpenArmDataset(
        data_root="../data/collecting",
        camera_names=["camera_head"],
        arms=["right"],
        obs_horizon=1,
        action_horizon=20,
    )
    data = dataset[len(dataset) // 2]
    print("Observation shapes:")
    for k, v in data["obs"].items():
        print(f"    {k}: {v.shape}")
    print(f"Action shape: {data['action'].shape}")
