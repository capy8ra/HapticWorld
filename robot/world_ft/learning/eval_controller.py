"""world_ft box-flip eval controller (high-rate, ~2 kHz) — RIGHT arm only, no leader.

Rollout flow (no leader/teleop):
  1. Launch the stack. The arm starts LIMP (kp=kd=0 + gravity comp) -- physically place it on the
     table near the box-flip start pose by hand.
  2. Press `h` -> the controller ramps the arm (speed-limited) to the exact start pose and holds it.
  3. Press ENTER -> start the policy rollout (right arm tracks the ACT chunk).
  4. Press `s` -> stop / back to limp at any time.

Because you hand-place near the start pose first, `h`'s straight-line joint ramp is a SMALL move
(it is not collision-aware, so keep the placement close to the start pose).

Feedforward `tau` is gravity/Coriolis compensation from OpenArm1Solver (bimanual model, left joints
held at 0 -- independent chains, so the right-arm gravity torque is unaffected). This matches how
`mit_command:tau` was produced during collection, so the tau_interaction the controller publishes is
train/eval-consistent.

SAFETY: this drives real motors and has NOT been validated on hardware here. Verify the start pose
and gravity comp in idle/`h` before pressing ENTER. Keep a hand on the e-stop.
"""
import sys
import os
import time
from typing import Optional

import numpy as np
import pinocchio as pin
import typer
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # robot/ (nova package root)
from nova.redis_client import RedisClient
from nova.tools.precise_sleep import FrequencyRegulator
from nova.robot.openarm1.solver import OpenArm1Solver
from nova.tools.keycounter import KeystrokeCounter, Key, KeyCode
from nova.tools.pose_interp import get_interp1d
from nova.tools.npy_read_write import load_dict_from_npy
from nova.tools.data_collection_path import get_data_path

from ckpt_select import resolve_ckpt, pick_reference_episode

FOLLOWER_KP = np.array([240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0])
FOLLOWER_KD = np.array([3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])

# --grasp-flow gripper constants (same values as ball_throw_solver.py's collection grasp)
GRIP_CLOSE_RATE_RAD_S = 0.4
GRIP_HOLD_S = 0.15
GRIP_CLOSE_MARGIN_RAD = 0.15


def _grip_reference(episode, window_s=0.5):
    """(open, hold, torque target) for --grasp-flow, from the reference episode itself:
    open = mean commanded gripper q over the LAST window (post-release), hold / torque
    target = means over the FIRST window (ball grasped at episode start). The collection
    rig enforces identical episode starts, so one episode stands in for the dataset mean.

    World-model SIM episodes (wipe_sim_act & co) carry no measured state; the
    WM-predicted TOTAL torque (trained on actual_torque_nm) stands in for the
    gripper torque target."""
    q = episode["follower_right_mit_command:q"]
    qts = episode["follower_right_mit_command:timestamp"]
    if "follower_right_state:actual_torque_nm" in episode:
        tq = episode["follower_right_state:actual_torque_nm"]
        tts = episode["follower_right_state:timestamp"]
    else:
        tq = episode["follower_controller_state:torque_pred_nm"]
        tts = episode["follower_controller_state:timestamp"]
        print("[grasp-flow] sim reference episode (no measured state) -> gripper "
              "torque target from WM torque_pred_nm")
    if tts.ndim > 1:
        tts = tts[:, 0]
    n0 = max(1, int((qts - qts[0] < window_s).sum()))
    m0 = max(1, int((tts - tts[0] < window_s).sum()))
    n1 = max(1, int((qts[-1] - qts < window_s).sum()))
    return float(q[-n1:, 7].mean()), float(q[:n0, 7].mean()), float(tq[:m0, 7].mean())


def _publish_recording_command(clients, command):
    """Start/stop the eval.yaml recorder workers (same redis protocol as nova collect_data:
    video_recorder listens on the perception bus, low_dim_npy_recorder on the control bus)."""
    t = time.time()
    batch = {"collect_data:command": command, "collect_data:timestamp": t}
    for c in clients:
        c.stream_add_batch(batch, {k: 10 for k in batch})
    return t


def ee_pos_R(solver, qR7):
    """World position of the right EE at arm configuration qR7 (7 joints)."""
    q = np.concatenate([np.zeros(9), np.asarray(qR7, float), [0.0, 0.0]])
    pin.framesForwardKinematics(solver.model, solver.data, q)
    return solver.get_ee_R_pose()[:3, 3].copy()


def offset_ee_xyz(solver, qR7, dxyz):
    """7 right-arm joints that move the right EE by dxyz meters in world axes (position only,
    orientation free), via small damped-least-squares Jacobian IK. World Z is up. Right arm
    occupies model config indices 9:16."""
    q = np.concatenate([np.zeros(9), np.asarray(qR7, float), [0.0, 0.0]])
    pin.framesForwardKinematics(solver.model, solver.data, q)
    target = solver.get_ee_R_pose()[:3, 3] + np.asarray(dxyz, float)
    for _ in range(60):
        pin.framesForwardKinematics(solver.model, solver.data, q)
        pin.computeJointJacobians(solver.model, solver.data, q)
        err = target - solver.get_ee_R_pose()[:3, 3]
        if np.linalg.norm(err) < 1e-4:
            break
        J = pin.getFrameJacobian(solver.model, solver.data, solver.ee_R_id,
                                 pin.LOCAL_WORLD_ALIGNED)[:3, 9:16]
        q[9:16] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
    return q[9:16]


def z_lift_dq(solver, qR7, dz):
    """7 right-arm joint deltas lifting the EE by dz meters in world Z: one damped
    least-squares Jacobian step at qR7 (dz stays small, so one step is accurate).
    Called at ~100 Hz by the press-relief admittance, so no iterative IK here."""
    q = np.concatenate([np.zeros(9), np.asarray(qR7, float), [0.0, 0.0]])
    pin.framesForwardKinematics(solver.model, solver.data, q)
    pin.computeJointJacobians(solver.model, solver.data, q)
    J = pin.getFrameJacobian(solver.model, solver.data, solver.ee_R_id,
                             pin.LOCAL_WORLD_ALIGNED)[:3, 9:16]
    err = np.array([0.0, 0.0, float(dz)])
    return J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)


def lower_action_chunk(solver, chunk, dz):
    """Apply an EE Z-offset (dz meters) to every pose in a (H, 8) commanded-q chunk (arm joints only).
    Called only when (re)building the trajectory interpolator (~every few s), not per control tick."""
    if dz == 0.0:
        return chunk
    out = np.asarray(chunk, dtype=float).copy()
    for i in range(out.shape[0]):
        out[i, :7] = offset_ee_xyz(solver, out[i, :7], [0.0, 0.0, dz])
    return out


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    follower_right: str = "follower_right",
    frequency_hz: float = 2000,
    # checkpoint selector: 'i' = latest image-only, 'it' = latest image+torque, 'its' = latest
    # image+state+torque; optional training-domain suffix '_real'/'_sim' (e.g. 'it_sim'; bare =
    # latest from either domain). An explicit model.pth path also works. When set, the home
    # reference episode is auto-derived from that checkpoint's training dataset (closest-to-mean
    # start pose), overriding reference_episode_lowdim_path. Use the SAME selector for the
    # inference worker in eval.yaml.
    model: Optional[str] = None,
    # manual fallback (used only when --model is not given): a training episode whose first
    # commanded q defines the reference start pose. MUST match the policy you're rolling out
    # (its demos' start pose) or the arm jumps when the policy engages:
    #   - sim-trained policy  -> a sim_training episode
    #   - real-trained policy -> a real_training episode
    reference_episode_lowdim_path: str = (
        "../data/sim_training/2026_07_22_10_50_13/low_dim_npys"),
    model_inference_worker_name: str = "model_inference",
    chunk_renew_every_s: float = 2.0,
    stitch_duration: float = 0.45,
    # command dq from the chunk (central difference). Teleop collection commands
    # dq = leader velocity; without this, deploy-time kd drags against fast motion the
    # demos never felt (~kd*qdot, significant only for dynamic tasks like ball-throw).
    dq_feedforward: bool = False,
    # collection-style start flow (ball-throw / wipe): 'h' homes with the gripper WIDE
    # OPEN, the first ENTER closes it onto the hand-placed object until the reference grip
    # force holds, the second ENTER starts the rollout. Once grasped, every later 'h'
    # re-homes WITH the object kept grasped (and idle keeps the gripper servoed); 'o'
    # releases. Off = legacy flow (home pose incl. gripper, ENTER rolls out immediately).
    grasp_flow: bool = False,
    grip_torque_nm: float = -1.0,  # grasp target; <0 -> from the reference episode
    # gripper-open pose; nan -> reference episode's last window (ball-style release at the
    # end). Wipe-style episodes never release, so a fallback of -1.03 rad kicks in there.
    grip_open_rad: float = float("nan"),
    # lock the gripper at the grasp pose DURING the rollout (wipe: the demos' gripper is
    # constant). Leave off for ball-throw -- the policy must release to throw.
    lock_grip: bool = False,
    # deploy-time correction: lower (-) / raise (+) EVERY predicted reach pose by this many meters in Z
    # (fixes a systematic vertical bias, e.g. the image-only policy reaching above a short box)
    action_z_offset_m: float = -0.00,
    # press-relief admittance (wipe): during the rollout, when |tau_interaction| on joints
    # 1-4 exceeds press_target_nm, LIFT the commanded trajectory (up to press_relief_max_m)
    # until the press returns to target -- the deploy-time stand-in for the operator's
    # force moderation during collection. Relief decays when the press is light, so the
    # policy can still push down to demo-level force. Target default = the wipe demos'
    # median per-episode p95 press (5.5 N.m).
    press_relief: bool = False,
    press_target_nm: float = 5.5,
    press_relief_max_m: float = 0.04,
    press_relief_up_mps: float = 0.03,
    press_relief_decay_mps: float = 0.01,
    # homing (`h`): speed-limited so the move to the start pose stays slow/safe
    homing_min_s: float = 1.5,
    homing_max_vel_deg_s: float = 30.0,
    # randomized start pose (robustness probe, e.g. for sim-trained policies whose demos all start
    # from one seed pose): each `h` press homes to the reference pose shifted by a fresh uniform EE
    # offset -- x,y in +/-home_rand_xy_m, z in [home_rand_z_min_m, home_rand_z_max_m] (position-only
    # IK, orientation free). The sampled and achieved offsets are printed per trial for logging.
    random_home: bool = False,
    home_rand_xy_m: float = 0.1,
    home_rand_z_min_m: float = -0.02,
    home_rand_z_max_m: float = 0.03,
    # --save: record each rollout (ENTER -> s / h / exit) as one episode via the recorder
    # workers in eval.yaml (video_recorder on the perception bus + low_dim_npy_recorder on the
    # control bus). Episodes -> data/eval_rollouts/<--model spec>/<timestamp>/ with
    # camera_head.mp4 + low_dim_npys/ + meta.yaml. The recorders' output dir comes from
    # $WORLD_FT_MODEL in eval.yaml -- launch the stack with the SAME spec as --model.
    save: bool = False,
    perception_redis_port: int = 6380,
    name: str = "eval_controller",
):
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=frequency_hz)
    solver = OpenArm1Solver()
    q_zeros = np.zeros(8)

    resolved_model_path = None
    if model is not None:
        resolved_model_path, ckpt_cfg = resolve_ckpt(model)
        reference_episode_lowdim_path = pick_reference_episode(ckpt_cfg["dataset_path"])
        print(f"[--model {model!r}] ckpt: {resolved_model_path}\n"
              f"[--model {model!r}] home reference (closest-to-mean start of "
              f"{ckpt_cfg['dataset_path']}): {reference_episode_lowdim_path}\n"
              f"[--model {model!r}] make sure the inference worker (eval.yaml) uses the same ckpt.")

    # --save: rollout recording state. rec['on'] flips at ENTER (start) and s/h/exit (stop).
    rec = {"on": False}
    last_dxyz = np.zeros(3)
    if save:
        perception_client = RedisClient(host=redis_host, port=perception_redis_port)
        rec_clients = (client, perception_client)
        save_output_dir = f"../data/eval_rollouts/{model if model is not None else 'manual'}"
        if perception_client.stream_get_batch(
                {"video_recorder:camera_head:state": str})["video_recorder:camera_head:state"] is None:
            print("[save] WARNING: no video_recorder detected on the perception bus. Launch the "
                  "eval.yaml stack (it includes the recorder workers) or nothing will be saved.")
        print(f"[save] each ENTER->s rollout will be recorded to {save_output_dir}/<timestamp>/ "
              f"(recorders write to $WORLD_FT_MODEL -- launch the stack with the same spec).")

        import atexit

        def _stop_recording_on_exit():
            if rec["on"]:
                try:
                    _publish_recording_command(rec_clients, "stop_recording")
                    print("[save] recording stopped & saved (exit).")
                except Exception:
                    pass

        atexit.register(_stop_recording_on_exit)

    reference_episode = load_dict_from_npy(reference_episode_lowdim_path)
    # commanded q is present in BOTH real and sim episodes (sim has no measured state); its first
    # value is the pose the policy's demos start from -> home here so the policy engages in-distribution
    reference_qpos_R = reference_episode["follower_right_mit_command:q"][0]
    print(f"Reference start qpos_R (deg): {np.rad2deg(reference_qpos_R).round(1).tolist()}")

    grip_cmd = grip_hold_time = 0.0
    grasp_q = None  # gripper q holding the object; kept across re-homes until 'o'
    if grasp_flow:
        grip_open, grip_hold_ref, grip_t_ref = _grip_reference(reference_episode)
        if not np.isnan(grip_open_rad):
            grip_open = grip_open_rad
        elif abs(grip_open - grip_hold_ref) < 0.05:
            # episodes that START and END grasped (wipe): the last-window "open" is really
            # the hold pose -- fall back to the collection rig's open pose
            grip_open = -1.03
            print(f"[grasp-flow] reference episode never releases -> open pose "
                  f"{grip_open:.2f} rad (override with --grip-open-rad)")
        if grip_torque_nm > 0:
            grip_t_ref = grip_torque_nm
        grip_close_dir = float(np.sign(grip_hold_ref - grip_open))
        grip_limit = grip_hold_ref + grip_close_dir * GRIP_CLOSE_MARGIN_RAD
        grip_cmd = grip_open
        print(f"[grasp-flow] gripper open={grip_open:.3f} rad | close limit={grip_limit:.3f} | "
              f"grip torque target={grip_t_ref:.2f} N.m | lock_grip={lock_grip}\n"
              "[grasp-flow] flow: h = home (gripper open) -> place object -> ENTER = grasp -> "
              "ENTER = rollout -> h = re-home (object KEPT) -> ENTER = rollout ... 'o' = release")

    eval_state = "idle"  # idle, enabling (homing), [grasping, grasped,] evaluating
    qpos_interp_R = None
    z_relief = 0.0            # press-relief admittance lift (m), rollout only
    dq_relief = np.zeros(7)   # joint-space image of z_relief, refreshed at ~100 Hz
    relief_every = max(1, int(frequency_hz // 100))
    if press_relief:
        print(f"[press-relief] ON: lift up to {press_relief_max_m*100:.0f} cm to keep "
              f"|tau_interaction| (joints 1-4) near {press_target_nm:.1f} N.m")

    with KeystrokeCounter() as key_counter:
        iter_idx = 0
        while True:
            t_now = time.time()
            data_batch = client.stream_get_batch({
                f"{follower_right}_state:actual_angle_rad": np.ndarray,
                f"{follower_right}_state:actual_velocity_radps": np.ndarray,
                f"{follower_right}_state:actual_torque_nm": np.ndarray,
                f"{model_inference_worker_name}:action_timestamp": np.ndarray,
                f"{model_inference_worker_name}:action_desired_qpos_right": np.ndarray,
            })

            model_action_timestamp = data_batch[f"{model_inference_worker_name}:action_timestamp"]
            model_desired_qpos_right = data_batch[f"{model_inference_worker_name}:action_desired_qpos_right"]
            qpos_R = data_batch[f"{follower_right}_state:actual_angle_rad"]
            qvel_R = data_batch[f"{follower_right}_state:actual_velocity_radps"]
            tau_actual_R = data_batch[f"{follower_right}_state:actual_torque_nm"]
            if any(v is None for v in (qpos_R, qvel_R)):
                reg.sleep()
                continue

            # gravity/Coriolis compensation for the RIGHT arm. The solver models both arms; the left
            # is held at 0 (independent chain -> does not affect the right-arm gravity torque).
            sim_qpos = np.concatenate([np.zeros(9), np.concatenate([qpos_R[:7], [0, 0]])])
            sim_qvel = np.concatenate([np.zeros(9), np.concatenate([qvel_R[:7], [0, 0]])])
            solver.forward_kinematics(sim_qpos)
            tau = solver.get_gravity_compensation_torques(sim_qpos, sim_qvel, np.zeros_like(sim_qpos))
            desired_tau_R = np.concatenate([tau[8:15], [0.0]])

            desired_timestamp = time.time()
            for press in key_counter.get_press_events():
                if press == KeyCode(char="h"):
                    if save and rec["on"]:  # homing ends the rollout -> close the episode
                        _publish_recording_command(rec_clients, "stop_recording")
                        rec["on"] = False
                        print("[save] recording stopped & saved (homing).")
                    eval_state = "enabling"
                    home_qpos_R = reference_qpos_R
                    if random_home:
                        # fresh EE offset per `h` press; IK from the reference pose (position only)
                        dxyz = np.array([
                            np.random.uniform(-home_rand_xy_m, home_rand_xy_m),
                            np.random.uniform(-home_rand_xy_m, home_rand_xy_m),
                            np.random.uniform(home_rand_z_min_m, home_rand_z_max_m),
                        ])
                        home_qpos_R = reference_qpos_R.copy()
                        home_qpos_R[:7] = offset_ee_xyz(solver, reference_qpos_R[:7], dxyz)
                        last_dxyz = dxyz
                        achieved = ee_pos_R(solver, home_qpos_R[:7]) - ee_pos_R(solver, reference_qpos_R[:7])
                        print(f"[random-home] EE offset sampled (cm): x{dxyz[0]*100:+.1f} "
                              f"y{dxyz[1]*100:+.1f} z{dxyz[2]*100:+.1f} | achieved: "
                              f"x{achieved[0]*100:+.1f} y{achieved[1]*100:+.1f} z{achieved[2]*100:+.1f}")
                        if np.linalg.norm(achieved - dxyz) > 0.01:
                            print("[random-home] WARNING: IK fell >1 cm short of the sampled offset "
                                  "(target near workspace limit); homing to the achieved pose.")
                    if grasp_flow:
                        home_qpos_R = home_qpos_R.copy()
                        if grasp_q is not None:
                            # re-home KEEPING the grasped object (collection-style)
                            home_qpos_R[7] = grip_cmd = grasp_q
                            eval_state = "grasped"
                        else:
                            # first home: arrive with the gripper WIDE OPEN
                            home_qpos_R[7] = grip_cmd = grip_open
                    # duration scales with distance, capped to homing_max_vel_deg_s, so the ramp
                    # to the start pose stays slow/safe.
                    max_disp = float(np.max(np.abs(home_qpos_R - qpos_R)))
                    homing_s = max(homing_min_s, max_disp / np.deg2rad(homing_max_vel_deg_s))
                    enabling_timestamps = np.array([desired_timestamp, desired_timestamp + homing_s])
                    qpos_interp_R = get_interp1d(enabling_timestamps,
                                                 np.stack([qpos_R, home_qpos_R], axis=0))
                    print(f"Homing {np.rad2deg(max_disp):.1f} deg to the start pose over {homing_s:.1f} s, "
                          "then holding. "
                          + ("Object kept grasped -- press ENTER to roll out."
                             if grasp_flow and grasp_q is not None else
                             "Place the object, then press ENTER to close the gripper."
                             if grasp_flow else "Press ENTER to roll out."))
                elif press in (Key.enter, KeyCode(char="\r"), KeyCode(char="\n"), KeyCode(char="g")):
                    # Enter (or `g` as a fallback if this terminal doesn't deliver Key.enter).
                    # With --grasp-flow the first ENTER (from the homed hold) closes the gripper;
                    # only the second one (state "grasped") starts the rollout.
                    if grasp_flow and eval_state == "enabling":
                        grip_hold_time = 0.0
                        eval_state = "grasping"
                        print("[grasp-flow] closing gripper ...")
                    elif grasp_flow and eval_state == "grasping":
                        print("[grasp-flow] still closing -- wait for the grasp to seat")
                    elif grasp_flow and eval_state == "idle":
                        print("[grasp-flow] arm is limp -- press h to home first")
                    elif model_action_timestamp is not None and model_desired_qpos_right is not None:
                        # refuse to roll out a DIFFERENT checkpoint than the worker runs
                        # (eval.yaml's WORLD_FT_MODEL defaults to it_sim -- forgetting it
                        # silently runs the box-flip policy; cost real time once)
                        worker_ckpt = client.stream_get_batch(
                            {f"{model_inference_worker_name}:ckpt_path": str}
                        )[f"{model_inference_worker_name}:ckpt_path"]
                        if (resolved_model_path is not None and worker_ckpt is not None
                                and os.path.abspath(worker_ckpt) != os.path.abspath(resolved_model_path)):
                            print("REFUSING to roll out: the inference worker runs a DIFFERENT "
                                  f"checkpoint!\n  worker:     {worker_ckpt}\n"
                                  f"  controller: {resolved_model_path}\n"
                                  "  Relaunch the stack with WORLD_FT_MODEL=<same spec as --model>.")
                            continue
                        eval_state = "evaluating"
                        # ramp from the CURRENT pose into the first predicted chunk over stitch_duration
                        # so the policy engaging doesn't snap the arm (e.g. if home != the policy's
                        # exact start pose). Same stitch trick used for chunk renewal below.
                        chunk = lower_action_chunk(solver, model_desired_qpos_right, action_z_offset_m)
                        t0 = time.time()
                        ramp_ts = np.concatenate([[t0], model_action_timestamp + stitch_duration])
                        ramp_q = np.concatenate([qpos_R[None, :], chunk], axis=0)
                        qpos_interp_R = get_interp1d(ramp_ts, ramp_q)
                        print(f"Rollout: ramping into the policy trajectory "
                              f"(EE Z offset {action_z_offset_m * 100:+.0f} cm).")
                        if save and not rec["on"]:
                            t_rec = _publish_recording_command(rec_clients, "start_recording")
                            rec["on"] = True
                            # meta.yaml alongside the recorders' camera_head.mp4 + low_dim_npys/
                            meta_path = get_data_path("meta", save_output_dir, t_rec) + ".yaml"
                            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                            with open(meta_path, "w") as f:
                                yaml.safe_dump(dict(
                                    model=model,
                                    model_path=resolved_model_path,
                                    reference_episode_lowdim_path=reference_episode_lowdim_path,
                                    action_z_offset_m=action_z_offset_m,
                                    chunk_renew_every_s=chunk_renew_every_s,
                                    stitch_duration=stitch_duration,
                                    random_home=random_home,
                                    home_offset_dxyz=[float(x) for x in last_dxyz],
                                    recording_start_time=float(t_rec)), f)
                            print(f"[save] recording -> {os.path.dirname(meta_path)}")
                    else:
                        print("Cannot start rollout: no model action on redis yet. Is "
                              "model_inference_worker publishing 'model_inference:action_desired_qpos_right'?")
                elif press == KeyCode(char="o"):
                    if grasp_flow and eval_state in ("enabling", "grasped"):
                        print("[grasp-flow] opening gripper (release) -- place the object, "
                              "then ENTER to grasp again")
                        grasp_q = None
                        grip_cmd = grip_open
                        eval_state = "enabling"
                    else:
                        print("[grasp-flow] 'o' releases only from the homed hold "
                              "(h first)" if grasp_flow else f"[key] {press!r} registered")
                elif press == KeyCode(char="s"):
                    eval_state = "idle"
                    qpos_interp_R = None
                    if save and rec["on"]:
                        _publish_recording_command(rec_clients, "stop_recording")
                        rec["on"] = False
                        print("[save] recording stopped & saved.")
                    print("Idle: limp with gravity comp (place the arm, then press `h`).")
                else:
                    print(f"[key] {press!r} registered (h=home, ENTER/g=grasp/rollout, "
                          "s=stop, o=release)")

            # right-arm chunk renewal / stitching near the end of the current chunk
            if eval_state == "evaluating" and \
                    t_now > qpos_interp_R.x[0] + chunk_renew_every_s + stitch_duration:
                last_end_time = qpos_interp_R.x[0] + chunk_renew_every_s + stitch_duration
                last_end_qpos_R = qpos_interp_R(last_end_time)
                chunk = lower_action_chunk(solver, model_desired_qpos_right, action_z_offset_m)
                stitched_ts = np.concatenate([[last_end_time], model_action_timestamp + stitch_duration])
                stitched_qpos_R = np.concatenate([last_end_qpos_R[None, :], chunk], axis=0)
                qpos_interp_R = get_interp1d(stitched_ts, stitched_qpos_R)
                print("Renewing right-arm interpolator with latest model output.")

            # --grasp-flow: close the gripper until the reference grip torque holds
            if grasp_flow and eval_state == "grasping":
                grip_cmd += grip_close_dir * GRIP_CLOSE_RATE_RAD_S / frequency_hz
                grip_torque = float(tau_actual_R[7]) if tau_actual_R is not None else 0.0
                if grip_close_dir * (grip_cmd - grip_limit) >= 0:
                    print("[grasp-flow] close limit reached without grip torque -- no ball? "
                          "reopening (place the ball, then ENTER again)")
                    grip_cmd = grip_open
                    eval_state = "enabling"
                elif grip_close_dir * grip_torque >= grip_t_ref:
                    grip_hold_time += 1.0 / frequency_hz
                    if grip_hold_time >= GRIP_HOLD_S:
                        eval_state = "grasped"
                        grasp_q = grip_cmd  # kept across re-homes until 'o'
                        print(f"[grasp-flow] grasped: torque {grip_torque:.2f} N.m "
                              f"(target {grip_t_ref:.2f}), gripper q {grip_cmd:.3f} rad -- "
                              "press ENTER to roll out")
                else:
                    grip_hold_time = 0.0

            # assemble the right-arm command for this control tick
            if eval_state == "idle":
                cmd_kp_R = cmd_kd_R = q_zeros
                cmd_q_R = qpos_R
                cmd_dq_R = q_zeros
                if grasp_flow and grasp_q is not None:
                    # arm limp, but keep holding the object (collection-style)
                    cmd_kp_R = np.concatenate([np.zeros(7), [FOLLOWER_KP[7]]])
                    cmd_kd_R = np.concatenate([np.zeros(7), [FOLLOWER_KD[7]]])
                    cmd_q_R = np.concatenate([qpos_R[:7], [grasp_q]])
            elif eval_state in ("enabling", "grasping", "grasped", "evaluating"):
                cmd_kp_R, cmd_kd_R = FOLLOWER_KP, FOLLOWER_KD
                cmd_q_R = qpos_interp_R(t_now)
                if grasp_flow and eval_state in ("enabling", "grasping", "grasped"):
                    # arm holds home; gripper under grasp control ('o' can reopen it here)
                    cmd_q_R = cmd_q_R.copy()
                    cmd_q_R[7] = grip_cmd
                elif eval_state == "evaluating" and lock_grip and grasp_q is not None:
                    cmd_q_R = cmd_q_R.copy()
                    cmd_q_R[7] = grasp_q  # --lock-grip: policy never commands the gripper
                if dq_feedforward:
                    h = 0.002  # central difference on the chunk interpolator
                    cmd_dq_R = (qpos_interp_R(t_now + h) - qpos_interp_R(t_now - h)) / (2 * h)
                else:
                    cmd_dq_R = q_zeros
            else:
                raise ValueError(f"Unknown eval_state: {eval_state}")

            # press-relief admittance: the deploy-time stand-in for the operator's force
            # moderation. While the measured press exceeds the target, ramp a world-Z lift
            # into the commanded pose; decay it when the press is light again.
            if press_relief:
                if eval_state == "evaluating" and tau_actual_R is not None:
                    press = float(np.abs((tau_actual_R - desired_tau_R)[:4]).max())
                    if press > press_target_nm:
                        z_relief += (press_relief_up_mps / frequency_hz) * min(
                            (press - press_target_nm) / press_target_nm, 2.0)
                    else:
                        z_relief -= press_relief_decay_mps / frequency_hz
                    z_relief = float(np.clip(z_relief, 0.0, press_relief_max_m))
                    if z_relief > 0.0:
                        if iter_idx % relief_every == 0:
                            dq_relief = z_lift_dq(solver, cmd_q_R[:7], z_relief)
                        cmd_q_R = cmd_q_R.copy()
                        cmd_q_R[:7] = cmd_q_R[:7] + dq_relief
                elif z_relief != 0.0:
                    z_relief = 0.0
                    dq_relief = np.zeros(7)

            command_batch = {
                # the inference worker uses this to freeze episode-constant features
                # (e.g. the wipe task's ink keypoints) for the duration of a rollout
                "eval_controller:rollout_active": 1.0 if eval_state == "evaluating" else 0.0,
                f"{follower_right}_mit_command:timestamp": desired_timestamp,
                f"{follower_right}_mit_command:kp": cmd_kp_R,
                f"{follower_right}_mit_command:kd": cmd_kd_R,
                f"{follower_right}_mit_command:q": cmd_q_R,
                f"{follower_right}_mit_command:dq": cmd_dq_R,
                f"{follower_right}_mit_command:tau": desired_tau_R,
            }
            # publish the interaction torque the policy reads as obs (measured - gravity-comp ff),
            # same definition as tau_interaction_publisher.py / training. Folded in here so the
            # right-only eval needs no separate publisher (which would block waiting on the left arm).
            if tau_actual_R is not None:
                command_batch["follower_controller_state:tau_interaction_R"] = tau_actual_R - desired_tau_R
                command_batch["follower_controller_state:timestamp"] = desired_timestamp
            command_batch_maxlen = {k: int(frequency_hz * 10) for k in command_batch}
            client.stream_add_batch(command_batch, command_batch_maxlen)

            # ~1 Hz rollout diagnostic: is the model chunk fresh, and are we commanding motion?
            if eval_state == "evaluating" and iter_idx % int(frequency_hz) == 0:
                age = time.time() - float(model_action_timestamp[0])
                print(f"[rollout] model chunk age={age:.2f}s | commanded delta from actual="
                      f"{np.rad2deg(np.abs(cmd_q_R - qpos_R)).max():.1f} deg "
                      f"(if this stays ~0, the policy is predicting a near-static chunk)"
                      + (f" | press-relief lift={z_relief*100:.1f} cm" if press_relief else ""))

            reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=name)
            iter_idx += 1
            if iter_idx == 1:
                print(f"{name} ready -- place the arm on the table, press `h` to home, ENTER to roll out.")


if __name__ == "__main__":
    typer.run(main)
