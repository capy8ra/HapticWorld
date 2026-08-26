"""Write tau_interaction_R into world-model sim episodes (data post-process).

Sim episodes record the WM's TOTAL predicted follower torque
(``follower_controller_state:torque_pred_nm`` — the stage-2 head was trained on measured
``actual_torque_nm``), while real episodes record the INTERACTION torque
(``tau_interaction_R = measured - gravity-comp feedforward``). This converts sim to the
real definition so torque-conditioned ACT trains and deploys on the same quantity:

    tau_interaction_R = torque_pred_nm - gravity(commanded q, zero velocity)

per frame, saved to ``low_dim_npys/follower_controller_state:tau_interaction_R.npy`` (the
shared ``follower_controller_state:timestamp.npy`` is already there). OpenArmDataset then
auto-selects tau_interaction over torque_pred_nm — no loader change needed.

Notes:
  * gravity is evaluated at the COMMANDED q (sim has no measured follower state) with zero
    velocity — pure gravity, no Coriolis (negligible at teleop speeds). Same solver + same
    definition as the real-data feedforward, so sim and real tau_interaction match.
  * RE-RUN after every new sim-collection batch, BEFORE training: LowDimDataset requires an
    identical stream set across episodes, so a mixed folder (some processed, some not)
    fails loudly at dataset init.
  * env: hapticworld (needs the MJCF pinocchio OpenArm1Solver), from the repo root:
        python world_ft/postprocess_sim_tau_interaction.py
"""
import os

import numpy as np
import typer

from nova.robot.openarm1.solver import OpenArm1Solver


def _gravity_R(solver, q7):
    """Gravity torque of the RIGHT arm at configuration q7 (7 joints), zero velocity.
    Same bimanual-model embedding as eval_controller.py (left arm held at 0)."""
    sim_q = np.concatenate([np.zeros(9), np.asarray(q7, float), [0.0, 0.0]])
    zeros = np.zeros_like(sim_q)
    solver.forward_kinematics(sim_q)
    tau = solver.get_gravity_compensation_torques(sim_q, zeros, zeros)
    return np.concatenate([tau[8:15], [0.0]])  # gripper: no gravity term


def main(
    data_dir: str = "../data/sim_training",
    force: bool = False,
):
    solver = OpenArm1Solver()
    episodes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d, "low_dim_npys")))
    n_done = n_skip = 0
    for ep in episodes:
        ld = os.path.join(data_dir, ep, "low_dim_npys")
        out_path = os.path.join(ld, "follower_controller_state:tau_interaction_R.npy")
        tq_path = os.path.join(ld, "follower_controller_state:torque_pred_nm.npy")
        if not os.path.isfile(tq_path):
            print(f"{ep}: no torque_pred_nm (not a sim episode?) -- skipping")
            continue
        if os.path.isfile(out_path) and not force:
            n_skip += 1
            continue

        q = np.load(os.path.join(ld, "follower_right_mit_command:q.npy"))          # (N, 8)
        q_ts = np.load(os.path.join(ld, "follower_right_mit_command:timestamp.npy"))
        tq = np.load(tq_path)                                                       # (M, 8)
        tq_ts = np.load(os.path.join(ld, "follower_controller_state:timestamp.npy"))
        if len(q_ts) == len(tq_ts) and np.allclose(q_ts, tq_ts):
            q_at_tq = q  # sim episodes: q and torque share one clock (per _ACTRecorder)
        else:
            q_at_tq = np.stack([np.interp(tq_ts, q_ts, q[:, j]) for j in range(q.shape[1])], axis=1)

        grav = np.stack([_gravity_R(solver, q_at_tq[i, :7]) for i in range(len(tq))])
        tau_int = (tq - grav).astype(np.float32)
        np.save(out_path, tau_int)
        n_done += 1
        print(f"{ep}: {len(tq)} frames | mean|torque_pred|={np.abs(tq).mean():.2f} "
              f"-> mean|tau_interaction|={np.abs(tau_int).mean():.2f} N.m")
    print(f"done: {n_done} processed, {n_skip} already had tau_interaction_R "
          f"(--force to redo), {len(episodes)} episodes total")


if __name__ == "__main__":
    typer.run(main)
