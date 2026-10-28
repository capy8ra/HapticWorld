"""Tiny gravity / IK helper for the in-simulator policy rollout (JSON lines over stdio).

The WM policy rollout (worldsim/scripts/inference/rollout_policy_in_wm.py) runs in the
same process. This server runs in
the hapticworld env (real pinocchio + MJCF OpenArm1Solver) and is spawned as a subprocess by the
rollout script. Protocol: one JSON object per line on stdin -> one JSON reply per line.

    {"op": "ping"}                                  -> {"ok": true}
    {"op": "gravity", "q7": [...]}                  -> {"tau8": [...]}   zero-velocity gravity,
                                                       right arm, 8-d (gripper 0) -- same math as
                                                       postprocess_sim_tau_interaction.py
    {"op": "ik_offset", "q7": [...], "dxyz": [...]} -> {"q7": [...]}     EE position offset via the
                                                       eval_controller DLS IK (orientation free)

Prints a {"ready": true} line once initialized. All non-protocol output goes to stderr.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root
from policy.robot_model.solver import OpenArm1Solver  # noqa: E402

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


def gravity8(solver, q7):
    """Zero-velocity gravity torque of the right arm at q7, 8-d (gripper 0)."""
    sim_q = np.concatenate([np.zeros(9), np.asarray(q7, float), [0.0, 0.0]])
    zeros = np.zeros_like(sim_q)
    solver.forward_kinematics(sim_q)
    tau = solver.get_gravity_compensation_torques(sim_q, zeros, zeros)
    return np.concatenate([tau[8:15], [0.0]])


def main():
    solver = OpenArm1Solver()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req["op"]
            if op == "ping":
                out = {"ok": True}
            elif op == "gravity":
                out = {"tau8": gravity8(solver, req["q7"]).tolist()}
            elif op == "ik_offset":
                q7 = offset_ee_xyz(solver, np.asarray(req["q7"], float),
                                   np.asarray(req["dxyz"], float))
                out = {"q7": q7.tolist()}
            else:
                out = {"error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001  -- report, keep serving
            out = {"error": str(e)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
