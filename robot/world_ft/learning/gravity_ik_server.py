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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_controller import offset_ee_xyz  # noqa: E402  (same DLS IK as --random-home)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # robot/ (nova package root)
from nova.robot.openarm1.solver import OpenArm1Solver  # noqa: E402


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
