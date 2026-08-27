"""Keyboard conductor for real-follower WM rollouts (teleop_wm_rollout_follower.yaml).

Replaces collect_data.py in the rollout rig: same ` / s / d keys, but instead of
driving recorders they drive the SIMULATOR (wm_follower_worker.py --conducted)
while wipe_solver keeps its exact collection behavior via collect_data:command.

  `  -> collect_data:command=start_recording  (solver: STITCH -> TELEOP, as in collection)
        wm_rollout:command=start              (worker: sim goes LIVE + records)
  s  -> collect_data:command=stop_recording   (solver: re-home flow, eraser kept)
        wm_rollout:command=stop               (worker: save rollout, reset to seed, re-arm)
  d  -> wm_rollout:command=reset ONLY         (worker: discard + reset to seed, re-arm;
        the robot keeps teleoping -- press ` for a fresh rollout, or s to re-home)

Never run together with collect_data.py (both would publish collect_data:command).
h / ENTER are handled by wipe_solver directly, unchanged.
"""
import os
import sys
import time

import typer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # robot/ (nova package root)
from nova.redis_client import RedisClient
from nova.tools.keycounter import KeystrokeCounter, KeyCode

MAXLEN = {"collect_data:command": 100, "collect_data:timestamp": 100,
          "wm_rollout:command": 100, "wm_rollout:timestamp": 100}


def main(redis_host: str = "localhost", redis_port: int = 6379):
    client = RedisClient(host=redis_host, port=redis_port)
    rolling = False

    def pub(solver_cmd, wm_cmd):
        now = time.time()
        batch = {}
        if solver_cmd:
            batch["collect_data:command"] = solver_cmd
            batch["collect_data:timestamp"] = now
        if wm_cmd:
            batch["wm_rollout:command"] = wm_cmd
            batch["wm_rollout:timestamp"] = now
        client.stream_add_batch(batch, MAXLEN)

    print("keys: ` = start rollout | s = stop + re-home | d = discard rollout "
          "(robot keeps teleoping) | h / ENTER belong to wipe_solver")
    print("wm rollout conductor ready")
    with KeystrokeCounter() as counter:
        while True:
            for ev in counter.get_press_events():
                if ev == KeyCode(char="`"):
                    if rolling:
                        print("[conductor] already rolling — s stops, d discards")
                    else:
                        rolling = True
                        pub("start_recording", "start")
                        print("[conductor] ` -> rollout START (solver stitches, sim live)")
                elif ev == KeyCode(char="s"):
                    if rolling:
                        rolling = False
                        pub("stop_recording", "stop")
                        print("[conductor] s -> rollout STOP + robot re-homes")
                    else:
                        # no rollout live (e.g. after d), but still let the solver
                        # re-home exactly as in collection
                        pub("stop_recording", None)
                        print("[conductor] s -> robot re-homes (no rollout was live)")
                elif ev == KeyCode(char="d"):
                    if rolling:
                        rolling = False
                        pub(None, "reset")
                        print("[conductor] d -> rollout DISCARDED (sim re-armed; robot "
                              "still teleoping — ` restarts, s re-homes)")
                    else:
                        print("[conductor] d ignored — nothing rolling")
            time.sleep(0.02)


if __name__ == "__main__":
    typer.run(main)
