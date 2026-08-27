#!/usr/bin/env python
"""Step-1 replacement: drive the world-model worker from a recorded episode instead of
the real leader arm.

This publishes a collected episode's per-frame action to
``leader_right_state:actual_angle_rad`` at the 10 Hz training cadence, so the UNCHANGED
``wm_follower_worker.py`` (step 2) consumes it exactly as if a human were moving the
leader. (The recorded ``action`` is ``follower_right_mit_command:q``, which in this stack
is a direct copy of the leader's measured angle, so replaying it as the leader angle is
faithful.)

It waits for the worker: on start it watches redis and only begins issuing actions once
the worker (step 2) starts reading, so playback always begins at frame 0 no matter which
you launch first.

Run (needs redis on 6379):
  # step 2 (unchanged): the worker
  python scripts/inference/wm_follower_worker.py \
    --ckpt outputs/world_ft_stage_2/<ts>/checkpoints/last.ckpt \
    --seed-episode data/world_ft_eval_0702b/val/episode_0.hdf5
  # step 1 (this): the recorded "leader"
  python scripts/inference/replay_leader_to_redis.py \
    --episode data/world_ft_eval_0702b/val/episode_0.hdf5
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "robot")))  # nova package
try:
    from nova.redis_client import RedisClient
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Could not import nova.redis_client -- is the HapticWorld repo layout intact "
        "(robot/nova next to worldsim/)?\n"
        f"(original error: {e})"
    )


def _worker_read_count(client: RedisClient) -> int:
    """How many XREVRANGE calls redis has served — the worker's per-step read. In this
    sim-only setup nothing else issues XREVRANGE, so a rising count == the worker polling."""
    try:
        cs = client.redis_client.info("commandstats")
    except Exception:
        return 0
    return int(cs.get("cmdstat_xrevrange", {}).get("calls", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="converted hdf5 whose action[t] is replayed")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--leader-right", default="leader_right", help="must match the worker's --leader-right")
    ap.add_argument("--hz", type=float, default=10.0, help="playback frame rate = training cadence (10)")
    ap.add_argument("--publish-hz", type=float, default=100.0, help="redis publish rate (>= --hz)")
    ap.add_argument("--loop", action="store_true", help="restart from frame 0 at the end instead of stopping")
    ap.add_argument("--hold-last-s", type=float, default=3.0, help="keep publishing the last frame this long after the end")
    ap.add_argument("--no-wait", action="store_true", help="start immediately instead of waiting for the worker")
    ap.add_argument("--wait-timeout-s", type=float, default=120.0, help="give up waiting for the worker after this")
    args = ap.parse_args()

    with h5py.File(args.episode, "r") as f:
        action = f["action"][()].astype(np.float64)  # (T, 8), same dtype the real controller emits
    T = action.shape[0]
    key = f"{args.leader_right}_state:actual_angle_rad"
    client = RedisClient(host=args.redis_host, port=args.redis_port)
    print(f"[replay] {args.episode}  T={T} (~{T/args.hz:.1f}s @ {args.hz:.0f}Hz)  ->  {key}")

    if not args.no_wait:
        baseline = _worker_read_count(client)
        print("[replay] waiting for the worker (step 2) to start reading ...")
        t0 = time.time()
        while _worker_read_count(client) <= baseline + 2:
            if time.time() - t0 > args.wait_timeout_s:
                print("[replay] wait timed out — starting anyway")
                break
            time.sleep(0.05)
        else:
            print("[replay] worker detected — starting playback from frame 0")

    dt = 1.0 / args.publish_hz
    t_start = time.time()
    last_print = -1.0
    while True:
        now = time.time()
        elapsed = now - t_start
        idx = int(elapsed * args.hz)
        if idx >= T:
            if args.loop:
                t_start = now
                idx = 0
            else:
                idx = T - 1
                if elapsed > T / args.hz + args.hold_last_s:
                    break
        client.stream_add_batch({key: action[idx]}, {key: 1})

        if now - last_print >= 1.0:
            print(f"[replay] t={elapsed:5.1f}s  frame {idx:4d}/{T-1}")
            last_print = now

        time.sleep(max(0.0, dt - (time.time() - now)))
    print("[replay] done")


if __name__ == "__main__":
    main()
