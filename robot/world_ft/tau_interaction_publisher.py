import os
import sys
import time

import numpy as np
import typer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # robot/ (nova package root)
from nova.redis_client import RedisClient
from nova.tools.precise_sleep import FrequencyRegulator


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    follower_left: str = "follower_left",
    follower_right: str = "follower_right",
    output_name: str = "follower_controller",
    frequency_hz: float = 500,
):
    """Publish the follower interaction torque (tau_interaction) to Redis.

    tau_interaction = follower measured joint torque - follower MIT feedforward torque.
    In this pipeline the follower MIT feedforward `tau` is the gravity/Coriolis
    compensation written by teleop_follower_solver.py, so this matches the
    `tau_interaction = tau_actual - tau_gravity_comp` definition used by mobile_openarm.

    The follower controllers expose `actual_torque_nm` and the follower solver
    exposes `mit_command:tau`, but no existing world_ft process publishes the
    difference, so this worker computes and republishes it for recording.
    """
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=frequency_hz)

    iter_idx = 0
    while True:
        data_batch = client.stream_get_batch(
            {
                f"{follower_left}_state:actual_torque_nm": np.ndarray,
                f"{follower_left}_mit_command:tau": np.ndarray,
                f"{follower_right}_state:actual_torque_nm": np.ndarray,
                f"{follower_right}_mit_command:tau": np.ndarray,
            }
        )

        # wait until all upstream streams are alive before computing
        if any(value is None for value in data_batch.values()):
            reg.sleep()
            continue

        tau_interaction_L = (
            data_batch[f"{follower_left}_state:actual_torque_nm"]
            - data_batch[f"{follower_left}_mit_command:tau"]
        )
        tau_interaction_R = (
            data_batch[f"{follower_right}_state:actual_torque_nm"]
            - data_batch[f"{follower_right}_mit_command:tau"]
        )

        output_batch = {
            f"{output_name}_state:timestamp": time.time(),
            f"{output_name}_state:tau_interaction_L": tau_interaction_L,
            f"{output_name}_state:tau_interaction_R": tau_interaction_R,
        }
        output_batch_maxlen = {key: int(frequency_hz * 10) for key in output_batch.keys()}
        client.stream_add_batch(output_batch, output_batch_maxlen)

        reg.sleep()
        iter_idx += 1

        if iter_idx == 1:
            print("tau interaction publisher ready")


if __name__ == "__main__":
    typer.run(main)
