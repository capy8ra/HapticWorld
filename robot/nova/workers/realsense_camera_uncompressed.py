import time
import numpy as np

import cv2
import typer
import pyrealsense2 as rs

from nova.redis_client import RedisClient

def main(
    redis_host: str = "localhost",
    redis_port: int = 6380,
    stream_key: str = "realsense",
    stream_max_duration_s: int = 30,
    serial: str = "324422301217",
    vis: bool = False
):
    
    client = RedisClient(host=redis_host, port=redis_port)
    
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    stream_maxlen = int(stream_max_duration_s * 30)  # maxlen in number of frames, assuming 30 fps
    
    print(f"[{serial}] warming up...")
    for _ in range(30):
        pipeline.wait_for_frames()

    iter_idx = 0
    frame = None
    last_recv = None
    while True:
        
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        t_recv = time.time()  # Litian: is this accurate enough?
        
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        img = np.asanyarray(color_frame.get_data())
        if img.max() == 0:
            continue
        
        output_batch =        {f"{stream_key}:timestamp": t_recv,        f"{stream_key}:image": img          }
        output_batch_maxlen = {f"{stream_key}:timestamp": stream_maxlen, f"{stream_key}:image": stream_maxlen}
        client.stream_add_batch(output_batch, output_batch_maxlen)
        
        # print stats
        if iter_idx % 10 == 0 and last_recv is not None:
            print(t_recv, "{:.2f} FPS".format(1 / (t_recv - last_recv)), f"{stream_key}:image", "Len:", client.stream_get_len(f"{stream_key}:image"))
        last_recv = t_recv
        
        # visualize image
        if vis:
            cv2.imshow(f"{stream_key}:image", img)
            cv2.pollKey()

        if iter_idx == 0:
            print("camera ready")
        
        iter_idx += 1


if __name__ == "__main__":
    typer.run(main)