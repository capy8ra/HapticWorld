import time

import cv2
import typer
from threadpoolctl import threadpool_limits


def main(
    dev_path: str = "/dev/video0",
    resolution: str = "640,480",
    fps: float = 30,
    use_yuv: bool = False,
    num_threads: int = 2,
    vis: bool = False,
    buffer_size: int = 2
):

    threadpool_limits(num_threads)
    cv2.setNumThreads(num_threads)

    w, h = [int(x) for x in resolution.split(",")]

    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if use_yuv:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))
    else:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    cap.set(cv2.CAP_PROP_FPS, fps)

    iter_idx = 0
    frame = None
    last_recv = None
    while True:
        ret, frame = cap.read(frame)
        t_recv = time.time()
        if not ret:
            raise RuntimeError(f"Failed to capture frame from {dev_path}. Make sure OBS is not open.")

        # https://github.com/opencv/opencv/blob/1db6a8a1f33786880706426313d72725f5f7bf99/modules/videoio/src/cap_v4l.cpp#L156-L159
        mt_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        t_cap = mt_cap - time.monotonic() + time.time()  # capture timestamp in epoch time

        if iter_idx % 10 == 0 and last_recv is not None:
            print("{:.2f} FPS".format(1 / (t_recv - last_recv)))
        last_recv = t_recv

        if vis:
            cv2.imshow(dev_path, frame)
            cv2.pollKey()

        if iter_idx == 0:
            print("camera ready")
        iter_idx += 1


if __name__ == "__main__":
    typer.run(main)
