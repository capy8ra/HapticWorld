import cv2
import numpy as np


def fisheye_center_crop(img, input_shape, square_crop_size, square_crop_shift=(0, 0)):
    H, W = input_shape
    center_y, center_x = H // 2 + square_crop_shift[1], W // 2 + square_crop_shift[0]
    half_crop_size = square_crop_size // 2
    crop_img = img[..., center_y - half_crop_size : center_y + half_crop_size, center_x - half_crop_size : center_x + half_crop_size, :]
    return crop_img

def jieruiweitong_fisheye_center_crop(img):
    assert img.shape[-3] == 480 and img.shape[-2] == 640, "Input image must be of shape (480, 640, C)"
    # for jieruiweitong's fisheye camera, the center crop should be shifted downwards by 40 pixels to better capture the workspace
    return fisheye_center_crop(img, input_shape=(480, 640), square_crop_size=350, square_crop_shift=(6, 8))

if __name__ == "__main__":
    import time
    from nova.learning.xarm.xarm_dataset import XArmDataset  # (not shipped here)
    action_timestamp_offsets = np.linspace(0, 1, 16)
    
    dataset = XArmDataset(
        data_root="data/2026_04_15_luqi_xarm",
        action_timestamp_offsets=action_timestamp_offsets
    )
    
    data_loading_times = []
    for i in range(len(dataset)):
        t0 = time.perf_counter()
        data = dataset[i]
        t1 = time.perf_counter()
        data_loading_times.append(t1 - t0)
        
        image = data["obs_camera_top"][0].transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        cv2.imshow("obs_camera_top", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # center_crop_img = jieruiweitong_fisheye_center_crop(image)
        # cv2.imshow("center_crop_img", cv2.cvtColor(center_crop_img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        for key, value in data.items():
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        
        print(f"{i}/{len(dataset)} Data loading time: {t1 - t0:.4f} seconds")

