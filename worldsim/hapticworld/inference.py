"""Load a trained HapticWorld model and seed it from a scene for interactive / policy rollouts."""
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from yixuan_utilities.draw_utils import center_crop

from hapticworld.algorithms.latent_dynamics.latent_world_model_torque import LatentWorldModelTorque
from hapticworld.utils.ckpt_utils import load_run_config


def load_model(ckpt_path: str, device: str, stage3_ckpt: str | None = None) -> tuple[LatentWorldModelTorque, object]:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path} (need the run's .hydra/config.yaml)")
    # The training config uses custom OmegaConf resolvers (registered in main.py):
    #   ${torch:float} -> torch.float, ${eval:'...'} -> eval(...) with np available.
    # Register them here too, or resolving cfg.algorithm.dtype / num_views raises.
    if not OmegaConf.has_resolver("torch"):
        OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
    cfg = load_run_config(cfg_path)
    load_dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.algorithm.load_ae = None
    model = LatentWorldModelTorque.load_from_checkpoint(
        ckpt_path, cfg=cfg.algorithm, map_location=device,
        dtype=load_dtype, strict=False, weights_only=False,
    )
    model.dynamics = model.dynamics.to(load_dtype)

    # Stage 3 fine-tunes ONLY the decoder (to be robust to the dynamics' latent errors),
    # so graft its decoder over the stage-2 model. Encoder + dynamics + torque head stay
    # from the stage-2 ckpt; the latent space is shared (both froze stage-1's encoder).
    if stage3_ckpt is not None:
        sd = torch.load(stage3_ckpt, map_location=device, weights_only=False)
        sd = sd.get("state_dict", sd)
        dec_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
        if not dec_sd:
            raise ValueError(f"No decoder.* weights in stage-3 ckpt {stage3_ckpt}")
        missing, unexpected = model.decoder.load_state_dict(dec_sd, strict=False)
        model.decoder = model.decoder.to(load_dtype)
        print(f"[wm] grafted stage-3 decoder from {stage3_ckpt} "
              f"({len(dec_sd)} tensors; missing={len(missing)} unexpected={len(unexpected)})")

    model.eval()
    return model, cfg


@torch.no_grad()
def seed_from_episode(
    model: LatentWorldModelTorque, seed_episode: str, res: int, device: str,
    amp_dtype: torch.dtype = torch.float32, amp_on: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (z_ctx (1,1,C,H,W), a_ctx (1,1,A) normalized, seed_gripper)."""
    obs_key = model.obs_keys[0]
    with h5py.File(seed_episode, "r") as f:
        frame0 = f["obs"]["images"][obs_key][0]      # (H,W,3) uint8
        action0 = f["action"][0].astype(np.float32)  # (8,) right q_des
    img = center_crop(frame0, res)
    img = cv2.resize(img, (res[1], res[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    img_t = model.normalizer[obs_key].normalize(img_t)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
        z0 = model.encoder_forward(img_t).float()[:, None]               # (1,1,C,H,W) fp32 state
    a0 = model.normalizer["action"].normalize(
        torch.from_numpy(action0)[None, None].to(device)
    ).clamp(-1.0, 1.0)                                                    # (1,1,A)
    return z0, a0, float(action0[7])
