"""Shared training core for the world_ft box-flip input-ablation matrix.

The entry scripts (train_<variant>_<data>.py) just build a CONFIG dict and call run_training().
Input variants are expressed by the config flags below:

    variant                 image_only  use_angle  use_torque   model
    image only              True        -          -            ACT_ImageOnly (proprio token dropped)
    image + torque          False       False      True         ACT_Abs (state = torque only)
    image + state + torque  False       True       True         ACT_Abs (state = angle + torque)

Torque source is auto-selected by the dataset: real data -> tau_interaction, world-model sim data ->
torque_pred_nm. Checkpoints go to model_ckpts/<exp_prefix>_<timestamp>/.
"""
import os
import sys
import time
import datetime
from time import strftime, localtime

import tqdm
import torch
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # robot/ (the nova package root)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)
from dataset import OpenArmDataset
from model import ACT_Abs
from model_image_only import ACT_ImageOnly
from nova.tools.helpers import to_cuda

torch.backends.cudnn.benchmark = True


def _datetime_str(ts):
    tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    return f"{strftime('%Y_%m_%d_%H_%M_%S', localtime(ts))}_{tz}"


def _save_checkpoint(path, info, model, optimizer):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
    with open(os.path.join(path, "info.yaml"), "w") as f:
        yaml.dump(info, f)


@torch.no_grad()
def _validate(act_model, val_dataloader):
    """Full pass over the held-out set: (mean loss, denormalized arm-joint chunk MAE deg)."""
    act_model.eval()
    losses, maes = [], []
    for batch_data in val_dataloader:
        batch_data = to_cuda(batch_data, non_blocking=True)
        loss, _ = act_model.compute_loss(batch_data)
        losses.append(loss.item())
        pred = act_model.predict_action(batch_data["obs"])["action"]
        gt = batch_data["action"]
        arm_dims = [i * 8 + j for i in range(pred.shape[-1] // 8) for j in range(7)]
        maes.append(torch.rad2deg((pred[..., arm_dims] - gt[..., arm_dims]).abs().mean()).item())
    act_model.train()
    return float(np.mean(losses)), float(np.mean(maes))


def run_training(config):
    config = dict(config)
    config["exp_dir"] = f"../data/model_ckpts/{config['exp_prefix']}_{_datetime_str(time.time())}"
    os.makedirs(config["exp_dir"], exist_ok=True)
    print(f"Experiment directory: {config['exp_dir']} created.")

    use_wandb = os.environ.get("WANDB_DISABLED", "0") != "1"
    if use_wandb:
        try:
            import wandb
            wandb.init(project=config["wandb_project"], name=os.path.basename(config["exp_dir"]), config=config)
        except Exception as e:  # noqa: BLE001
            print(f"wandb unavailable ({e}); continuing without logging.")
            use_wandb = False

    image_only = config["image_only"]
    dataset = OpenArmDataset(
        data_root=config["dataset_path"],
        camera_names=["camera_head"],
        arms=config["arms"],
        action_horizon=config["action_horizon"],
        action_duration=config["action_duration"],
        obs_horizon=config["obs_horizon"],
        obs_duration=config["obs_duration"],
        image_size=config["image_size"],
        image_only=image_only,
        use_angle=config.get("use_angle", True),
        use_torque=config.get("use_torque", True),
        blur_aug_res=config.get("blur_aug_res", 0),
        blur_aug_p=config.get("blur_aug_p", 0.5),
        ink_enhance=config.get("ink_enhance", False),
        ink_keypoints_k=config.get("ink_keypoints_k", 0),
        crop=config.get("crop", "center"),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        prefetch_factor=config["prefetch_factor"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )

    if image_only:
        act_model, optimizer = ACT_ImageOnly.build_model_and_optimizer(
            n_obs_steps=config["obs_horizon"], action_chunk_size=config["action_horizon"],
            seed=config["seed"], arm_names=config["arms"])
    else:
        act_model, optimizer = ACT_Abs.build_model_and_optimizer(
            n_obs_steps=config["obs_horizon"], action_chunk_size=config["action_horizon"],
            seed=config["seed"], arm_names=config["arms"],
            use_angle=config["use_angle"], use_torque=config["use_torque"],
            extra_obs_dim=2 * config.get("ink_keypoints_k", 0))
    act_model.compute_normalizer(dataloader)
    act_model.cuda()
    act_model.train()

    # optional in-loop validation on a held-out episode folder (val_dataset_path);
    # runs every val_every_n_steps (incl. step 0), logs to wandb + exp_dir/val_metrics.yaml
    val_dataloader = None
    val_history = []
    if config.get("val_dataset_path"):
        val_dataset = OpenArmDataset(
            data_root=config["val_dataset_path"],
            camera_names=["camera_head"],
            arms=config["arms"],
            action_horizon=config["action_horizon"],
            action_duration=config["action_duration"],
            obs_horizon=config["obs_horizon"],
            obs_duration=config["obs_duration"],
            image_size=config["image_size"],
            image_only=image_only,
            use_angle=config.get("use_angle", True),
            use_torque=config.get("use_torque", True),
            ink_enhance=config.get("ink_enhance", False),
            ink_keypoints_k=config.get("ink_keypoints_k", 0),
        )  # note: no blur augmentation on validation
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=False,
            num_workers=4, prefetch_factor=config["prefetch_factor"],
            persistent_workers=True, pin_memory=True)
        print(f"validation: {config['val_dataset_path']} "
              f"({len(val_dataset)} samples) every {config.get('val_every_n_steps', 2000)} steps")

    epoch_iter_idx = 0
    train_step_idx = 0
    last_epoch_loss = 0
    t_train_loop_start = time.time()
    while True:
        t_epoch_start = time.time()
        progress_bar = tqdm.tqdm(iterable=dataloader, total=len(dataloader),
                                 desc=f"Epoch {epoch_iter_idx} training")

        def _info():
            return dict(
                start_time=t_train_loop_start, time_now=time.time(),
                elapsed_time=time.time() - t_train_loop_start, epoch_idx=epoch_iter_idx,
                train_step_idx=train_step_idx, config=config, epoch_loss=last_epoch_loss,
                datetime_now=_datetime_str(time.time()))

        _save_checkpoint(os.path.join(config["exp_dir"], "ckpt_latest"), _info(), act_model, optimizer)

        epoch_loss = []
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_data = to_cuda(batch_data, non_blocking=True)
            if train_step_idx % config["save_every_n_steps"] == 0:
                _save_checkpoint(os.path.join(config["exp_dir"], f"ckpt_step_{train_step_idx}"),
                                 _info(), act_model, optimizer)
            if val_dataloader is not None and \
                    train_step_idx % config.get("val_every_n_steps", 2000) == 0:
                val_loss, val_mae = _validate(act_model, val_dataloader)
                print(f"\n[val @ step {train_step_idx}] loss={val_loss:.5f} | "
                      f"arm-joint chunk MAE={val_mae:.2f} deg")
                val_history.append(dict(step=train_step_idx, val_loss=val_loss,
                                        val_mae_deg=val_mae))
                with open(os.path.join(config["exp_dir"], "val_metrics.yaml"), "w") as f:
                    yaml.dump(val_history, f)
                if use_wandb:
                    wandb.log({"val/loss": val_loss, "val/mae_deg": val_mae},
                              step=train_step_idx)
            optimizer.zero_grad()
            loss, stats_dict = act_model.compute_loss(batch_data)
            loss.backward()
            optimizer.step()
            train_step_idx += 1
            epoch_loss.append(loss.item())
            if train_step_idx % 10 == 0:
                progress_bar.set_postfix(step=train_step_idx, loss=stats_dict["loss"])
                if use_wandb:
                    wandb.log({"train/step_loss": stats_dict["loss"]}, step=train_step_idx)

        epoch_iter_idx += 1
        last_epoch_loss = np.mean(epoch_loss).item()
        print(f"    experiment {config['exp_dir']} took "
              f"{(time.time() - t_epoch_start) / 3600:.3f} hours, epoch average loss: {last_epoch_loss:.6f}")
        if use_wandb:
            wandb.log({"train/epoch_loss": last_epoch_loss, "train/epoch": epoch_iter_idx},
                      step=train_step_idx)


# base config shared by all variants; entry scripts override the variant + dataset fields
BASE_CONFIG = dict(
    seed=42,
    arms=["right"],
    action_horizon=20,
    action_duration=2.0,
    obs_horizon=1,
    obs_duration=1.0,
    image_size=224,
    batch_size=128,
    num_workers=20,
    prefetch_factor=6,
    save_every_n_steps=5000,
)
