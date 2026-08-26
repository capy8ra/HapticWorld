"""ACT policy for the world_ft box-flip task (OpenArm, single head camera).

Adapted from nova/learning/mobile_openarm_graphics_card/model.py:
  * single camera (`camera_head`).
  * `arm_names` + `use_torque` define the dims:
        obs_dim    = n_arms * (8 + 8*use_torque)   # per arm: actual_angle (+ tau_interaction)
        action_dim = n_arms * 8                     # per arm: commanded q
  * default is right-arm only ("right",) with torque -> obs_dim 16, action_dim 8.
"""
import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from nova.tools.helpers import to_cuda
from nova.learning.model.act.detr.main import build_ACT_model_and_optimizer


class ImageNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), requires_grad=False)

    def normalize(self, x):
        return (x - self.mean) / self.std


class GaussianNormalizer(nn.Module):
    def __init__(self, mu: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("std", std)

    def action_dim(self):
        return self.mu.shape[0]

    def normalize(self, x):
        return (x - self.mu) / self.std

    def denormalize(self, x):
        return x * self.std + self.mu


class ACT_Abs(nn.Module):

    def __init__(self, model, chunk_size, obs_dim, n_obs_steps, action_dim, hidden_dim, arm_names):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.arm_names = list(arm_names)
        self.img_normalizer = ImageNormalizer()
        self.state_normalizer = GaussianNormalizer(
            torch.zeros(obs_dim, dtype=torch.float32), torch.ones(obs_dim, dtype=torch.float32))
        self.action_normalizer = GaussianNormalizer(
            torch.zeros(action_dim, dtype=torch.float32), torch.ones(action_dim, dtype=torch.float32))
        self.action_dim = action_dim
        self.obs_lowdim_map = nn.Linear(n_obs_steps * obs_dim, hidden_dim)
        self.action_map = nn.Linear(hidden_dim, action_dim)

    def reset(self):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build_model_and_optimizer(cls, n_obs_steps, action_chunk_size, seed,
                                  arm_names=("right",), use_angle=True, use_torque=True,
                                  extra_obs_dim=0):
        n_arms = len(arm_names)
        # per arm: joint angle (8) if use_angle + torque (8) if use_torque -- must match dataset
        # obs; extra_obs_dim = appended task features (e.g. 2k ink keypoints, wipe task)
        obs_dim = n_arms * ((8 if use_angle else 0) + (8 if use_torque else 0)) + extra_obs_dim
        action_dim = n_arms * 8
        hidden_dim = 256
        camera_names = [f"camera_head_{i}" for i in range(n_obs_steps)]
        config = dict(
            seed=seed,
            camera_names=camera_names,
            num_queries=action_chunk_size,
            state_dim=hidden_dim,
        )
        model, optimizer = build_ACT_model_and_optimizer(config)
        return cls(
            model,
            chunk_size=action_chunk_size,
            obs_dim=obs_dim,
            n_obs_steps=n_obs_steps,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            arm_names=arm_names,
        ), optimizer

    def get_obs(cls, batch_data):
        img = batch_data["obs"]["camera_head"]  # (B, n_obs_steps, C, H, W)
        qpos = batch_data["obs"]["joint_state"]  # (B, n_obs_steps, obs_dim)
        return img, qpos

    def get_action(cls, batch_data):
        return batch_data["action"]  # (B, action_chunk_size, action_dim)

    def forward(self, img, qpos):
        a_hat, _, (_, _) = self.model(qpos, img)
        return a_hat

    @torch.no_grad()
    def compute_normalizer(self, dataloader, min_std=1e-2):
        # min_std floors the per-dim std. A static arm has ~0 joint-angle std; without a
        # floor the normalizer would divide those dims by ~1e-6 and blow tiny sensor noise
        # up into the model. (For right-arm-only this mainly guards the gripper joint.)
        print("Computing state and action normalizers...")
        all_lowdim_obs = []
        all_actions = []
        for batch_data in tqdm.tqdm(dataloader):
            batch_data = to_cuda(batch_data, non_blocking=True)
            img, qpos = self.get_obs(batch_data)
            actions = self.get_action(batch_data)
            B, TO, O = qpos.shape
            B, TA, A = actions.shape
            all_lowdim_obs.append(qpos)
            all_actions.append(actions)
        all_lowdim_obs = torch.cat(all_lowdim_obs, dim=0).reshape(-1, O)
        all_actions = torch.cat(all_actions, dim=0).reshape(-1, A)
        obs_mu = torch.mean(all_lowdim_obs, dim=0)
        obs_std = torch.clamp(torch.std(all_lowdim_obs, dim=0), min=min_std)
        action_mu = torch.mean(all_actions, dim=0)
        action_std = torch.clamp(torch.std(all_actions, dim=0), min=min_std)
        print("Computed obs mean:", obs_mu)
        print("Computed obs std:", obs_std)
        print("Computed action mean:", action_mu)
        print("Computed action std:", action_std)
        self.state_normalizer = GaussianNormalizer(obs_mu, obs_std)
        self.action_normalizer = GaussianNormalizer(action_mu, action_std)

    def compute_loss(self, batch_data):
        img, qpos = self.get_obs(batch_data)
        actions = self.get_action(batch_data)

        normalized_img = self.img_normalizer.normalize(img)
        normalized_qpos = self.state_normalizer.normalize(qpos)
        normalized_qpos = normalized_qpos.view(normalized_qpos.shape[0], -1)
        normalized_actions = self.action_normalizer.normalize(actions)

        obs_lowdim = self.obs_lowdim_map(normalized_qpos)
        a_hat_premap = self(normalized_img, obs_lowdim)
        a_hat = self.action_map(a_hat_premap)

        loss = F.l1_loss(a_hat, normalized_actions)
        stats_dict = dict(loss=loss.item())
        return loss, stats_dict

    @torch.no_grad()
    def predict_action(self, batch_data):
        img, qpos = self.get_obs(dict(obs=batch_data))
        normalized_img = self.img_normalizer.normalize(img)
        normalized_qpos = self.state_normalizer.normalize(qpos)
        normalized_qpos = normalized_qpos.view(normalized_qpos.shape[0], -1)
        obs_lowdim = self.obs_lowdim_map(normalized_qpos)
        a_hat_premap = self(normalized_img, obs_lowdim)
        a_hat = self.action_map(a_hat_premap)
        a_hat = self.action_normalizer.denormalize(a_hat)
        out = dict(action=a_hat)
        # split the action back into per-arm commanded-q chunks
        for i, arm in enumerate(self.arm_names):
            out[f"openarm_{arm}_action"] = a_hat[..., i * 8:(i + 1) * 8]
        return out


if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    act_model, optimizer = ACT_Abs.build_model_and_optimizer(
        n_obs_steps=1, action_chunk_size=20, seed=0, arm_names=("right",), use_torque=True)
    parameter_count = count_parameters(act_model)
    print("ACT_Abs model parameters:", parameter_count)
    print("ACT_Abs model parameters (M):", parameter_count / 1_000_000)
