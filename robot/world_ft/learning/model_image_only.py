"""State-free / image-only ACT policy for world_ft box-flip.

Same ACT backbone as model.py but with NO proprioceptive (joint state) input: the DETR decoder is
built with use_proprio=False (the proprio token is dropped), and this wrapper carries no state
normalizer / obs_lowdim_map. Observation is the head camera only; target is the commanded q chunk.

Works on data with no follower state at all (the world-simulator sessions), and on the real data
when you simply don't load the state (dataset image_only=True).
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

    def normalize(self, x):
        return (x - self.mu) / self.std

    def denormalize(self, x):
        return x * self.std + self.mu


class ACT_ImageOnly(nn.Module):

    def __init__(self, model, chunk_size, action_dim, hidden_dim, arm_names):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.arm_names = list(arm_names)
        self.img_normalizer = ImageNormalizer()
        self.action_normalizer = GaussianNormalizer(
            torch.zeros(action_dim, dtype=torch.float32), torch.ones(action_dim, dtype=torch.float32))
        self.action_dim = action_dim
        # no state_normalizer, no obs_lowdim_map -- this policy sees only images
        self.action_map = nn.Linear(hidden_dim, action_dim)

    def reset(self):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build_model_and_optimizer(cls, n_obs_steps, action_chunk_size, seed, arm_names=("right",)):
        action_dim = len(arm_names) * 8
        hidden_dim = 256
        camera_names = [f"camera_head_{i}" for i in range(n_obs_steps)]
        config = dict(
            seed=seed,
            camera_names=camera_names,
            num_queries=action_chunk_size,
            state_dim=hidden_dim,
            use_proprio=False,          # <- drop the proprio token: state-free ACT
        )
        model, optimizer = build_ACT_model_and_optimizer(config)
        return cls(
            model,
            chunk_size=action_chunk_size,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            arm_names=arm_names,
        ), optimizer

    def get_obs(cls, batch_data):
        return batch_data["obs"]["camera_head"]  # (B, n_obs_steps, C, H, W)

    def get_action(cls, batch_data):
        return batch_data["action"]  # (B, action_chunk_size, action_dim)

    def forward(self, img):
        a_hat, _, (_, _) = self.model(None, img)   # qpos=None -> image-only
        return a_hat

    @torch.no_grad()
    def compute_normalizer(self, dataloader, min_std=1e-2):
        print("Computing action normalizer (image-only: no state)...")
        all_actions = []
        for batch_data in tqdm.tqdm(dataloader):
            batch_data = to_cuda(batch_data, non_blocking=True)
            actions = self.get_action(batch_data)
            B, TA, A = actions.shape
            all_actions.append(actions)
        all_actions = torch.cat(all_actions, dim=0).reshape(-1, A)
        action_mu = torch.mean(all_actions, dim=0)
        action_std = torch.clamp(torch.std(all_actions, dim=0), min=min_std)
        print("Computed action mean:", action_mu)
        print("Computed action std:", action_std)
        self.action_normalizer = GaussianNormalizer(action_mu, action_std)

    def compute_loss(self, batch_data):
        img = self.get_obs(batch_data)
        actions = self.get_action(batch_data)

        normalized_img = self.img_normalizer.normalize(img)
        normalized_actions = self.action_normalizer.normalize(actions)

        a_hat_premap = self(normalized_img)
        a_hat = self.action_map(a_hat_premap)

        loss = F.l1_loss(a_hat, normalized_actions)
        stats_dict = dict(loss=loss.item())
        return loss, stats_dict

    @torch.no_grad()
    def predict_action(self, batch_data):
        img = self.get_obs(dict(obs=batch_data))
        normalized_img = self.img_normalizer.normalize(img)
        a_hat_premap = self(normalized_img)
        a_hat = self.action_map(a_hat_premap)
        a_hat = self.action_normalizer.denormalize(a_hat)
        out = dict(action=a_hat)
        for i, arm in enumerate(self.arm_names):
            out[f"openarm_{arm}_action"] = a_hat[..., i * 8:(i + 1) * 8]
        return out


if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    act_model, optimizer = ACT_ImageOnly.build_model_and_optimizer(
        n_obs_steps=1, action_chunk_size=20, seed=0, arm_names=("right",))
    parameter_count = count_parameters(act_model)
    print("ACT_ImageOnly parameters:", parameter_count)
    print("ACT_ImageOnly parameters (M):", parameter_count / 1_000_000)
