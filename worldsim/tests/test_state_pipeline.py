"""Unit / smoke tests for the state-conditioned stage-2 pipeline (CPU).

Run from the repo root:  python tests/test_state_pipeline.py

Covers:
  1. StateBranch shapes + residual zero-init prior (s_hat == s at init);
  2. full training_step forward/backward on a synthetic batch (hydra-composed
     config): six finite losses, gradients reach state branch, trunk torque
     head AND the dynamics U-Net; frozen encoder gets none;
  3. coupled_rollout output shapes/finiteness (state branch + diffusion
     feeding each other).
"""
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from omegaconf import OmegaConf  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

if not torch.cuda.is_available():
    class _FakeProps:
        major, minor = 0, 0

    torch.cuda.get_device_properties = lambda dev: _FakeProps()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))  # sys.path bootstrap: worldsim root
from interactive_world_sim.algorithms.latent_dynamics.models.state_branch import (  # noqa: E402
    StateBranch,
)


def test_state_branch() -> None:
    torch.manual_seed(0)
    B = 3
    sb = StateBranch()
    s = torch.randn(B, 8) * 0.5
    a = torch.randn(B, 8) * 0.5
    feat = torch.randn(B, 128)
    s1, h = sb.step(s, a, feat)
    assert s1.shape == (B, 8) and h.shape == (2, B, 256)
    # zero-init residual -> persistence prior, through the smooth bound
    expect = sb.out_clamp * torch.tanh(s / sb.out_clamp)
    assert torch.allclose(s1, expect, atol=1e-6), "zero-init should be soft-bounded persistence"
    in_range = s.abs() <= 1.0  # data lives in [-1, 1]
    assert (s1 - s)[in_range].abs().max() < 0.05, "soft bound must be near-identity in data range"
    # the smooth bound must keep gradients alive even far outside the range
    big = torch.full((B, 8), 10.0, requires_grad=True)
    out = sb.out_clamp * torch.tanh((big + 0) / sb.out_clamp)
    out.sum().backward()
    assert big.grad.abs().min() > 0, "bound gradient died outside range"
    print("[1/3] StateBranch shapes + soft-bounded persistence prior OK")


def build_model():
    from hydra import compose, initialize_config_dir

    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model_state import (  # noqa: E501
        LatentWorldModelState,
    )
    from interactive_world_sim.utils.normalizer import (
        LinearNormalizer,
        array_to_stats,
        get_image_range_normalizer,
        get_range_normalizer_from_stat,
    )
    import numpy as np

    with initialize_config_dir(
        config_dir=os.path.join(REPO, "configurations"), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=exp_latent_dyn_state",
                "algorithm=latent_world_model_state",
                "dataset=world_ft_state_dataset",
                "dataset.horizon=10",
                "dataset.obs_keys=[camera_1_color]",
                "~dataset.shape_meta.obs.camera_0_color",
                "algorithm.metrics=[]",
                "algorithm.training_stage=2",
                "algorithm.latent_dim=512",
                "+name=state_unit_test",
            ],
        )
    model = LatentWorldModelState(cfg.algorithm)
    model.log = lambda *a, **k: None

    normalizer = LinearNormalizer()
    rng = np.random.default_rng(0)
    normalizer["action"] = get_range_normalizer_from_stat(
        array_to_stats(rng.uniform(-1.5, 1.5, size=(1000, 8)).astype(np.float32))
    )
    # qpos normalizer with a degenerate gripper dim (range ~1e-3), mirroring the
    # dataset's protection: it must pass through un-scaled, not blow up.
    q = rng.uniform(-1.5, 1.5, size=(1000, 8)).astype(np.float32)
    q[:, 7] = rng.uniform(-0.011, -0.010, size=1000)
    stat = array_to_stats(q)
    normalizer["qpos"] = get_range_normalizer_from_stat(
        stat, ignore_dim=(stat["max"] - stat["min"]) < 0.05
    )
    assert float(normalizer["qpos"].params_dict["scale"][7]) == 1.0, (
        "degenerate gripper dim must not be rescaled"
    )
    normalizer["camera_1_color"] = get_image_range_normalizer()
    model.set_normalizer(normalizer)
    return model


def make_batch(B: int = 1, T: int = 10) -> dict:
    g = torch.Generator().manual_seed(0)
    a = torch.randn(B, T, 8, generator=g) * 0.3
    return {
        "obs": {"camera_1_color": torch.rand(B, T, 3, 128, 128, generator=g)},
        "action": a,
        "qpos_right": a + torch.randn(B, T, 8, generator=g) * 0.01,
        "joint_torque": torch.randn(B, T, 8, generator=g) * 3.0,
    }


def grad_sum(module: torch.nn.Module) -> float:
    return sum(
        float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None
    )


def test_optimizer(model) -> None:
    cfg = model.configure_optimizers()
    opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
    assert len(opt.param_groups) == 3, "expected dynamics + torque head + state branch groups"
    assert len(sched.base_lrs) == 3, "scheduler must cover ALL param groups (warmup!)"
    print("[opt] 3 param groups, scheduler covers all (state branch gets warmup)")


def test_training_step(model) -> None:
    model.train()
    out = model.training_step(make_batch(), 0)
    total = out["loss"]
    assert torch.isfinite(total), f"non-finite loss {total}"
    model.zero_grad(set_to_none=True)
    total.backward()
    g_state = grad_sum(model.state_branch)
    g_trunk_head = grad_sum(model.torque_head)
    g_unet = grad_sum(model.dynamics.module.mid_block)
    g_enc = grad_sum(model.encoder)
    assert g_state > 0, "no gradient in state branch"
    assert g_trunk_head > 0, "no gradient in trunk torque head"
    assert g_unet > 0, "no gradient in dynamics U-Net"
    assert g_enc == 0, "frozen encoder received gradient"
    print(f"[2/3] training_step OK: loss={float(total):.4f}  "
          f"grads: state={g_state:.2e} trunk={g_trunk_head:.2e} unet={g_unet:.2e} enc=0")


def test_coupled_rollout(model) -> None:
    model.eval()
    B, T = 1, 14
    batch = make_batch(B, T)
    with torch.no_grad():
        obs = model.normalizer["camera_1_color"].normalize(
            batch["obs"]["camera_1_color"][:, 0]
        ).float()
        z0 = model.encoder_forward(obs)[:, None]
        out = model.coupled_rollout(z0, batch["qpos_right"][:, 0], batch["action"])
    C = model.num_latent_channel
    assert out["z_seq"].shape == (B, T, C, 32, 32)
    assert out["q"].shape == (B, T - 1, 8)
    assert torch.isfinite(out["z_seq"]).all() and torch.isfinite(out["q"]).all()
    print("[3/3] coupled_rollout OK:", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    torch.manual_seed(0)
    test_state_branch()
    model = build_model()
    test_optimizer(model)
    test_training_step(model)
    test_coupled_rollout(model)
    print("\nALL STATE PIPELINE TESTS PASSED")
