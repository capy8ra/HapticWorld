# world_ft — model architecture reference

Complete description of every learned component in the world_ft system, written to support the
paper. Two model families are covered:

1. **The force-augmented interactive world simulator (WM)** — the IWS latent-diffusion world
   model plus the joint-torque head (the core contribution). Code: `HapticWorld/worldsim`.
2. **The downstream ACT policies** (± torque input) trained on real and WM-generated data.
   Code: `world_ft/learning/` in this repo.

All hyperparameters below are the **production values** (from
`checkpoints/wm/boxflip/stage2/.hydra/config.yaml` and `learning/train_core.py`), not
defaults. Measured evaluation numbers live in [WM_EVAL.md](WM_EVAL.md); operational commands in
[PIPELINE.md](PIPELINE.md); project map in [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

---

## 1. Relation to the IWS baseline

The baseline is the vision-only Interactive World Simulator (IWS, arXiv 2603.08546): a
three-stage latent world model (observation autoencoder → action-conditioned latent dynamics →
decoder robustness finetune) that predicts future camera frames from robot actions in real time.

**world_ft keeps the IWS backbone frame-for-frame identical and adds exactly one capability:
per-frame prediction of the right arm's 8 joint torques in raw N·m.** Concretely:

| | IWS (baseline) | world_ft |
|---|---|---|
| Stage 1: autoencoder | trained | **unchanged** |
| Stage 2: latent dynamics | trained (latents only) | **+ torque head, joint loss** `L_dyn + λ·L_torque` (λ=1) |
| Stage 3: decoder finetune | trained | **unchanged** |
| Outputs per step | next frame | next frame **+ 8-D joint torque (N·m)** |
| Uses at inference | video prediction | video **+ haptic feedback for in-sim teleop, torque labels for synthetic data, force-conditioned policy rollout** |

The addition is implemented as an isolated subclass (`LatentWorldModelTorque` extends the IWS
`LatentWorldModel`; stages 1/3 delegate to the parent untouched), so the baseline comparison in
the paper is a strict superset claim: everything IWS can do, plus force.

## 2. Data interface (what the WM consumes and predicts)

- **Observation** `o_t`: single head camera (`camera_1_color`), RGB **128×128**, 10 Hz.
  Normalized by a per-channel `LinearNormalizer` (`norm_scale` 6.0).
- **Action** `a_t`: right-arm commanded joint position `q_des` ∈ ℝ⁸ (7 joints + gripper),
  10 Hz, normalized (`LinearNormalizer`).
- **Torque target** `τ_t` ∈ ℝ⁸: `obs/joint_torque` = `actual_torque_nm` — the **total measured
  motor torque** (gravity/dynamics included), deliberately NOT the interaction torque and
  deliberately **not normalized** (raw N·m in and out; see §5.4).
- **Training window**: T = 10 consecutive frames (`dataset.horizon`), i.e. 1 s of interaction.
  Validation rolls 200-frame sequences.

## 3. Stage 1 — observation autoencoder

### 3.1 Encoder (704 parameters — deliberately minimal)

A five-conv stack (`LatentWorldModel._build_model`, latent_world_model.py):

```
Conv2d(3→4, 3×3)
[ SiLU → Conv2d(4→4, 3×3) → SiLU → Conv2d(4→4, 3×3, stride 2) ] × 2
```

maps 3×128×128 → **z ∈ ℝ^{4×32×32}** (`num_latent_channel=4`, `num_latent_downsample=2`).
After encoding, the 4 channels are **L2-normalized at every spatial location** (per view), so the
latent lives on a per-pixel unit sphere — this is the invariant the dynamics model predicts in,
and rollout latents are re-projected onto it after every generated frame.

The encoder is ~700 parameters total: virtually all representational capacity sits in the
*decoder*, and the latent is closer to a learned 4-channel downsampled "image" than a deep
feature. This matters for the torque story: a torque readout from z alone must re-derive contact
information from a very shallow code (what the `separate` head does), whereas the trunk head
reads the dynamics network's internal features (§5).

### 3.2 Decoder — conditional consistency-diffusion U-Net (27.6 M parameters)

`CMDecoder` (algorithms/models/cm_decoder.py) = a ControlNet pair:

- **Base U-Net** `CMControlledUnetModel` (19.7 M): standard ADM-style image diffusion U-Net over
  3×128×128; `model_channels=64`, `channel_mult=[1,2,3]`, 2 res-blocks per level, attention at
  feature resolutions {2,4,8} with 64 head channels, scale-shift (FiLM) norm, resblock
  up/downsampling, dropout 0.1.
- **ControlNet branch** `CMControlNet` (7.9 M): consumes the latent z (4×32×32), upsamples it
  ×4 back to 128×128 (`num_cond_upsamples=2`), and injects per-level control features into the
  base U-Net (control scales fixed at 1.0).

So decoding is *generative*: the image is sampled by denoising from pure Gaussian noise
**conditioned on z**, not regressed from z. Diffusion setup shared by decoder and dynamics:
sigmoid beta schedule, 1000 train timesteps, **v-prediction** objective, noise clipped to ±6σ,
uniform loss weighting.

**Consistency-trajectory (CTM) parameterization.** Both decoder and dynamics are trained as
trajectory-consistency models: the network gets (x_t, t, s) — a sample at noise level t and a
*target* level s < t — predicts v, converts to x̂₀, and outputs the anchored interpolation

```
G_θ(x_t, t, s) = (s/t)·x_t + (1 − s/t)·x̂₀        (cm_utils.py, CTM_calc_out)
```

i.e. a single network evaluation jumps from level t directly to level s (identity when s=t,
plain x̂₀ when s=0). Training draws (t,s) pairs with the `terminal_only` strategy: t = 999 and
s uniformly among the K-step grid points (K = `dec_infer_steps` = 3 for the decoder), with a
second forward leg s→0 whose target is the clean image, so the model is supervised on exactly
the jump sizes used at inference. **Inference decode = 3 CTM jumps (999→666→333→0) from pure
noise** — real-time at 10 Hz, vs 50+ DDIM steps for a standard diffusion decoder.

Stage-1 training: encoder + decoder jointly, MSE on the jump targets (the encoder gets its
gradient *through* the ControlNet conditioning — there is no reconstruction bottleneck loss).

## 4. Stage 2 — action-conditioned latent dynamics

### 4.1 Architecture: causal spatio-temporal U-Net (5.84 M parameters)

`CMLatentDynamics` (models/cm_latent_dynamics.py) operates on latent *sequences*
(B, 4, T, 32, 32). Production config: `dim=64`, `dim_mults=[1,2]`, `attn_heads=4`,
`attn_dim_head=128`, rotary temporal position embeddings, causal attention.

```
input z_seq (4ch) ── Conv3d(1×5×5) → 64ch ── causal TemporalAttention
  down level 0 (64ch):  [ResBlock ×2, spatial linear-attn, causal temporal attn]
  down level 1 (128ch): [ResBlock ×2, spatial full-attn,   causal temporal attn]
  ── mid_block (128ch): ResBlock → spatial attn → causal temporal attn → ResBlock   ← torque tap
  up level 1, up level 0 (skip-connected mirrors)
  cat with post-init features → ResBlock(128→64) → Conv3d(64→4, 1×1) = v̂
```

Three properties worth stating precisely in the paper:

- **No spatial resampling.** `Downsample`/`Upsample` are replaced by `nn.Identity`; the latent
  stays 32×32 through the whole network (it is already 16× downsampled). "Levels" differ only in
  channel width. The mid-block is therefore a *feature bottleneck* (widest channels, deepest
  layer), not a spatial one.
- **Time is handled only by attention.** All convolutions are (1×k×k) — purely spatial. Temporal
  mixing happens exclusively in the temporal-attention blocks, which are **causal** with rotary
  embeddings: frame t's features can only attend to frames ≤ t. This is what later lets a
  per-frame MLP read the mid-block as a causally-filtered state summary (§5.2).
- **Conditioning is per-frame FiLM.** Two embeddings are injected into every ResBlock as
  scale/shift on the group-norm output: (a) the noise-level embedding — sinusoidal→MLP for both
  the current level t and the target level s, concatenated (512-D); (b) the action embedding —
  q_des ∈ ℝ⁸ through an MLP 8→64→128→128→512. So the dynamics is conditioned on the *commanded*
  trajectory, the input the operator/policy actually controls.

### 4.2 Training: diffusion forcing with terminal-only noise

Per window (T=10, frozen encoder): per-frame noise levels are drawn as

- context frames 0..T−2: small random level t = s ∈ [1, 100) (`prev_frame_noise_scale=0.1`) —
  since t = s, the CTM output passes these frames through untouched and they contribute zero
  loss; they exist to make the model robust to slightly-corrupted context;
- terminal frame T−1: t = 999 (pure noise), s = 0 (`dyn_infer_steps=1` → single-jump training).

One forward of the whole noisy window (actions as FiLM conditions) yields the predicted clean
terminal latent; loss is MSE against the (shared-noise) target at level s. Effectively:
**"given 9 nearly-clean context latents and the 10-frame action sequence, generate latent 10 in
one consistency jump."**

Optimization (all stages): AdamW, wd 1e-4, β=(0.9, 0.999), grad-clip 1.0, bf16-mixed, 8 GPUs,
batch 16/GPU. Stage 2: lr 8e-5, 10 k linear warmup, ~1 M max steps. Stage-2 trainable
parameters = dynamics + torque head; encoder/decoder frozen.

### 4.3 Inference: autoregressive one-jump rollout

`dynamics_forward` (latent_world_model.py) generates one frame per control tick:

1. keep a sliding window of the last `n_tokens=10` latents (real context or own predictions);
2. append a pure-noise slot (clamped ±6σ);
3. context frames are assigned the **stabilization level** t = 15 (a diffusion-forcing trick:
   telling the model its context is "slightly noisy" even when clean measurably stabilizes long
   rollouts);
4. one CTM jump 999→0 for the new frame, conditioned on the action window;
5. re-project the new latent onto the per-pixel unit sphere (§3.1).

One dynamics forward (5.8 M params) + one 3-jump decode per tick sustains the 10 Hz interactive
loop on a single RTX 5080 (decode runs on a separate thread; the diffusion decoder is not
thread-safe — every decode holds `_DECODE_LOCK`).

## 5. The torque head (world_ft's core addition)

All code isolated in two files: `models/torque_predictor.py` (the heads) and
`latent_world_model_torque.py` (wiring, losses, validation). Design invariants, both variants:

- **Torque stays out of the diffusion.** τ is a deterministic physical readout, not a denoised
  variable — no noise is ever added to torque, and `CMLatentDynamics.forward`'s tensor-only
  return type is untouched (the `EinopsWrapper`/`CTM_calc_out` arithmetic never sees torque).
- **Dense supervision**: MSE over all T frames of the window (unlike the latent loss, which is
  effectively terminal-only).
- **Raw N·m end-to-end** (§5.4).

### 5.1 Variant A — `separate`: standalone re-encoding head (322 K params)

`TorquePredictor`: per-frame CNN over the latent (4→32→64→128 ch, stride-2 convs 32→4 px, global
average pool → 128-D) ∥ per-frame action MLP (8→64→64), concatenated into a 2-layer causal GRU
(hidden 128) → linear → τ ∈ ℝ⁸ per frame. Torque input sequence = [clean context latents
z₀..z_{T−2}, predicted terminal latent ẑ_{T−1}], so the terminal-frame torque gradient does flow
into the dynamics U-Net, but the head otherwise learns its own features from the 704-parameter
encoder's shallow code.

### 5.2 Variant B — `trunk_midblock`: shared-trunk readout (181 K params; **production**)

`TrunkTorqueHead` + a forward hook. The idea: the dynamics U-Net already computes, at its
mid-block bottleneck, a per-frame feature that is (a) the network's widest/deepest
representation, (b) action-conditioned via FiLM, and (c) **causally temporally filtered** (the
mid-block sits after temporal attention). That is exactly the state summary a torque estimator
needs — so read it instead of re-learning it.

**Mechanism** (latent_world_model_torque.py):

```python
# registration (build time) — zero changes to the U-Net itself
self.dynamics.module.mid_block.register_forward_hook(self._capture_trunk_feat)

def _capture_trunk_feat(self, _m, _in, out):        # fires on EVERY dynamics forward
    self._trunk_feat = out                          # (B, 128, T, 32, 32)

def _pool_trunk_feat(self, t_levels):
    feat = self._trunk_feat.mean(dim=(-1, -2))      # spatial GAP → (B, 128, T)
    feat = feat.permute(0, 2, 1)                    # (B, T, 128)
    return self.torque_head(feat, noise_level=t_levels.transpose(0, 1))
```

**Head**: per-frame MLP. Input = 128-D pooled mid-block feature ⊕ 64-D **noise-level
embedding** (`nn.Embedding(1000, 64)`, indexed by that frame's diffusion timestep t);
192 → 256 → 256 → 8 with SiLU. No recurrence — the temporal work is already done in the trunk.

**Why the noise conditioning** (`torque_noise_cond=true`): the trunk runs on *noisy* latents —
under terminal-only sampling the last frame is near-pure noise while context frames are
near-clean — so the same physical state produces systematically different mid-block features
depending on the frame's noise level. Feeding the level lets the MLP normalize across noise
regimes; train and inference use the same sampler, so the conditioning distribution matches.

**Training subtlety worth a footnote**: the hook fires on *every* dynamics forward, so when the
config runs a second consistency leg (`dyn_infer_steps>1`, pred_u), the features from the first
(pred_s) forward are snapshotted into a local before any other call and restored for the torque
loss — the torque readout is defined on the same forward pass that produces the latent
prediction. The reference is dropped right after the loss to release the autograd graph.

**What the gradient does**: `L = L_dyn + 1.0 · MSE(τ̂, τ)`, and the torque MSE backpropagates
through pooling into **the entire U-Net trunk** (mid-block, down path, action embedding). Unlike
variant A, the torque signal shapes the shared representation — the world model is trained to
carry contact information in the same features that predict the image. This is the mechanism
behind the paper's "genuinely shared visual-force representation" claim, and it is also the
practical winner: the trunk head outperformed the separate head, and the plain trunk beat the
later state-conditioned variant (kept as ablations; WM_EVAL §3c).

### 5.3 Torque at inference

Features only exist during a dynamics forward, so torque extraction *replays* one:
`_val_torque_pred` takes the current latent window + action window, draws training-style
terminal-only noise levels, adds noise, runs one dynamics forward (output discarded — it is run
for the hook side-effect), and pools the mid-block → τ̂ for every frame of the window. Long
validation sequences are processed in non-overlapping 10-frame windows and stitched.

- **Interactive teleop** (`wm_follower_worker.py`): every 10 Hz tick calls
  `_val_torque_pred(z_ctx, a_ctx)[:, -1]` → current-frame τ̂ (one extra 5.8 M-param forward,
  negligible), publishes raw N·m to redis; the leader arm renders `τ̂ − gravity(q_leader)` as
  force feedback (LPF + clamp + watchdog on the haptic side).
- **Open-loop eval** (`rollout_world_ft.py`): the same estimator teacher-forced on GT latents —
  the offline torque-RMSE metric and the online haptic signal come from one code path.
- **Policy-in-WM rollout** (`rollout_policy_in_wm.py`): torque-conditioned policies receive
  `τ̂ − gravity(q_cmd)` (gravity from the pinocchio bridge), matching the real robot's
  `tau_interaction` definition.

Calibration headline (details/dates in [WM_EVAL.md](WM_EVAL.md)): predicted contact torque
matches reality within ~10 % on the load-bearing joints (contact p95 ≈ 7–8 N·m), with free-space
baselines and 10 Hz temporal statistics also matching.

### 5.4 Why raw N·m (no normalizer on torque)

The head's output is directly a physical quantity: the haptic loop, the recorded synthetic-data
labels, and every plot consume it without carrying normalizer statistics; `√MSE` is directly an
RMSE in N·m; and the total-torque definition (rather than interaction torque) keeps the stage-2
target identical to what the motors actually measure — the interaction component is derived
downstream by explicitly subtracting the gravity model, in both real and sim data
(`postprocess_sim_tau_interaction.py`), so both domains train on the *same* quantity.

## 6. Stage 3 — decoder robustness finetune

Rollout latents drift off the encoder's manifold, so stage 3 finetunes **only the decoder**
(lr ×0.1) on latents perturbed with σ=0.02 Gaussian noise (encoder frozen, no_grad; same
consistency losses as stage 1). Encoder, dynamics, and torque head are untouched — stage-2 and
stage-3 checkpoints therefore compose freely **provided they share the same stage-1 lineage**
(compatibility rule: check `load_ae` in both configs). At deploy, the stage-3 decoder is grafted
over the stage-2 model (`load_torque_model`).

## 7. Abandoned / ablation variants (kept in-tree, not built upon)

- **State-conditioned stage 2** (`latent_world_model_state.py`, ckpts `stage2_state`+
  `stage3_state`): widens the FiLM condition to [q_des, q_measured] (16-D) and adds an
  autoregressive GRU `StateBranch` predicting realized joint state, BPTT through the unroll;
  trunk torque head inherited unchanged. **Underperformed the plain trunk+torque-head** — kept
  as the justified-simplicity ablation (WM_EVAL §3c).
- **Factored robot branch** (`models/robot_branch.py`): no learned torque head at all — a GRU
  predicts the tracking deviation δ = q_des − q, and torque is recovered *analytically* through
  the known 2 kHz impedance law τ = kp·δ + kd·(dq_des − dq) + τ_ff (verified R² ≈ 0.995 against
  measured torque on raw recordings). Design retained in-tree as the physics-grounded
  alternative; the learned-τ_ff surrogate it would need off-manifold was evaluated and rejected
  (cross-episode R² collapse).
- **Separate GRU head** (§5.1): the direct head-architecture ablation for the paper
  (optional stage-2 torque-loss ablation still open, PROJECT_SUMMARY "paper skeleton").

## 8. Downstream ACT policies (`world_ft/learning/`)

Standard ACT (action-chunking transformer, DETR-VAE backbone from
`nova/learning/model/act/detr/`), 31.47 M parameters, with two wrappers:

- **`ACT_Abs`** (model.py): image + low-dim state token. State vector per arm =
  [joint angle (8) if `use_angle`] ⊕ [torque (8) if `use_torque`]; the training matrix uses
  image-only (`i`), image+torque (`it`, 8-D state = torque only), and image+angle+torque
  (`its`, 16-D). The flattened state is mapped by a linear layer to one 256-D token.
- **`ACT_ImageOnly`** (model_image_only.py): the proprio token is removed entirely
  (`use_proprio=False`; the DETR decoder's extra-token stack shrinks from [latent, proprio] to
  [latent] — the one local modification to the vendored ACT transformer). Runs on WM sessions
  that have no follower state.

Backbone/transformer config: ResNet-18 (ImageNet-init, sine positional embedding, trained at
lr 1e-5), transformer d=256, 8 heads, ff 2048, 4 encoder / 6 decoder layers, 20 action queries.

**The CVAE branch of ACT is effectively disabled**: the wrappers call the model without the
action sequence, so the style latent z (dim 32) is zero at train and test time and no KL loss is
applied — training is pure L1 on the normalized 20-step action chunk. Worth one sentence in the
paper since "ACT" usually implies the CVAE.

I/O: single head camera, 640×480 → center-crop → 224×224, ImageNet normalization; `n_obs_steps=1`;
action = right-arm q_des chunk, 20 steps @ 10 Hz (2 s), executed receding-horizon
(0.5 s renew at deploy). State and action use per-dim Gaussian normalizers computed on the
training set (std floored at 1e-2 — a static gripper dim would otherwise amplify sensor noise
×10⁶). Torque input stream is auto-selected by the dataset: real data →
`tau_interaction_R` (measured − gravity feedforward); WM data → post-processed
`torque_pred_nm − gravity(q_cmd)` — the same physical quantity by construction (§5.4).
Optimization: AdamW lr 1e-4 (backbone 1e-5), wd 1e-4, batch 128. `_blur` variants add the
224→128→224 resample on 50 % of samples to make real-trained policies sharpness-invariant for
in-WM evaluation (the WM's renders are ~12× softer than the camera).

## 9. Parameter counts (measured by instantiation, production configs)

| component | params | trained in |
|---|---:|---|
| Encoder | **704** | stage 1 |
| Decoder (base U-Net + ControlNet) | 19.68 M + 7.92 M = **27.60 M** | stages 1, 3 |
| Dynamics U-Net (`CMLatentDynamics`) | **5.84 M** | stage 2 |
| Torque head, trunk_midblock (production) | **181 K** (64 K of it the noise-level table) | stage 2 |
| Torque head, separate (ablation) | 322 K | stage 2 |
| **Full WM at inference** (enc + dyn + head + dec) | **≈ 33.6 M** | |
| ACT policy (any variant) | **31.47 M** | phase E |

The torque capability costs **0.5 % of the WM's parameters** (181 K / 33.6 M) and one extra
dynamics forward per control tick.

## 10. Code map

| what | where (repo-relative) |
|---|---|
| Stage-1/2/3 base algorithm (IWS) | `worldsim/interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` |
| Torque subclass (stage-2 wiring, losses, val) | `.../latent_dynamics/latent_world_model_torque.py` |
| Both torque heads | `.../latent_dynamics/models/torque_predictor.py` |
| Dynamics U-Net (mid-block at L248) | `.../latent_dynamics/models/cm_latent_dynamics.py` |
| Decoder (ControlNet pair) | `worldsim/interactive_world_sim/algorithms/models/cm_decoder.py` |
| CTM scheduler (`CTM_calc_out`, noise sampling) | `worldsim/interactive_world_sim/utils/cm_utils.py` |
| State-conditioned ablation | `.../latent_dynamics/latent_world_model_state.py` + `models/state_branch.py` |
| Factored analytic-torque design | `.../latent_dynamics/models/robot_branch.py` |
| Production configs (per task) | `worldsim/world_ft_ckpts/<task>/.../.hydra/config.yaml` |
| Live in-sim teleop worker | `worldsim/scripts/inference/wm_follower_worker.py` |
| Open-loop WM eval | `worldsim/scripts/inference/rollout_world_ft.py` |
| Policy-in-WM rollout | `worldsim/scripts/inference/rollout_policy_in_wm.py` |
| ACT wrappers | `world_ft/learning/model.py`, `model_image_only.py` |
| ACT training loop + config | `world_ft/learning/train_core.py` |
| ACT dataset (torque auto-select, blur aug) | `world_ft/learning/dataset.py` |
| Vendored ACT/DETR (use_proprio mod) | `nova/learning/model/act/detr/` |
