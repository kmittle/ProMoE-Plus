# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProMoE-Plus implements **ProMoE** (ICLR 2026), a Mixture-of-Experts framework for scaling Diffusion Transformers (DiTs) on ImageNet class-conditional generation. The key contribution is a two-step router with explicit routing guidance: conditional routing (separating cond/uncond tokens) followed by prototypical routing (learnable cluster centers for semantic expert assignment), plus a routing contrastive loss for expert specialization.

## Common Commands

### Environment Setup
```bash
conda create -n promoe python=3.10 -y && conda activate promoe
pip install -r requirements.txt
```

### Training
```bash
# Standard ProMoE training
python train.py --config configs/004_ProMoE_L.yaml

# REPA-enabled training (aligns with frozen DINOv2 teacher)
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml

# MoS REPA training (Mixture-of-Softmaxes routing with REPA)
python train_with_MoS_repa.py --config configs/004_ProMoE_B_repa_MoS.yaml

# MAE-alignment / noise-expert training
python train_with_mae.py --config configs/004_ProMoE_B_group_align.yaml

# Offline/local pretrained weights (all training scripts + sample.py support --vae-path;
# train_with_repa.py and train_with_MoS_repa.py also support --repa-enc-path)
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml \
  --vae-path /path/to/sd-vae-ft-mse --repa-enc-path /path/to/dinov2_state_dict.pth
```

### Sampling
```bash
# Single GPU, default settings (500k checkpoint, 50K images, CFG 1.0/1.5)
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml

# Custom settings
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml \
  --step_list_for_sample 200000,300000 --guide_scale_list 1.0,1.5,4.0 --num_fid_samples 10000
```

### End-to-End Scripts
Scripts under `scripts/` run train + sample + eval in one go. Organized by experiment family:

| Directory | Variants |
|-----------|----------|
| `scripts/repa/` | REPA, REPA-Shared, REPA-Cond, Router, Router-Contra, Routed, Double-Share, Cross-Attention (global-pre/global-block/expert-local/proto), L/XL scale-up |
| `scripts/dynamic_repa/` | REPA-Dyna, Dyna-Select (r25/r75), Dyna-Scale, Dyna-Only |
| `scripts/MoS_repa/` | MoS, MoS Naive, MoS Naive Choice (block-range sweep, Sep, Blockwise, PerBlock, Fused, Shared-Align, RMSNorm, No-Coeff, proj_coeff sweep), Multi-Align (±dynamic), Cross-Attention MoS, L/XL scale-up |
| `scripts/hierar/` | Hierarchical, Hierarchical-Expert, NoPenalty, Expert-REPA-Dyna |
| `scripts/mae_align/` | MAE alignment, MAE alignment with projection |
| `scripts/noise_expert/` | Noise expert, Noise expert proj, EMA on noise/shared |
| `scripts/expert_contra/` | Expert contrastive output/param |

```bash
# Example: run a MoS experiment end-to-end
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_b3_5_train_sample_eval.sh
```

### VAE Latent Preprocessing (speeds up training)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python preprocess/preprocess_vae.py \
  --latent_save_root "/path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz"
```

### Evaluation (separate conda env with TensorFlow)
```bash
conda create -n promoe_eval python=3.9 -y && conda activate promoe_eval
cd evaluation && pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Run from inside evaluation/ directory
CUDA_VISIBLE_DEVICES=0 python run_eval.py /path/to/generated/images
# Pack PNGs to NPZ without running evaluator
python run_eval.py /path/to/generated/images --count 50000 --no-eval
```

## Architecture

### Configuration System
- `config.py`: Global defaults using EasyDict. Defines base model configs (`DiT_S_config` through `DiT_XL_config`) and MoE-specific configs (`DiffMoE_DiT_*`, `TCDiT_*`, `ECDiT_*`).
- `configs/*.yaml`: Per-experiment overrides deep-merged onto `config.py` defaults at runtime via `deep_update()` in `utils.py`.
- **Config merging flow**: ProMoE models reuse base DiT configs (e.g., `ProMoE_TC_L` maps to `DiT_L_config` in `model_dict`). The YAML adds `MoE_config` as a nested dict under the base config key (e.g., `DiT_L_config.MoE_config`), which `deep_update()` merges in. This means MoE parameters are not in `config.py` for ProMoE — they come entirely from YAML.
- The YAML filename (minus extension) becomes `custom_cfg_name`, which determines the output subdirectory: `outputs/{model_name}/{custom_cfg_name}/` containing `checkpoints/`, `training.log`, `sample.log`, `tensorboard/`, and `sample/step{N}/`.

### Model Registry
`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, and `train_with_mae.py` each define a `model_dict` mapping `model_name` strings to `(ModelClass, config_key)` pairs. `sample.py` merges all four dicts so it can sample from any model variant. Adding a new model requires an entry in the appropriate training script's `model_dict`. Note: `train.py` hosts most model families (base DiT, baselines, ProMoE-TC/EC, noise expert variants, expert contrastive); `train_with_MoS_repa.py` hosts MoS, MoS Naive, MoS Naive Choice (B/L/XL), MoS Naive Choice Sep, MoS Naive Choice Fused, MoS Naive Choice Blockwise, MoS Choice PerBlock, Multi-Align, and Cross-Attention variants (both standard REPA and MoS); `train_with_mae.py` only hosts group_align models.

### Model Hierarchy (in `models/`)
All model files follow the `models_*.py` naming convention. Key layers:

- **`modules.py`** — Shared building blocks: `Attention`, `PatchEmbed`, `TimestepEmbedder`, `LabelEmbedder`, `FinalLayer`, `MLP`/`Mlp`, `SwiGLU`, `MoeMLP`.
- **`models_DiT.py`** — Dense DiT baseline. `DiTBlock` uses AdaLN-Zero modulation (6-param per-sample conditioning from timestep+class). All ProMoE variants inherit this block structure.
- **Baselines**: `models_TCDiT.py` (Token-Choice MoE), `models_ECDiT.py` (Expert-Choice MoE), `models_DiffMoE.py` (capacity prediction).
- **`models_ProMoE_TC.py`** — Main proposed model. `SparseMoeBlock` implements two-step routing: (1) conditional routing separates uncond tokens (class=1000) to a dedicated expert, (2) prototypical routing assigns cond tokens via cosine similarity to learnable `cluster_centers`. Includes routing contrastive loss via `AddAuxiliaryLoss` autograd trick. `models_ProMoE_EC.py` is the Expert-Choice variant (recommended for DDPM).

**Variant families** (all extend ProMoE-TC):

| Family | Files | Training script | Key difference |
|--------|-------|----------------|----------------|
| REPA | `_repa.py`, `_repa_shared.py`, `_repa_cond.py` | `train_with_repa.py` | MLP projectors align DiT features with frozen DINOv2 teacher |
| Dynamic REPA | `_repa_dyna.py`, `_dyna_scale.py`, `_dyna_select.py`, `_dyna_only.py` | `train_with_repa.py` | Timestep-dependent REPA loss weighting/selection |
| Router REPA | `_repa_router.py`, `_repa_router_contra.py`, `_repa_routed.py`, `_repa_double_share.py` | `train_with_repa.py` | Alignment at router/prototype level instead of block output |
| MoS REPA | `_repa_MoS.py`, `_MoS_naive.py`, `_MoS_naive_choice.py`, `_MoS_naive_choice_.py` (sep), `_MoS_naive_choice_blockwise.py`, `_MoS_choice_per_block.py` | `train_with_MoS_repa.py` | `BlockRouter` selects which teacher blocks to align with per-token. Config ablation flags: `router_norm_type` (`"layernorm"` default / `"rmsnorm"`), `align_target` (`"block_output"` default / `"shared_expert"` — falls back to block output for dense blocks) |
| Fused MoS | `_MoS_naive_choice_fused.py` | `train_with_MoS_repa.py` | Fuses top-k teacher blocks via routing weights before alignment; shared Transformer+sigmoid `CoeffPredictor` (input: projected student features `z_proj` + conditioning) predicts per-token loss weight |
| Multi-Align | `_repa_multi_align.py` | `train_with_MoS_repa.py` | Per-token sigmoid coefficients from `AlignCoefficientPredictor`. Config ablation flag: `use_dynamic_coeff` (`True` default / `False` removes predictor, uses uniform weighting) |
| Hierarchical | `_hierar.py`, `_hierar_expert.py`, `_hierar_expert_repa_dyna.py` | `train.py` | Hierarchical routing structure |
| Noise Expert | `_noise_expert.py`, `_noise_expert_proj.py`, `_noise_expert_ema.py` | `train.py` | Dedicated noise-level expert; EMA variant calls `update_noise_expert_ema()` after each optimizer step |
| Expert Contrastive | `_expert_contra.py` | `train.py` | Pairwise L2 repulsion on expert outputs or params |
| Group Align | `_group_align.py`, `_group_align_proj.py` | `train_with_mae.py` | Group alignment without REPA |
| Cross-Attention | `_repa_cross_global_pre.py`, `_repa_cross_global_block.py`, `_repa_cross_expert_local.py`, `_repa_cross_proto.py` | `train_with_MoS_repa.py` | Inter-token attention-weighted REPA alignment at different positions (pre-MoE, block, expert-local, prototype) |
| Cross-Attention MoS | `_repa_MoS_naive_choice_cross_global_pre.py`, `..._block.py`, `..._expert_local.py`, `..._proto.py` | `train_with_MoS_repa.py` | MoS + cross-attention alignment (combines block router teacher selection with cross-alignment) |
| Ablations | `_sigmoid.py`, `_symmetric.py` | `train.py` | Routing gating variants |

**REPA model forward() behavior**: Returns `(pred, zs_proj)` during training (eval returns only `pred`). The `_repa_shared.py` variant aligns shared expert output specifically — requires `encoder_depth` to point to a MoE block.

### Auxiliary Loss Convention
Model `forward()` returns either a plain tensor (DiT) or a tuple for models with auxiliary losses:
- **DiffMoE**: Returns `(pred, "Capacity_Pred", layer_idx_list, ones_list, pred_c_list, loss_weight)`. Training loop computes BCEWithLogitsLoss for capacity prediction.
- **ProMoE**: Uses `AddAuxiliaryLoss` autograd function to inject contrastive loss gradients directly into the forward pass — returns a plain tensor but the auxiliary loss gradient flows through automatically.
- **ProMoE-REPA**: Returns `(pred, zs_proj)` during training. The training loop in `train_with_repa.py` computes `compute_repa_loss(teacher_z, zs_proj)` and adds it weighted by `proj_coeff`. Total loss = MSE + REPA loss * `proj_coeff` + routing contrastive loss (via autograd).
- **ProMoE-MoS-REPA**: Returns `(pred, mos_repa_loss)` during training, where `mos_repa_loss` is a scalar computed inside the model (weighted cosine similarity across selected teacher blocks). The training loop in `train_with_MoS_repa.py` multiplies by `proj_coeff` (default 0.5). Total loss = MSE + mos_repa_loss * `proj_coeff` + routing contrastive loss (via autograd). Note: `teacher_all_z` (all teacher block features) is passed to forward; the model selects which teacher blocks to align with via its router.
- **ProMoE-Multi-Align**: Returns `(pred, repa_loss)` during training. Similar to MoS-REPA but aligns with teacher last layer only; `AlignCoefficientPredictor` produces per-token sigmoid coefficients that weight the alignment loss. When `use_dynamic_coeff=False`, the predictor is removed and alignment uses uniform weighting (plain mean of negative cosine similarity). Also trained via `train_with_MoS_repa.py`.
- **ProMoE-Fused-MoS-REPA**: Returns `(pred, mos_repa_loss)` during training. Like MoS-REPA but fuses top-k teacher block features (weighted sum using routing weights, no re-normalization) before computing cosine similarity. A shared `CoeffPredictor` (Transformer + sigmoid, input = projected student features `z_proj` + conditioning `c`) predicts per-token loss weight. Trained via `train_with_MoS_repa.py`.
- **ProMoE-Cross-Attention**: Returns `(pred, cross_align_loss)` during training. The cross-alignment loss uses attention weights (global, block-level, expert-local, or prototype-based) to weight the cosine similarity between student projections and teacher features. Computed inside the model via `compute_cross_align_loss()`. Trained via `train_with_MoS_repa.py` with same `proj_coeff` weighting.

### REPA Module (`repa/` vs `REPA/`)
- `repa/` (lowercase) — ProMoE's REPA integration: encoder loading, loss computation, used by `train_with_repa.py`.
- `REPA/` (uppercase) — Separate standalone REPA subproject (original codebase). Treat changes there as scoped work independent from ProMoE.
- `repa/encoder.py` — Loads frozen DINOv2 teacher encoders (`dinov2-vit-{b,l,g}` and `dinov2reg-vit-{b,l,g}`). Downloads via torch.hub on first use, caches to `pretrained_ckpt/encoder/`. Handles positional embedding resampling for target resolution.
- `repa/loss.py` — `compute_repa_loss(z_teacher, z_student_list)`: negative cosine similarity between teacher patch features and projected student features, averaged across alignment points.
- `repa/encoder.py` also provides `extract_all_teacher_block_features()` (returns features from all intermediate blocks, used by MoS training) and `get_num_teacher_blocks()` (returns block count for a given encoder type).
- `train_with_repa.py` — Extended training loop that loads raw images alongside VAE latents, extracts teacher features with `extract_teacher_features()`, and adds REPA projection loss to the total loss.

### REPA Parameters — Two-Level `repa_config` Gotcha
YAML files have **two** `repa_config` blocks with different scopes — this is the most common source of config bugs:
- **`DiT_B_config.repa_config`** (nested under the model config key) — read by the model at init time. Controls projectors (`encoder_depth`, `z_dims`, `projector_dim`), router REPA settings, and MoS-specific knobs (`align_blocks`, `num_teacher_blocks`, `mos_top_k`).
- **Top-level `repa_config`** — read by the training loop. Controls `enc_type` (teacher encoder to load) and `proj_coeff` (REPA loss weight). `enc_type` must match between both levels.

For MoS variants, `num_teacher_blocks` is auto-injected by `train_with_MoS_repa.py` if not specified. See existing YAML configs for the full parameter set.

### Cross-Alignment Stability Constraints
Cross-alignment variants (`cross_global_pre`, `cross_global_block`, `cross_expert_local`, `cross_proto`, and their MoS counterparts) have two constraints any new variant must preserve:

1. **Clamp `cos_sim` to `[-1, 1]` after `F.normalize + torch.bmm`.** Under bf16 autocast, `rsqrt` and `matmul` precision can produce cosine similarities slightly outside `[-1, 1]`, which accumulates into loss spikes and eventual MSE divergence (observed in plans 04, 08 crashes). Every `compute_cross_align_loss` / `compute_cross_mos_repa_loss` in the 8 cross-alignment models applies `.clamp(-1.0, 1.0)` after `torch.bmm(z_proj_norm, teacher_norm.T)`.

2. **Detach the block output before feeding it to a block-wise weight-prediction module.** For `cross_global_block` and `cross_expert_local` variants (both standard and MoS), the attention module that predicts cross-alignment weights consumes the aligned DiT block's output `x`. Without `x.detach()`, two gradient paths leak into the same block: the projection path (which pushes features toward teacher) and the attention path (which pushes features to differentiate same-expert tokens for sharp attention). This creates gradient conflict and manifests as early MSE spikes (plan 03: step ~9890) or late MSE divergence (plan 02: step ~371k). The fix: call the attention module with `x.detach()` (e.g. `self.expert_local_attn(x.detach(), mask)`), keep the projection call on the original `x`. The attention module's internal parameters still receive gradient via `cross_align_loss`; only the gradient flow back into the DiT block is cut. `cross_global_pre` variants are exempt because they apply attention to the initial patch embedding (before any DiT block), and `cross_proto` variants are exempt because their weights come from MoE routing (`_proto_sim`) rather than a dedicated weight-prediction module.

See `collapse_smoking_test/crash_diagnosis_report.md` for the full investigation.

### TrainingMonitor (Crash Diagnosis Utility)
`utils.py` exposes a `TrainingMonitor` class that captures the precursor signals relevant to the cross-alignment crashes above (attention-row collapse, exploding projector features, runaway per-group grad norms, routing collapse, loss jumps). It is designed to be wired into a crashed model's re-run with minimal changes:

```python
from utils import TrainingMonitor
monitor = TrainingMonitor(model, logger=logger, log_every=cfg.log_interval,
                          enabled=(rank == 0), writer=tb_writer)
# inside the training loop, AFTER backward() + clip_grad_norm_, BEFORE zero_grad():
monitor.on_step(step, losses=logged_loss_dict)
```

Pass the existing `SummaryWriter` as `writer=` to mirror every stat to TensorBoard under the `monitor/{grad,attn,proj,coeff,router,cc,cross}/...` namespaces alongside the periodic text log. Omit it to log only to the logger.

Mechanics — all non-invasive, no model code changes required:
- Installs `forward_hook`s by **class name** on `ExpertLocalAttention` / `BlockAlignAttention` / `GlobalPreAttention` (attention maps), `CoeffPredictor` / `AlignCoefficientPredictor` (per-token sigmoid coefficients), and `BlockRouter` / `PerBlockRouter` / `AdaLNRouter` (router outputs — dispatched per class because the three have different output shapes and softmax conventions).
- Auto-detects any top-level `nn.ModuleList` whose attribute name ends in `projectors` — covers `projectors`, `mos_projectors`, `align_projectors`, `router_projectors`.
- Iterates `named_parameters()` each step for grad-norm stats grouped by param-name substring (attn_modules / projectors / block_router / coeff_predictor / cluster_centers / shared_expert / moe_experts / backbone). Frozen params (`requires_grad=False`, e.g. `pos_embed`, `noise_expert_ema` params) are skipped.
- `cluster_centers` is a parameter inside each `SparseMoeBlock` (NOT on the top-level DiT), so stats are aggregated via `named_parameters()` traversal, not a top-level attribute lookup.
- `ExpertLocalAttention` uses masked softmax + `nan_to_num(0)`, which produces fully-zero rows for uncond tokens (labels==1000) by design (~10% under CFG). The attention hook filters to "active rows" (`row_sum > 1e-4`) before computing min/max, and reports `inactive_frac` as a benign diagnostic instead of an alert.
- Dense DiT / non-MoE / non-REPA models degrade gracefully — no hooks are installed, only backbone grad norms are reported.
- Every stat path is wrapped in try/except so a monitoring bug cannot take the training run down.

### Key MoE Parameters (in YAML `MoE_config`)
Core parameters: `num_routed_experts` (typically 12), `top_k` (experts per token, default 1), `use_shared_expert`/`use_uncond_expert`, `interleave` (alternate MoE/dense layers).

Constraints to know:
- `routing_contrastive_lam` must be **0** for `repa_router` model. For `repa_router_contra`, it is the total budget shared between REPA alignment and contrastive via linear handoff over `router_loss_decay_steps`.
- `expert_contrastive_blocks` must all be MoE blocks (asserted at init).
- `noise_expert_ema` model: noise expert params are `requires_grad=False` and updated via EMA — excluded from optimizer.

### Training Pipeline (`train.py`)
- PyTorch DDP for multi-GPU distributed training via `mp.spawn`. The YAML `gpu_ids` field sets `CUDA_VISIBLE_DEVICES` and determines the DDP world size (`gpus_per_machine`).
- Training hyperparameters (`lr`, `total_train_batch_size`, `weight_decay`, `num_steps`) are per-YAML — `config.py` only has framework defaults like `max_grad_norm`, `betas`, `weighting_scheme`.
- Logit-normal timestep sampling (SD3-style) with Rectified Flow objective
- Mixed precision with bfloat16; gradient clipping at `max_grad_norm=0.5`
- EMA model maintained for stable generation
- Supports both raw image loading and pre-computed VAE latents (`use_pre_latents=True`)
- Loss = MSE reconstruction + auxiliary losses (routing contrastive for ProMoE, capacity prediction for DiffMoE, expert contrastive for expert_contra)
- After each optimizer step, `train.py` calls `model.module.update_noise_expert_ema()` if the model exposes it (noise_expert_ema models). Noise expert parameters are excluded from the optimizer (`requires_grad=False`).
- Checkpoints saved every `save_ckpt_interval` steps to `outputs/{model_name}/{custom_cfg_name}/checkpoints/`

### Sampling Pipeline (`sample.py`)
- FlowMatchEulerDiscreteScheduler from diffusers
- Classifier-free guidance: runs cond and uncond forward passes separately (not batched together), applies `guidance_scale * (cond - uncond) + uncond`
- Loads EMA weights (`ema_model_state_dict`) from checkpoints for sampling
- Checkpoint selection: if `step_list_for_sample` is set, loads only those checkpoints; otherwise scans `checkpoints/` for steps divisible by `sample_every_step`
- Supports resumable sampling — skips batches where output images already exist
- Extracts Inception features for FID computation alongside generated images (optional, `save_inception_features=True`)
- Output: `outputs/{model_name}/{custom_cfg_name}/sample/step{N}/`

### Pretrained Weights
- VAE loading uses `load_vae()` from `utils.py`: checks `pretrained_ckpt/vae/{repo_id}/` for a local copy first; if absent, downloads from HuggingFace and saves locally for future use.
- All training entry points (`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, `train_with_mae.py`, `sample.py`, `preprocess/preprocess_vae.py`) use this cached loading path.
- REPA teacher encoders (DINOv2) are cached to `pretrained_ckpt/encoder/` after first download via torch.hub.

### Analysis Tools (`analyses/`)
- `run_compute_flops.py` — Computes theoretical FLOPs, activated parameters, and expert frequencies for checkpoints.
- `run_tokenwise_tsne.py` / `run_samplewise_pooled_tsne.py` / `run_imagewise_tsne.py` — t-SNE visualization of expert routing at different granularities.
- `run_repa_dyna_heatmap.py` — Heatmap visualization of dynamic REPA weights across timesteps.
- `run_token_choice_expert_heatmap.py` — Heatmap of token-to-expert assignment patterns.
- `run_mos_routing_analysis.py` — MoS router teacher block selection analysis: per-block and aggregated frequency histograms, timestep evolution, token variance, and routing entropy. Uses hook-based routing weight capture (`analyses/mos_routing/extract.py`), online statistical aggregation (`analyses/mos_routing/aggregate.py`), and plotting (`analyses/mos_routing/plotting.py`). Supports all MoS model variants (global/blockwise/per_block/mos router types) via auto-detection.
- Each entry script has a matching `analyses/<basename>.md` usage guide. Reusable helpers live in `analyses/t_SNE/`, `analyses/heatmap/`, `analyses/flops/`, `analyses/mos_routing/`.

## Coding Conventions
- 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Model files follow `models_*.py` naming pattern. Preserve numeric experiment prefixes in config names (e.g., `004_ProMoE_L.yaml`).
- **1-indexed naming convention**: Script and config filenames use 1-indexed block numbers (e.g., `b3_5` means blocks 3-5 human-readable), while YAML `align_blocks` uses 0-indexed Python indices (e.g., `[2, 3, 4]`). Always maintain this distinction.
- No formatter or linter is configured — match surrounding style in the file you edit.
- No `tests/` directory; validate changes with `python -m py_compile <file>` for syntax checks and targeted smoke tests (short training run, sample pass).

### Shell Script Convention
- **All new three-in-one (train + sample + eval) `.sh` experiment scripts must follow the `scripts/template.sh` pattern** — otherwise the other experiment server cannot run them. (Legacy split-purpose scripts under `scripts/repa/` such as `train_repa_B.sh` / `sample_and_eval_repa_B.sh` predate the template and are exempt; new work should not introduce more of them.)
- **Sequential pipeline pattern**: The template implements a train→stop→sample+eval→resume loop. For each step in `step_list_for_sample`: (1) generate a temp config with `num_steps` set to the checkpoint step + 1 and `resume_checkpoint: True`, (2) train until that step then exit, (3) sample + eval with GPUs fully free, (4) resume for the next step. The final step uses the original `num_steps` from config. This avoids concurrent training + sampling, which can exceed GPU memory for XL-scale models.
- Key template patterns: `set -euo pipefail`, locate repo root via `SCRIPT_DIR`/`REPO_ROOT`, parse `model_name`/`gpu_ids`/`num_fid_samples`/`step_list_for_sample`/`orig_num_steps` from YAML using inline Python, call training/sampling/evaluation with absolute python paths, and `find ... -name images | sort -V` for evaluation directory traversal. Never use `conda activate`.
- **When creating a new script**, only two things need changing from template.sh: the `CONFIG` path and `LOG` filename. Also change the training entrypoint in the train step (`train_with_repa.py`, `train_with_MoS_repa.py`, or `train.py`) to match the model family.
- End-to-end scripts under `scripts/` use **absolute python paths** (e.g., `/mnt/workspace/yujie/.conda/envs/promoe/bin/python`) instead of `conda activate` for company server compatibility.
- Training/sampling uses the `promoe` env; evaluation uses the `fid_eval` env.

## Adding a New Experiment

### Config-driven ablation (no new model file)
If the ablation is controlled by an existing config flag (e.g., `router_norm_type`, `align_target`, `use_dynamic_coeff`), only a new YAML config and shell script are needed — the `model_name` stays the same. Add the flag with a default that preserves backward compatibility.

### New model variant
1. **Model**: Create `models/models_ProMoE_TC_<variant>.py`. Inherit from the closest existing variant. Follow `forward()` return conventions (see Auxiliary Loss Convention above).
2. **Register**: Add a `(ModelClass, config_key)` entry to `model_dict` in the appropriate training script (`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`). `sample.py` merges all dicts automatically.
3. **Config**: Create `configs/004_ProMoE_<size>_<variant>.yaml`. Set `model_name` to match the registered key. Add `MoE_config` and/or `repa_config` under the model config key as needed.
4. **Shell script**: Copy `scripts/template.sh` to `scripts/<family>/run_<size>_<variant>_train_sample_eval.sh`. Update `CONFIG` and `LOG` variables. Choose the correct training entrypoint in the train step.
5. **Validate**: `python -m py_compile models/models_ProMoE_TC_<variant>.py` then a short training run.

## Important Notes
- All paper results use `qk_norm=False`. Enable `qk_norm=True` for training beyond 2M steps.
- Token-Choice routing is default; use Expert-Choice (`models_ProMoE_EC.py`) for DDPM training.
- Evaluation requires a separate TensorFlow environment and the reference batch `VIRTUAL_imagenet256_labeled.npz` from OpenAI's guided-diffusion. `evaluation/download_ref_batches.py` can auto-download these.
- `cfg.data_path` in `config.py` must be set to your ImageNet train directory.
- Multi-GPU sampling produces different random sequences than single-GPU (different class label ordering).
- REPA training requires raw images (not just pre-computed latents) since the teacher encoder operates on pixel space. The dataset returns `(path, label, latent, raw_image)` when `load_raw_image=True`.
- Offline/air-gapped training: all training scripts and `sample.py` accept `--vae-path`; `train_with_repa.py` and `train_with_MoS_repa.py` also accept `--repa-enc-path`. See `ProMoE-REPA.md` for details.
- `preprocess/image_paths_cache.txt` caches the dataset file list; delete and rebuild it after switching datasets or reorganizing files.
- When `use_pre_latents=True`, the latent directory must be a sibling of `train/` named `sd-vae-ft-mse_Latents_256img_npz` — the code derives latent paths by replacing `train` in image paths.
- `model.py` at the repo root is an unrelated reference file (not imported anywhere in the project). Ignore it when navigating the codebase — the project's models live in `models/`.

## Workflow Rules
- **Clean up smoke-test artifacts immediately.** After a smoke test or sanity run finishes (success or failure), delete the temporary scripts, generated configs, output directories (e.g., `tb_smoke_*/`, `collapse_smoking_test*/`, `outputs/<model>/<smoke_cfg>/`), and any caches that exist only because of the smoke test. Do not let debug-only artifacts accumulate in the working tree. Long-lived artifacts — real training outputs under `outputs/`, `pretrained_ckpt/`, `training_logs/`, and project-level `__pycache__/` — are out of scope and must not be touched.
- **Run all background processes in a new tmux window of the current session.** Never use `command &`, `nohup`, or `Bash`'s `run_in_background=true` for anything that doesn't return promptly (training, sampling, long evals, watch loops, dev servers). Use:
  ```
  test -n "${TMUX:-}" || { echo "not inside tmux — attach first"; exit 1; }
  tmux new-window -t "$(tmux display-message -p '#S')" -n <name> '<command>'
  ```
  If `$TMUX` is unset, **abort and ask the user to attach to a tmux session first** — do not silently fall back to backgrounding. Short synchronous commands (`ls`, `grep`, `py_compile`, `git status`, …) continue to run in the foreground.
- **A push request implicitly authorizes a commit of the current WIP.** When the user asks to push (any phrasing — `push`, `推送`, `最后 push 所有改动`, etc.), treat it as one combined instruction: commit any uncommitted WIP relevant to the conversation first, then push. Do not ask for a separate commit confirmation. All other git-safety rules still apply: no `--no-verify`, no force-push to `main`/`master`, never stage secrets or `*.local.json` files, never `git add -A`/`.` (stage explicit paths only).

## Companion Documentation
- `ProMoE-REPA.md` — Detailed guide for all REPA variant workflows, configuration reference, and FAQ.
- `AGENTS.md` — Full project structure reference, output layout, testing guidelines, and commit conventions.
- `analyses/README.md` — Overview of analysis entrypoints; per-script usage in `analyses/<basename>.md` files.
- `plans/` — Implementation plans for Cross-Attention variants (`plan_01` through `plan_08`), covering both standard REPA and MoS cross-alignment designs.
- `implementation-plan.md` — Draft plan (Chinese) for a future "attention-weighted same-expert same-image alignment" experiment family. Not yet implemented; reference for forthcoming work, not current code.

## Project-Local Skills (`.claude/skills/`)
Two project-specific slash commands live under `.claude/skills/`. They encode the project-aware checks (model_dict ↔ models/ ↔ configs/ ↔ scripts/ four-way consistency, cross-alignment stability invariants, TrainingMonitor hook integrity) so future Claude instances don't have to re-derive them.

- `/check` — Carpet-style code-quality loop on **the entire codebase**: scan → fix → commit (per iteration) → smoke test. Terminates after 5 consecutive iterations with zero findings, or hard caps at 20 iterations. Smoke test is `py_compile` + import check only — never starts real training/sampling. Each iteration produces its own commit (`chore(check): iter N — ...`); never amends, force-pushes, or pushes.
- `/debug` — Same loop shape as `/check`, but the scan is **scoped to the current uncommitted diff** (modified + staged + untracked relative to HEAD). Does **not** commit during the loop — leaves the validated WIP dirty for the user to commit themselves. Stops immediately if the working tree is already clean.

Both skills explicitly refuse to: push/force-push/amend, run real training/sampling, edit runtime artifact dirs (`outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`, `collapse_smoking_test*/`), or touch the vendored `REPA/` (uppercase) subproject.
