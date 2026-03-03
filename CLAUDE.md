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
```bash
# Train REPA variant
bash scripts/train_repa_B.sh
# Train REPA-Shared variant (align shared expert output with teacher)
bash scripts/train_repa_shared_B.sh

# Sample + evaluate in one go (handles conda env switching)
bash scripts/sample_and_eval_repa_B.sh
bash scripts/sample_and_eval_repa_shared_B.sh
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
CUDA_VISIBLE_DEVICES=0 python run_eval.py /path/to/generated/images
```

## Architecture

### Configuration System
- `config.py`: Global defaults using EasyDict. Defines base model configs (`DiT_S_config` through `DiT_XL_config`) and MoE-specific configs (`DiffMoE_DiT_*`, `TCDiT_*`, `ECDiT_*`).
- `configs/*.yaml`: Per-experiment overrides deep-merged onto `config.py` defaults at runtime via `deep_update()` in `utils.py`.
- **Config merging flow**: ProMoE models reuse base DiT configs (e.g., `ProMoE_TC_L` maps to `DiT_L_config` in `model_dict`). The YAML adds `MoE_config` as a nested dict under the base config key (e.g., `DiT_L_config.MoE_config`), which `deep_update()` merges in. This means MoE parameters are not in `config.py` for ProMoE — they come entirely from YAML.
- The YAML filename (minus extension) becomes `custom_cfg_name`, which determines the output subdirectory: `outputs/{model_name}/{custom_cfg_name}/`.

### Model Registry
`train.py` and `train_with_repa.py` each define a `model_dict` mapping `model_name` strings to `(ModelClass, config_key)` pairs. `sample.py` merges both dicts so it can sample from any model variant. Adding a new model requires an entry in the appropriate training script's `model_dict`.

### Model Hierarchy (in `models/`)
- `modules.py` — Shared building blocks: `Attention`, `PatchEmbed`, `TimestepEmbedder`, `LabelEmbedder`, `FinalLayer`, `MLP`/`Mlp`, `SwiGLU`, `MoeMLP`, and sinusoidal position embedding utilities.
- `models_DiT.py` — Dense DiT baseline (no MoE). `DiTBlock` uses AdaLN-Zero modulation (6-param per-sample conditioning from timestep+class).
- `models_TCDiT.py` / `models_ECDiT.py` — Token-Choice and Expert-Choice MoE baselines.
- `models_DiffMoE.py` — DiffMoE baseline with capacity prediction.
- **`models_ProMoE_TC.py`** — Main proposed model. `SparseMoeBlock` implements two-step routing: (1) conditional routing separates uncond tokens (class=1000) to a dedicated expert, (2) prototypical routing assigns cond tokens via cosine similarity to learnable `cluster_centers`. Includes routing contrastive loss via `AddAuxiliaryLoss` autograd trick.
- `models_ProMoE_EC.py` — Expert-Choice variant of ProMoE (recommended for DDPM training).
- `models_ProMoE_TC_repa.py` — ProMoE-TC with REPA projectors. Adds MLP projectors (`build_repa_projector`) that align intermediate DiT features with a frozen DINOv2 teacher encoder. In training, `forward()` returns `(pred, zs_proj)` where `zs_proj` is a list of projected features for REPA loss; in eval mode returns only `pred`.
- `models_ProMoE_TC_repa_shared.py` — REPA variant that aligns the **shared expert output** (rather than the full block output) with the DINOv2 teacher. `SparseMoeBlock.forward()` returns `(final_output, loss, shared_output)` and `DiTBlock.forward()` returns `(x, shared_output)`. The projector at `encoder_depth` operates on `shared_output` instead of `x`. Requires `encoder_depth` to point to a MoE block (asserted at init).

### Auxiliary Loss Convention
Model `forward()` returns either a plain tensor (DiT) or a tuple for models with auxiliary losses:
- **DiffMoE**: Returns `(pred, "Capacity_Pred", layer_idx_list, ones_list, pred_c_list, loss_weight)`. Training loop computes BCEWithLogitsLoss for capacity prediction.
- **ProMoE**: Uses `AddAuxiliaryLoss` autograd function to inject contrastive loss gradients directly into the forward pass — returns a plain tensor but the auxiliary loss gradient flows through automatically.
- **ProMoE-REPA**: Returns `(pred, zs_proj)` during training. The training loop in `train_with_repa.py` computes `compute_repa_loss(teacher_z, zs_proj)` and adds it weighted by `proj_coeff`. Total loss = MSE + REPA loss * `proj_coeff` + routing contrastive loss (via autograd).

### REPA Module (`repa/` vs `REPA/`)
- `repa/` (lowercase) — ProMoE's REPA integration: encoder loading, loss computation, used by `train_with_repa.py`.
- `REPA/` (uppercase) — Separate standalone REPA subproject (original codebase). Treat changes there as scoped work independent from ProMoE.
- `repa/encoder.py` — Loads frozen DINOv2 teacher encoders (`dinov2-vit-{b,l,g}` and `dinov2reg-vit-{b,l,g}`). Downloads via torch.hub on first use, caches to `pretrained_ckpt/encoder/`. Handles positional embedding resampling for target resolution.
- `repa/loss.py` — `compute_repa_loss(z_teacher, z_student_list)`: negative cosine similarity between teacher patch features and projected student features, averaged across alignment points.
- `train_with_repa.py` — Extended training loop that loads raw images alongside VAE latents, extracts teacher features with `extract_teacher_features()`, and adds REPA projection loss to the total loss.

### REPA Parameters (in YAML `repa_config`)
- `enc_type`: Teacher encoder model (e.g., `"dinov2-vit-b"`)
- `encoder_depth`: Which transformer layer to extract student features from (e.g., 4)
- `z_dims`: List of projection dimensions for alignment (e.g., `[768]`)
- `projector_dim`: Hidden size of the 3-layer MLP projector (e.g., 2048)
- `proj_coeff`: Weight of REPA loss in total loss (e.g., 0.5)

### Key MoE Parameters (in YAML `MoE_config`)
- `num_routed_experts`: Number of routable experts (typically 12)
- `top_k`: Experts per token (default 1)
- `routing_contrastive_lam`: Weight of contrastive loss (default 1.0)
- `routing_contrastive_temperature`: Contrastive loss temperature (default 0.07)
- `use_shared_expert` / `use_uncond_expert`: Toggle shared global expert and dedicated unconditional expert
- `interleave`: Whether to alternate MoE and dense FFN layers
- `router_weight_mode`: How to weight expert outputs (`"softmax"`, `"identity"`)

### Training Pipeline (`train.py`)
- PyTorch DDP for multi-GPU distributed training via `mp.spawn`
- Logit-normal timestep sampling (SD3-style) with Rectified Flow objective
- Mixed precision with bfloat16; gradient clipping at `max_grad_norm=0.5`
- EMA model maintained for stable generation
- Supports both raw image loading and pre-computed VAE latents (`use_pre_latents=True`)
- Loss = MSE reconstruction + auxiliary losses (routing contrastive for ProMoE, capacity prediction for DiffMoE)
- Checkpoints saved every `save_ckpt_interval` steps to `outputs/{model_name}/{custom_cfg_name}/checkpoints/`

### Sampling Pipeline (`sample.py`)
- FlowMatchEulerDiscreteScheduler from diffusers
- Classifier-free guidance: runs cond and uncond forward passes separately (not batched together), applies `guidance_scale * (cond - uncond) + uncond`
- Loads EMA weights (`ema_model_state_dict`) from checkpoints for sampling
- Supports resumable sampling — skips batches where output images already exist
- Extracts Inception features for FID computation alongside generated images (optional, `save_inception_features=True`)
- Output: `outputs/{model_name}/{custom_cfg_name}/sample/step{N}/`

### Pretrained Weights
- VAE loading uses `load_vae()` from `utils.py`: checks `pretrained_ckpt/vae/{repo_id}/` for a local copy first; if absent, downloads from HuggingFace and saves locally for future use.
- All training entry points (`train.py`, `train_with_repa.py`, `sample.py`, `preprocess/preprocess_vae.py`) use this cached loading path.
- REPA teacher encoders (DINOv2) are cached to `pretrained_ckpt/encoder/` after first download via torch.hub.

## Coding Conventions
- 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Model files follow `models_*.py` naming pattern. Preserve numeric experiment prefixes in config names (e.g., `004_ProMoE_L.yaml`).
- No formatter or linter is configured — match surrounding style in the file you edit.
- No `tests/` directory; validate changes with targeted smoke tests (short training run, sample pass).

## Important Notes
- All paper results use `qk_norm=False`. Enable `qk_norm=True` for training beyond 2M steps.
- Token-Choice routing is default; use Expert-Choice (`models_ProMoE_EC.py`) for DDPM training.
- Evaluation requires a separate TensorFlow environment and the reference batch `VIRTUAL_imagenet256_labeled.npz` from OpenAI's guided-diffusion. `evaluation/download_ref_batches.py` can auto-download these.
- `cfg.data_path` in `config.py` must be set to your ImageNet train directory.
- Multi-GPU sampling produces different random sequences than single-GPU (different class label ordering).
- REPA training requires raw images (not just pre-computed latents) since the teacher encoder operates on pixel space. The dataset returns `(path, label, latent, raw_image)` when `load_raw_image=True`.
- Offline/air-gapped training: pass `--vae-path /path/to/sd-vae-ft-mse` and `--repa-enc-path /path/to/dinov2_state_dict.pth` to skip automatic downloads. See `ProMoE-REPA.md` for details.
