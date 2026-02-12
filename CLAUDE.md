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
python train.py --config configs/004_ProMoE_L.yaml
```

### Sampling
```bash
# Single GPU, default settings (500k checkpoint, 50K images, CFG 1.0/1.5)
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml

# Custom settings
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml \
  --step_list_for_sample 200000,300000 --guide_scale_list 1.0,1.5,4.0 --num_fid_samples 10000
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
- `config.py`: Global defaults using EasyDict. All model size configs (S/B/M/L/XL) and MoE configs are defined here.
- `configs/*.yaml`: Per-experiment overrides deep-merged onto `config.py` defaults at runtime via `deep_update()` in `utils.py`.
- YAML configs set `model_name`, `gpu_ids`, batch size, learning rate, MoE hyperparameters, and sampling parameters.

### Model Registry
`train.py` and `sample.py` contain a `model_dict` mapping `model_name` strings to `(ModelClass, config_key)` pairs. Adding a new model requires an entry here.

### Model Hierarchy (in `models/`)
- `modules.py` — Shared building blocks: `Attention`, `PatchEmbed`, `TimestepEmbedder`, `LabelEmbedder`, `FinalLayer`, `MLP`, `SwiGLU`.
- `models_DiT.py` — Dense DiT baseline (no MoE). `DiTBlock` uses AdaLN-Zero modulation (6-param per-sample conditioning from timestep+class).
- `models_TCDiT.py` / `models_ECDiT.py` — Token-Choice and Expert-Choice MoE baselines.
- `models_DiffMoE.py` — DiffMoE baseline with capacity prediction.
- **`models_ProMoE_TC.py`** — Main proposed model. `SparseMoeBlock` implements two-step routing: (1) conditional routing separates uncond tokens (class=1000) to a dedicated expert, (2) prototypical routing assigns cond tokens via cosine similarity to learnable cluster centers. Includes routing contrastive loss.
- `models_ProMoE_EC.py` — Expert-Choice variant of ProMoE (recommended for DDPM training).

### Key MoE Parameters (in YAML `MoE_config`)
- `num_routed_experts`: Number of routable experts (typically 12)
- `top_k`: Experts per token (default 1)
- `routing_contrastive_lam`: Weight of contrastive loss (default 1.0)
- `routing_contrastive_temperature`: Contrastive loss temperature (default 0.07)
- `use_shared_expert` / `use_uncond_expert`: Toggle shared global expert and dedicated unconditional expert

### Training Pipeline (`train.py`)
- PyTorch DDP for multi-GPU distributed training
- Logit-normal timestep sampling (SD3-style) with Rectified Flow objective
- Mixed precision with bfloat16; gradient clipping at `max_grad_norm=0.5`
- EMA model maintained for stable generation
- Supports both raw image loading and pre-computed VAE latents (`use_pre_latents=True`)
- Loss = MSE reconstruction + auxiliary losses (routing contrastive for ProMoE, capacity prediction for DiffMoE)

### Sampling Pipeline (`sample.py`)
- FlowMatchEulerDiscreteScheduler from diffusers
- Classifier-free guidance via `forward_with_cfg` (batches cond+uncond in single forward pass)
- Extracts Inception features for FID computation alongside generated images
- Output: `outputs/{model_name}/checkpoints/sample/step{N}/` with PNG images named `img{idx}_class{label}.png`

### Output Directory Structure
```
outputs/{model_name}/
├── checkpoints/
│   ├── ckpt_step_{N}.pth
│   └── sample/
│       └── step{N}/
│           └── img256_cfg{scale}_seed0_FID{K}K_bs{B}_ema/
```

## Important Notes
- All paper results use `qk_norm=False`. Enable `qk_norm=True` for training beyond 2M steps.
- Token-Choice routing is default; use Expert-Choice (`models_ProMoE_EC.py`) for DDPM training.
- Evaluation requires a separate TensorFlow environment and the reference batch `VIRTUAL_imagenet256_labeled.npz` from OpenAI's guided-diffusion.
- `cfg.data_path` in `config.py` must be set to your ImageNet train directory.
