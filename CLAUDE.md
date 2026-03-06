# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProMoE-Plus is a Mixture-of-Experts (MoE) framework for scaling Diffusion Transformers (DiT) with explicit routing guidance, published at ICLR 2026. It implements prototypical routing strategies (Token-Choice and Expert-Choice) and hierarchical expert specialization for ImageNet 256x256 image generation.

## Commands

### Environment Setup
```bash
conda create -n promoe python=3.10 -y && pip install -r requirements.txt
```

### Training
```bash
python train.py --config configs/004_ProMoE_L.yaml
# Short smoke test:
python train.py --config configs/004_ProMoE_S.yaml --num_steps 100
```

### Sampling / Inference
```bash
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml
```

### VAE Latent Preprocessing
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess/preprocess_vae.py --latent_save_root /path/to/latents
```

### Evaluation (requires separate Python 3.9 + TensorFlow environment)
```bash
cd evaluation && pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py /path/to/generated/images
```

### No test suite exists — validate changes with short training or sampling runs.

## Architecture

### Core Pipeline
- **config.py**: Centralized defaults as `EasyDict`; YAML configs override these via `deep_update()`.
- **train.py**: Distributed training entry point. Contains `model_dict` mapping model names to classes. Uses `torchrun`/`mp.spawn` with NCCL backend, bfloat16 AMP, AdamW, EMA (`decay=0.9999`), and gradient clipping (`max_grad_norm=0.5`).
- **sample.py**: Loads EMA checkpoint, generates images using `FlowMatchEulerDiscreteScheduler` (Rectified Flow). Supports multi-GPU sampling and Inception feature caching for FID.
- **models/modules.py**: Shared components — `Attention`, `MoeMLP`, `PatchEmbed`, timestep/label embeddings.

### Model Variants (in `models/`)
Each file exports a `DiT` class. Naming: `models_<Variant>.py`.

| File | Description |
|------|-------------|
| `models_DiT.py` | Dense DiT baseline |
| `models_TCDiT.py` / `models_ECDiT.py` | Token-Choice / Expert-Choice baselines |
| `models_DiffMoE.py` | DiffMoE baseline (prior work) |
| `models_ProMoE_TC.py` | **Main published model** — prototypical routing with cosine similarity |
| `models_ProMoE_TC_symmetric.py` | Symmetric routing variant |
| `models_ProMoE_TC_sigmoid.py` | Sigmoid routing variant |
| `models_ProMoE_EC.py` | Expert-Choice variant of ProMoE |
| `models_ProMoE_TC_hierar.py` | Hierarchical sub-prototypes per expert (4 sub-prototypes) |
| `models_ProMoE_TC_hierar_expert.py` | Variable-capacity experts (1x→3x intermediate size) + cost penalty |

### Routing Architecture (ProMoE)
1. **Stage 1**: Unconditional tokens (class=1000) → fixed `uncond_expert`
2. **Stage 2**: Conditional tokens → cosine similarity against learned `cluster_centers` → top-k selection
3. Auxiliary losses injected via `AddAuxiliaryLoss` custom autograd function (forward is identity, backward adds loss gradient)

### Key MoE Config Parameters
- `num_routed_experts`: Number of routed experts (typically 12)
- `top_k`: Experts per token (typically 1)
- `routing_contrastive_lam`: Contrastive loss weight (default 1.0)
- `routing_contrastive_temperature`: 0.07
- `sub_prototype_diversity_lam`: Sub-prototype diversity loss (hierar variants, 0.1)
- `cost_penalty_lam`: Capacity cost penalty (hierar_expert variant)

## Config System
YAML files in `configs/` override `config.py` defaults. Naming convention: `NNN_<ModelName>_<Size>[_variant].yaml` (e.g., `004_ProMoE_B_hierar_expert.yaml`). The model name in YAML must match a key in `train.py:model_dict`.

## Adding a New Model Variant
1. Create `models/models_<Variant>.py` exporting a `DiT` class
2. Import and register in `train.py:model_dict`
3. Add model config dict to `config.py`
4. Create YAML config in `configs/`

## Conventions
- 4-space indentation, `snake_case` functions, `CamelCase` classes
- Commit messages: short, lowercase, imperative (e.g., `add EC routing config`)
- Never commit dataset paths, cached latents, or large checkpoints
- Distributed training: only rank 0 handles logging/checkpointing
- Checkpoints saved as `ckpt_step_{N}.pth` in `outputs/<model>/<config>/`
