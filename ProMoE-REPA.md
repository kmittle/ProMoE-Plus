# ProMoE-REPA: ProMoE with Representation Alignment

This document explains how to train, sample, and evaluate ProMoE-REPA: ProMoE (MoE Diffusion Transformer) with REPA (REPresentation Alignment) integrated into the training pipeline. REPA accelerates convergence and improves generation quality by aligning intermediate DiT features with a frozen DINOv2 teacher encoder.

## Table of Contents

- [Method Overview](#method-overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Sampling](#sampling)
- [Evaluation](#evaluation)
- [Custom Pretrained Weight Paths](#custom-pretrained-weight-paths)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [FAQ](#faq)

---

## Method Overview

ProMoE-REPA adds a representation alignment loss on top of the standard ProMoE training objective:

```text
Total Loss = MSE (Rectified Flow) + routing_contrastive_loss (auto-injected) + proj_coeff * REPA_loss
```

- **MSE Loss**: the Rectified Flow reconstruction target.
- **Routing Contrastive Loss**: injected automatically by ProMoE's `AddAuxiliaryLoss` mechanism during backpropagation, so it does not need to be added manually in the main training loop.
- **REPA Loss**: computed as negative cosine similarity between projected DiT hidden features and patch-level features from a frozen DINOv2 teacher encoder.

During training, the REPA model's `forward()` returns `(pred, zs_proj)`, where `zs_proj` is the list of projected features used for alignment. During sampling, only the prediction is used, and the projector branch is ignored.

## Environment Setup

### Training and Sampling Environment

```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```

Main dependencies include PyTorch 2.6+, diffusers 0.32+, timm 1.0+, einops, and accelerate.

### Evaluation Environment (Separate)

Evaluation depends on TensorFlow and should use a separate conda environment to avoid conflicts:

```bash
conda create -n promoe_eval python=3.9 -y
conda activate promoe_eval
cd evaluation
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

## Data Preparation

### 1. ImageNet Dataset

Download the [ImageNet](http://image-net.org/download) training set and update the dataset path in `config.py`:

```python
cfg.data_path = "/path/to/ImageNet/train"
```

### 2. VAE Latent Preprocessing (Recommended)

Precomputing VAE latents can significantly speed up training because the VAE forward pass is skipped during training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python preprocess/preprocess_vae.py \
    --latent_save_root "/path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz"
```

After preprocessing, set `use_pre_latents: True` in the YAML config (already enabled in the provided REPA config).

**Note**: REPA training still requires the original images in addition to VAE latents, because the teacher encoder runs in pixel space. When REPA is enabled, the dataset automatically loads both latent files and raw images (`load_raw_image=True`), so no extra setup is required.

## Training

### Quick Start

```bash
# Convenience script
bash scripts/train_repa_B.sh

# Or run directly
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml
```

### Multi-GPU Training

Training GPUs are specified by the `gpu_ids` field in the YAML config, and training uses PyTorch DDP:

```yaml
gpu_ids: [0, 1, 2, 3]
```

### Training Outputs

- Checkpoints are saved to: `outputs/{model_name}/{config_name}/checkpoints/`
- TensorBoard logs are written to: `outputs/{model_name}/{config_name}/tensorboard/`
- The text log file is written to: `outputs/{model_name}/{config_name}/training.log`
- The checkpoint interval is controlled by `save_ckpt_interval` (default in `configs/004_ProMoE_B_repa.yaml`: every 50000 steps)
- Training can resume automatically from the latest checkpoint when `resume_checkpoint: True`

### Training Notes

- All paper results use `qk_norm: False`. If you plan to train beyond 2M steps, enabling `qk_norm: True` is recommended for stability.
- The current REPA variants use Token-Choice routing via `models/models_ProMoE_TC_repa.py`.
- Mixed precision uses `bfloat16`, and gradient clipping uses `max_grad_norm = 0.5`.

## Sampling

Sampling uses the same YAML config file as training and runs through `sample.py`:

```bash
# Default config behavior: load the 500K checkpoint, generate 50K images, CFG 1.0 and 1.5
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_B_repa.yaml

# Custom settings
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config configs/004_ProMoE_B_repa.yaml \
    --step_list_for_sample 300000 \
    --guide_scale_list 1.0,1.5,4.0 \
    --num_fid_samples 10000
```

### Multi-GPU Sampling

You can use multiple GPUs either through `CUDA_VISIBLE_DEVICES` or by setting `sample_gpu_ids` in the YAML config. Note that the random sequence used during sampling (for example, sampled class labels) will differ between single-GPU and multi-GPU runs.

### Sampling Outputs

Generated images are saved as PNG files under:

```text
outputs/{model_name}/{config_name}/sample/step{N}/img{image_size}_cfg{cfg_scale}_seed{global_seed}_FID{K}K_bs{sample_batch_size}_ema/images/
```

For example, with the default REPA-B config at step 500000 and CFG 1.5, the image folder is created under:

```text
outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/sample/step500000/...
```

## Evaluation

### One-Click Sampling + Evaluation

```bash
bash scripts/sample_and_eval_repa_B.sh
```

This helper script:

- samples 50K images from the step-300000 checkpoint in the `promoe` environment
- switches to the `promoe_eval` environment
- evaluates every generated `images/` folder under `outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/sample/step300000/`

### Manual Evaluation

```bash
conda activate promoe_eval
cd evaluation

# The reference batch will be downloaded automatically if missing
CUDA_VISIBLE_DEVICES=0 python run_eval.py /path/to/generated/images
```

Evaluation metrics include FID, IS, sFID, Precision, and Recall.

The reference batch file `VIRTUAL_imagenet256_labeled.npz` is downloaded automatically into `evaluation/` on the first run if it is missing. The download path is protected by a file lock so multiple processes do not collide.

## Custom Pretrained Weight Paths

If the machine cannot access the internet, or if you already have local copies of the pretrained weights, you can pass local paths on the command line to skip automatic downloads.

### `--vae-path`

Specify a local VAE directory (diffusers format). This is supported by `train.py`, `train_with_repa.py`, and `sample.py`:

```bash
# Training
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml \
    --vae-path /path/to/local/sd-vae-ft-mse

# Sampling
python sample.py --config configs/004_ProMoE_B_repa.yaml \
    --vae-path /path/to/local/sd-vae-ft-mse
```

### `--repa-enc-path`

Specify a local REPA teacher encoder `state_dict` file (`.pth`). This is supported only by `train_with_repa.py`:

```bash
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml \
    --repa-enc-path /path/to/dinov2_vitb14/state_dict.pth
```

### Provide Both

```bash
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml \
    --vae-path /path/to/local/sd-vae-ft-mse \
    --repa-enc-path /path/to/dinov2_vitb14/state_dict.pth
```

If neither argument is provided, the code follows the default behavior: it downloads and caches the weights automatically on first use, then loads them from `pretrained_ckpt/` afterward.

## Configuration Reference

See `configs/004_ProMoE_B_repa.yaml` for a complete example. The key parameters are listed below.

### REPA Parameters (Top-Level `repa_config`)

| Parameter | Description | Default |
|------|------|--------|
| `enc_type` | Teacher encoder type | `"dinov2-vit-b"` |
| `proj_coeff` | Weight of the REPA loss | `0.5` |

### Model-Level REPA Parameters (`DiT_*_config.repa_config`)

| Parameter | Description | Example |
|------|------|--------|
| `enc_type` | Teacher encoder type (must match the top-level setting) | `"dinov2-vit-b"` |
| `encoder_depth` | DiT layer index used for alignment features | `4` |
| `z_dims` | Projection output dimensions (must match the teacher `embed_dim`) | `[768]` |
| `projector_dim` | Hidden dimension of the projector MLP | `2048` |

### MoE Parameters (`DiT_*_config.MoE_config`)

| Parameter | Description | Default |
|------|------|--------|
| `num_routed_experts` | Number of routed experts | `12` |
| `top_k` | Number of experts selected per token | `1` |
| `routing_contrastive_lam` | Weight of the routing contrastive loss | `1.0` |
| `routing_contrastive_temperature` | Temperature for the contrastive loss | `0.07` |
| `use_shared_expert` | Whether to use a shared expert | `True` |
| `interleave` | Whether to interleave MoE and dense FFN layers | `True` |
| `router_weight_mode` | Expert output weighting mode | `"identity"` |

### Training Parameters

| Parameter | Description | Example |
|------|------|--------|
| `total_train_batch_size` | Global batch size | `256` |
| `lr` | Learning rate | `0.0001` |
| `num_steps` | Total training steps | `500000` |
| `save_ckpt_interval` | Checkpoint save interval | `50000` |
| `use_pre_latents` | Whether to use precomputed VAE latents | `True` |

## Project Structure

```text
ProMoE-Plus/
├── train.py                        # Standard ProMoE training entrypoint
├── train_with_repa.py              # ProMoE-REPA training entrypoint
├── sample.py                       # Sampling entrypoint (shared across model variants)
├── config.py                       # Global default configuration
├── utils.py                        # Shared utilities (load_vae, deep_update, etc.)
├── configs/
│   ├── 004_ProMoE_B_repa.yaml      # REPA-B config
│   ├── 004_ProMoE_L.yaml           # ProMoE-L config
│   └── ...
├── models/
│   ├── modules.py                  # Shared modules (Attention, MLP, embedders, etc.)
│   ├── models_ProMoE_TC_repa.py    # ProMoE-TC with REPA projectors
│   ├── models_ProMoE_TC.py         # ProMoE Token-Choice
│   ├── models_ProMoE_EC.py         # ProMoE Expert-Choice
│   └── ...
├── repa/
│   ├── encoder.py                  # DINOv2 teacher encoder loading
│   └── loss.py                     # REPA loss computation
├── REPA/                           # Upstream REPA subproject kept in the repo
├── scripts/
│   ├── train_repa_B.sh             # Training convenience script
│   └── sample_and_eval_repa_B.sh   # Sampling + evaluation helper
├── preprocess/
│   └── preprocess_vae.py           # VAE latent preprocessing
├── evaluation/
│   ├── run_eval.py                 # FID/evaluation entrypoint
│   ├── evaluator.py                # OpenAI evaluator wrapper
│   └── download_ref_batches.py     # Automatic reference batch download
├── pretrained_ckpt/                # Cache directory for pretrained weights
│   ├── vae/                        #   VAE weights
│   └── encoder/                    #   DINOv2 teacher weights
└── outputs/                        # Training outputs (checkpoints, logs, samples)
```

## FAQ

**Q: How much slower is REPA training than standard training?**  
A: The extra cost comes from the teacher encoder forward pass and the projection loss. In practice, it typically adds about 10-15% training time. The teacher encoder is frozen, so it does not require backpropagation.

**Q: Can I train using only precomputed VAE latents without loading raw images?**  
A: No. REPA training must load raw images because the DINOv2 teacher encoder operates in pixel space. The dataset returns both the latent and the raw image when REPA is enabled.

**Q: What should `z_dims` be set to?**  
A: It must match the teacher encoder `embed_dim`. For example, `dinov2-vit-b` uses 768, `dinov2-vit-l` uses 1024, and `dinov2-vit-g` uses 1536.

**Q: What if pretrained weight downloads fail or conflict across multiple processes?**  
A: The code already guards downloads so rank 0 downloads first and other ranks wait. You can also bypass downloads entirely by passing `--vae-path` and `--repa-enc-path`.
