# ProMoE-REPA

This document reflects the current REPA-related code in `ProMoE-Plus`. It covers the standard ProMoE-REPA path, dynamic REPA variants, router-guided variants, and MoS-REPA variants that are currently wired into the repository.

## Table of Contents

- [What Is Included](#what-is-included)
- [Method Overview](#method-overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Sampling](#sampling)
- [Evaluation](#evaluation)
- [Pretrained Weight Paths and Caches](#pretrained-weight-paths-and-caches)
- [Configuration Reference](#configuration-reference)
- [Relevant Files](#relevant-files)
- [FAQ and Notes](#faq-and-notes)

---

## What Is Included

### Training Entrypoints

| Entrypoint | Scope |
| --- | --- |
| `train_with_repa.py` | Standard REPA and most REPA-derived ProMoE variants |
| `train_with_MoS_repa.py` | MoS-REPA and MoS-REPA-Naive |
| `sample.py` | Shared sampling entrypoint for base, REPA, and MoS-REPA models |

`sample.py` merges model registries from `train.py`, `train_with_repa.py`, and `train_with_MoS_repa.py`, so one script can sample all registered families.

### REPA Families Currently Registered

`train_with_repa.py` registers:

- `ProMoE_TC_REPA_{S,B,L,XL}`
- `ProMoE_TC_REPA_DYNA_{S,B,L,XL}`
- `ProMoE_TC_REPA_DYNA_SELECT_{S,B,L,XL}`
- `ProMoE_TC_REPA_DYNA_SCALE_{S,B,L,XL}`
- `ProMoE_TC_REPA_DYNA_ONLY_{S,B,L,XL}`
- `ProMoE_TC_REPA_Shared_{S,B,L,XL}`
- `ProMoE_TC_REPA_Cond_{S,B,L,XL}`
- `ProMoE_TC_REPA_Router_{S,B,L,XL}`
- `ProMoE_TC_REPA_Router_Contra_{S,B,L,XL}`
- `ProMoE_TC_REPA_Routed_{S,B,L,XL}`
- `ProMoE_TC_REPA_Double_Share_{S,B,L,XL}`

`train_with_MoS_repa.py` registers:

- `ProMoE_TC_REPA_MoS_{B,L}`
- `ProMoE_TC_REPA_MoS_Naive_{B,L}`

The repo currently ships example YAMLs mainly for B-scale REPA experiments.

---

## Method Overview

### Standard REPA Family

For the standard REPA training path, the explicit loss assembled in `train_with_repa.py` is:

```text
total_loss = mse_loss + repa_loss_term
```

with model-internal auxiliary MoE losses added inside the model path when enabled, for example routing contrastive loss or router alignment losses.

- `mse_loss`: the Rectified Flow / diffusion training target.
- `repa_loss_term`: negative cosine similarity between projected student tokens and frozen teacher patch tokens.
- Auxiliary routing losses: injected by the model via the existing MoE loss path rather than being manually re-added in the outer loop.

`compute_repa_loss()` in `repa/loss.py` supports two cases:

- plain projected features: `[(B, T, D), ...]`
- token-weighted projected features: `[(z_proj, token_weight), ...]`

The second form is used by dynamic REPA variants so per-token weighting can be applied inside the REPA loss itself.

### MoS-REPA Family

MoS-REPA uses a different training path:

- the teacher encoder exposes features from all transformer blocks
- the model returns `(pred, mos_repa_loss)`
- the outer loop applies `mos_repa_loss * proj_coeff`

This makes MoS-REPA a block-to-block alignment problem rather than a single-layer alignment problem.

### Variant Summary

- `REPA`: aligns the configured DiT hidden layer with teacher patch features.
- `REPA-Shared`: aligns shared-expert output.
- `REPA-Cond`: aligns only conditional samples and skips unconditional/null-class samples.
- `REPA-Routed`: aligns routed expert output.
- `REPA-Double-Share`: adds a dedicated REPA-only shared branch for alignment.
- `REPA-DYNA`: uses token-weighted REPA alignment.
- `REPA-DYNA-SELECT`: keeps only a token ratio for REPA via `repa_select_ratio`.
- `REPA-DYNA-SCALE`: uses learnable scaling with token selection.
- `REPA-DYNA-ONLY`: ablation where dynamic weights are capped by `proj_coeff`.
- `REPA-Router`: adds REPA-guided router alignment.
- `REPA-Router-Contra`: linearly hands off router auxiliary weight from router REPA to routing contrastive loss over `router_loss_decay_steps`.
- `MoS-REPA`: each DiT block has its own teacher-block router.
- `MoS-REPA-Naive`: uses a global transformer-based block router.

---

## Environment Setup

### Training / Sampling Environment

```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```

### Evaluation Environment

Evaluation uses the TensorFlow-based pipeline in `evaluation/` and is best kept in a separate environment:

```bash
conda create -n promoe_eval python=3.9 -y
conda activate promoe_eval
cd evaluation
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Point the Repo to ImageNet `train/`

Set `cfg.data_path` in `config.py`, or override it from YAML (or via the `PROMOE_DATA_PATH` env var):

```python
cfg.data_path = "/path/to/ImageNet/train"
```

For an automated fresh-server setup, `preprocess/prepare_imagenet.sh` downloads full-resolution ImageNet-1K (HuggingFace→ModelScope), materialises it to `datasets/imagenet/train/`, and VAE-encodes it (see CLAUDE.md "Dataset auto-preparation"). It materialises **raw JPEGs plus latents**, so REPA training — which also needs raw images for teacher features — works from the same prepared tree.

### 2. Optional but Recommended: Precompute VAE Latents

```bash
python preprocess/preprocess_vae.py \
  --latent_save_root /path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz
```

When `use_pre_latents: True`, training loads `.latent.npz` files instead of running the VAE online.

### 3. Latent Mapping Rule

The training code derives each latent path by:

- replacing `train` in the image path with `sd-vae-ft-mse_Latents_256img_npz`
- changing the suffix to `.latent.npz`

Example:

```text
/path/to/ImageNet/train/n01440764/img1.JPEG
-> /path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz/n01440764/img1.latent.npz
```

### 4. Cache Behavior

When `use_pre_latents=True`, dataset traversal uses:

```text
preprocess/image_paths_cache.txt
```

If dataset contents or layout change, regenerate that cache before rerunning.

### 5. Important REPA Detail

Even with precomputed latents, REPA training still needs raw images because the teacher encoder runs in pixel space.

- `train_with_repa.py` loads `(img_path, label, latent_z, raw_image)` when REPA is enabled
- `train_with_MoS_repa.py` does the same for MoS-REPA

---

## Training

### Standard REPA

```bash
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml
```

### MoS-REPA

```bash
python train_with_MoS_repa.py --config configs/004_ProMoE_B_repa_MoS.yaml
```

### Optional Local Weight Overrides

```bash
python train_with_repa.py \
  --config configs/004_ProMoE_B_repa.yaml \
  --vae-path /path/to/sd-vae-ft-mse \
  --repa-enc-path /path/to/state_dict.pth
```

```bash
python train_with_MoS_repa.py \
  --config configs/004_ProMoE_B_repa_MoS.yaml \
  --vae-path /path/to/sd-vae-ft-mse \
  --repa-enc-path /path/to/state_dict.pth
```

### Convenience Scripts

Simple train-only wrappers:

- `bash scripts/repa/train_repa_B.sh`
- `bash scripts/repa/train_repa_shared_B.sh`
- `bash scripts/repa/train_repa_cond_B.sh`

One-click train/sample/eval wrappers:

- `bash scripts/repa/run_B_repa_router_train_sample_eval.sh`
- `bash scripts/repa/run_B_repa_router_contra_train_sample_eval.sh`
- `bash scripts/repa/run_B_repa_routed_train_sample_eval.sh`
- `bash scripts/repa/run_B_repa_double_share_train_sample_eval.sh`
- `bash scripts/dynamic_repa/run_B_repa_dyna_train_sample_eval.sh`
- `bash scripts/dynamic_repa/run_B_repa_dyna_select_train_sample_eval.sh`
- `bash scripts/dynamic_repa/run_B_repa_dyna_select_r25_train_sample_eval.sh`
- `bash scripts/dynamic_repa/run_B_repa_dyna_select_r75_train_sample_eval.sh`
- `bash scripts/dynamic_repa/run_B_repa_dyna_scale_train_sample_eval.sh`
- `bash scripts/dynamic_repa/run_B_repa_dyna_only_train_sample_eval.sh`
- `bash scripts/MoS_repa/run_B_repa_mos_train_sample_eval.sh`
- `bash scripts/MoS_repa/run_B_repa_mos_naive_train_sample_eval.sh`

### Training Notes

- `gpu_ids` from YAML is used to set `CUDA_VISIBLE_DEVICES`.
- Output root is:

```text
outputs/<model_name>/<config_stem>/
```

- `custom_cfg_name` is automatically taken from the YAML filename stem.
- Provided B-scale REPA configs currently use:
  - `total_train_batch_size: 256`
  - `num_steps: 501000`
  - `save_ckpt_interval: 1000`
  - `param_dtype: torch.bfloat16` (from global defaults)
  - `max_grad_norm: 0.5`
- If `resume_checkpoint: True` and no checkpoint exists, the loader logs an error and training continues from step 0.
- When `resume_checkpoint_step` is set, the loader will try that exact step first.

### Output Layout

```text
outputs/<model_name>/<config_stem>/
├── checkpoints/ckpt_step_*.pth
├── training.log
└── tensorboard/
```

---

## Sampling

Use the same YAML that was used for training:

```bash
python sample.py --config configs/004_ProMoE_B_repa.yaml
```

Override checkpoints / CFG scales / sample count from the command line:

```bash
python sample.py \
  --config configs/004_ProMoE_B_repa.yaml \
  --step_list_for_sample 300000,500000 \
  --guide_scale_list 1.0,1.5 \
  --num_fid_samples 50000
```

Sampling also supports a local VAE path:

```bash
python sample.py \
  --config configs/004_ProMoE_B_repa.yaml \
  --vae-path /path/to/sd-vae-ft-mse
```

### Checkpoint Selection Rules

- If `step_list_for_sample` is set, only those checkpoints are loaded.
- Otherwise, `sample.py` scans `checkpoints/` and loads steps divisible by `sample_every_step`.
- If `--num_fid_samples` is passed, `save_img_num` is updated to the same value.

### GPU Selection Rules

- If `sample_gpu_ids` is provided in config/kwargs, `sample.py` uses it.
- Otherwise, it uses all currently visible GPUs.

### Sampling Outputs

Generated PNGs are saved under:

```text
outputs/<model_name>/<config_stem>/sample/step<step>/img<image_size>_cfg<guide_scale>_seed<global_seed>_FID<K>K_bs<sample_batch_size>_ema/images/
```

Example:

```text
outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/sample/step500000/...
```

Image filenames include class labels in the form:

```text
img000123_class456.png
```

That suffix is required by the evaluation pipeline.

---

## Evaluation

### Manual Evaluation

Run from inside `evaluation/`:

```bash
cd evaluation
python run_eval.py /path/to/generated/images --count 50000
```

Pack PNGs to NPZ without running the evaluator:

```bash
python run_eval.py /path/to/generated/images --count 50000 --no-eval
```

### Helper Scripts

- `bash scripts/repa/sample_and_eval_repa_B.sh`
- `bash scripts/repa/sample_and_eval_repa_shared_B.sh`
- `bash scripts/repa/sample_and_eval_repa_cond_B.sh`

### Evaluation Notes

- `evaluation/run_eval.py` always calls `ensure_ref_batches()` first.
- Missing reference batches are downloaded automatically.
- Output files:
  - `<image_folder>.npz`
  - `<image_folder>_eval_openai.txt` when evaluation is run
- The evaluator expects generated filenames containing `_class<id>.png`.

---

## Pretrained Weight Paths and Caches

### Command-Line Overrides

| Argument | Supported by |
| --- | --- |
| `--vae-path` | `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, `sample.py` |
| `--repa-enc-path` | `train_with_repa.py`, `train_with_MoS_repa.py` |

Sampling never needs teacher weights, so `sample.py` does not accept `--repa-enc-path`.

### Automatic Cache Locations

| Component | Cache location |
| --- | --- |
| VAE | `pretrained_ckpt/vae/<hf_repo_id_with_slash_replaced>/` |
| REPA teacher encoder | `pretrained_ckpt/encoder/<hub_name>/state_dict.pth` |

Examples:

- VAE cache for `stabilityai/sd-vae-ft-mse`:

```text
pretrained_ckpt/vae/stabilityai--sd-vae-ft-mse/
```

- DINOv2-B cache:

```text
pretrained_ckpt/encoder/dinov2_vitb14/state_dict.pth
```

### Distributed Download Behavior

For both the VAE and REPA teacher:

- rank 0 performs the first download/cache population
- other ranks wait at a barrier
- all ranks then load from local cache

Supported teacher `enc_type` strings follow the pattern:

```text
{family}-{arch}-{size}
```

Examples:

- `dinov2-vit-b`
- `dinov2-vit-l`
- `dinov2-vit-g`
- `dinov2reg-vit-b`

---

## Configuration Reference

See `configs/004_ProMoE_B_repa.yaml`, `configs/004_ProMoE_B_repa_MoS.yaml`, and nearby variants for full examples.

### Common Top-Level Fields

| Field | Meaning | Typical value in provided B configs |
| --- | --- | --- |
| `model_name` | Model registry key | `ProMoE_TC_REPA_B` |
| `gpu_ids` | Training GPUs; used to set `CUDA_VISIBLE_DEVICES` | `[0, 1, 2, 3]` |
| `image_size` | Training and sampling resolution | `256` |
| `total_train_batch_size` | Global train batch size | `256` |
| `lr` | AdamW learning rate | `0.0001` |
| `weight_decay` | AdamW weight decay | `0` |
| `use_pre_latents` | Use precomputed VAE latents | `True` |
| `resume_checkpoint` | Resume from latest checkpoint if available | `True` |
| `num_steps` | Total train steps | `501000` |
| `save_ckpt_interval` | Checkpoint save interval | `1000` |
| `log_interval` | Log interval in training loop | `10` |
| `step_list_for_sample` | Explicit checkpoints for sampling | `[300000, 500000]` |
| `num_fid_samples` | Number of images to generate/evaluate | `50000` |
| `sample_batch_size` | Per-process sampling batch size | `128` |
| `sample_steps` | Number of diffusion sampling steps | `250` |
| `sample_shift` | Scheduler shift used at sampling time | `1.0` |
| `guide_scale_list` | CFG scales to sample | `[1.0, 1.5]` |
| `save_inception_features` | Save inception activations instead of only PNGs | `False` |

### Top-Level `repa_config`

This block is read by `train_with_repa.py` or `train_with_MoS_repa.py`.

| Field | Meaning |
| --- | --- |
| `enc_type` | Teacher encoder type, for example `dinov2-vit-b` |
| `proj_coeff` | Global coefficient applied to REPA or MoS-REPA loss |

Example:

```yaml
repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

### Model-Level `DiT_*_config.repa_config`

This block configures where and how the model exposes REPA features.

Core fields used by the standard REPA path:

| Field | Meaning |
| --- | --- |
| `enc_type` | Teacher type; when present it should match the top-level `repa_config.enc_type` |
| `encoder_depth` | DiT layer index used for alignment |
| `z_dims` | Projector output dims; must match teacher `embed_dim` |
| `projector_dim` | Hidden width of the projector MLP |

Variant-specific fields:

| Field | Used by | Meaning |
| --- | --- | --- |
| `repa_select_ratio` | `DYNA_SELECT`, `DYNA_SCALE` | Fraction of tokens kept for REPA |
| `proj_coeff` | `DYNA_ONLY` | Per-model cap used in the ablation path |
| `router_repa_coeff` | `Router` | Router alignment coefficient |
| `router_loss_decay_steps` | `Router_Contra` | Steps for linear handoff from router REPA to routing contrastive loss |
| `num_teacher_blocks` | `MoS` / `MoS_Naive` | Teacher depth used for block-level routing |
| `router_hidden_dim` | `MoS_Naive` | Hidden width of the transformer block router |
| `num_router_blocks` | `MoS_Naive` | Number of transformer blocks in the block router |
| `router_num_heads` | `MoS_Naive` | Attention heads in the block router |

MoS-specific validation rules:

- `z_dims` entries must match the teacher `embed_dim`
- `enc_type` must match the top-level `repa_config.enc_type`
- `num_teacher_blocks` must match the actual teacher depth
- if `num_teacher_blocks` is omitted, `train_with_MoS_repa.py` auto-injects it from the teacher type

### `MoE_config` Fields Seen Across REPA YAMLs

| Field | Typical value |
| --- | --- |
| `num_routed_experts` | `12` |
| `moe_intermediate_size` | `1536` |
| `shared_expert_intermediate_size` | `1536` |
| `use_shared_expert` | `True` |
| `interleave` | `True` |
| `top_k` | `1` |
| `router_weight_mode` | `identity` |
| `routing_contrastive_lam` | usually `1`, but `0` in `REPA_Router` |
| `use_top_k_for_routing_contrastive` | `True` |
| `routing_contrastive_temperature` | `0.07` |

For `configs/004_ProMoE_B_repa_router.yaml`, keep:

```yaml
routing_contrastive_lam: 0
```

because the routing contrastive code path is intentionally disabled in that router variant.

---

## Relevant Files

```text
train_with_repa.py
train_with_MoS_repa.py
sample.py
config.py
repa/encoder.py
repa/loss.py
models/models_ProMoE_TC_repa*.py
configs/004_ProMoE_B_repa*.yaml
scripts/repa/
scripts/dynamic_repa/
scripts/MoS_repa/
evaluation/run_eval.py
```

Useful directories:

- `scripts/repa/`: baseline REPA, Shared, Cond, Router, Routed, Double-Share helpers
- `scripts/dynamic_repa/`: DYNA, DYNA-SELECT, DYNA-SCALE, DYNA-ONLY helpers
- `scripts/MoS_repa/`: MoS-REPA helpers
- `REPA/`: upstream-style standalone REPA subproject kept in the repo

---

## FAQ and Notes

**Q: Does REPA work with precomputed latents?**  
A: Yes, but raw images are still loaded for teacher feature extraction.

**Q: What should `z_dims` be?**  
A: It must match the teacher encoder embedding size. For example, DINOv2-B uses `768`, DINOv2-L uses `1024`, and DINOv2-G uses `1536`.

**Q: What happens if `resume_checkpoint: True` but there is no checkpoint yet?**  
A: The loader logs an error and training proceeds from step 0.

**Q: Can `sample.py` run every model family in this repo?**  
A: It can sample all models registered by `train.py`, `train_with_repa.py`, and `train_with_MoS_repa.py`.

**Q: Are all wrapper scripts fully portable?**  
A: Not quite. Some newer one-click experiment scripts in `scripts/repa/` and `scripts/MoS_repa/` contain machine-specific Python interpreter paths. If they do not match your environment, edit them or use the direct Python entrypoints above.

**Q: Why does evaluation fail if I rename the images?**  
A: `evaluation/run_eval.py` extracts labels from the filename suffix `_class<id>.png`. If that suffix is missing, the evaluator cannot build the labeled NPZ correctly.

