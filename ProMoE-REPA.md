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
| `train_with_MoS_repa.py` | MoS-REPA, Multi-Align, Teacher-Affinity Routing, spectral responsibility, teacher-conditioned expert geometry, first-order denoising-regret routing, and cross-alignment variants |
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

Representative `train_with_MoS_repa.py` registry keys include:

- `ProMoE_TC_REPA_MoS_{B,L}`
- `ProMoE_TC_REPA_MoS_Naive_{B,L}`
- `ProMoE_TC_REPA_Multi_Align_B`
- `ProMoE_TC_REPA_Multi_Align_Affinity_B`
- `ProMoE_TC_REPA_Multi_Align_SRSR_B`
- `ProMoE_TC_REPA_Multi_Align_TCEG_B`
- `ProMoE_TC_REPA_Multi_Align_FDRR_B`

See `train_with_MoS_repa.py:model_dict` for the complete current registry.

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

MoS-REPA and Multi-Align use a different training path:

- the teacher encoder exposes features from all transformer blocks
- standard models return `(pred, mos_repa_loss)`
- Teacher-Affinity Multi-Align returns `(pred, mos_repa_loss, teacher_affinity_loss)`
- SRSR Multi-Align returns `(pred, mos_repa_loss, spectral_responsibility_loss)`
- TCEG Multi-Align returns `(pred, mos_repa_loss, expert_geometry_loss)`
- FDRR Multi-Align returns `(pred, mos_repa_loss, denoising_regret_loss)`
- the outer loop applies `mos_repa_loss * proj_coeff` and the configured third-loss coefficient when present

MoS variants route among teacher blocks, making them block-to-block alignment methods. Multi-Align, Teacher-Affinity Multi-Align, SRSR, TCEG, and FDRR share the trainer but align selected DiT blocks only with the teacher's last layer.

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
- `Multi-Align`: aligns several DiT blocks with the teacher's last layer using per-token dynamic coefficients.
- `Teacher-Affinity Routing`: keeps Multi-Align and adds a parameter-free training loss that matches pooled DINO patch affinities to one MoE router's soft co-assignment affinities for conditional samples.
- `SRSR`: keeps Multi-Align and adds a training-only responsibility loss at one MoE block. The existing shared branch is aligned with a fixed low-pass DINO target and the routed branch with the complementary high-pass residual; a reverse flag provides the causal control.
- `TCEG`: keeps Multi-Align and, for conditional samples at one aligned top-1 MoE block, groups raw routed-expert outputs and frozen teacher tokens by the detached expert assignment. It matches the unique off-diagonal entries of their independently centered, normalized centroid Gram matrices. A fixed spatial roll of teacher tokens is the negative control.
- `FDRR`: keeps Multi-Align and, after a warm-up, periodically samples conditional tokens at one top-1 MoE block. It holds the selected route weight fixed, estimates the diffusion-MSE change for a runner-up or random equal-compute challenger from the suffix gradient, keeps the highest-confidence half, and applies pairwise BCE only to the existing `cluster_centers`. Its inner diffusion-MSE gradient query suppresses `AddAuxiliaryLoss` injection; the normal outer backward still includes routing contrastive gradients. A within-image roll of the utility labels is the matched negative control; evaluation adds no parameters or FLOPs. Launch remains blocked until the canonical probe-v4 10K gate and the separate Base-shape overhead/memory gate both pass.

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
conda create -n fid_eval python=3.9 -y
conda activate fid_eval
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

For an automated fresh-server setup, `preprocess/prepare_imagenet.sh` downloads full-resolution ImageNet-1K (HuggingFace→ModelScope), materialises it to `/lustre01/yujie/dataset/imagenet/train/`, and VAE-encodes it (see CLAUDE.md "Dataset auto-preparation"). It materialises **raw JPEGs plus latents**, so REPA training — which also needs raw images for teacher features — works from the same prepared tree.

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
| `teacher_affinity_coeff` | Global coefficient applied to Teacher-Affinity Routing loss; `0` disables its contribution |
| `spectral_responsibility_coeff` | Global coefficient applied to the SRSR branch-responsibility loss |
| `expert_geometry_coeff` | Global coefficient applied to the TCEG centroid-geometry loss |
| `denoising_regret_coeff` | Global coefficient applied to the FDRR pairwise router loss |

`teacher_affinity_coeff`, `spectral_responsibility_coeff`, `expert_geometry_coeff`, and `denoising_regret_coeff` are separate experiment arms; `train_with_MoS_repa.py` rejects configurations with more than one positive coefficient.

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
| `router_hidden_dim` | `MoS_Naive`, `Multi-Align`, `Teacher-Affinity`, `SRSR`, `TCEG`, `FDRR` | Hidden width of the transformer router or alignment coefficient predictor |
| `num_router_blocks` | `MoS_Naive`, `Multi-Align`, `Teacher-Affinity`, `SRSR`, `TCEG`, `FDRR` | Number of transformer blocks in the router or alignment coefficient predictor |
| `router_num_heads` | `MoS_Naive`, `Multi-Align`, `Teacher-Affinity`, `SRSR`, `TCEG`, `FDRR` | Attention heads in the router or alignment coefficient predictor |
| `align_blocks` | `Multi-Align`, `Teacher-Affinity`, `SRSR`, `TCEG`, `FDRR` | Zero-based DiT blocks aligned with the teacher's last layer |
| `teacher_affinity_block` | `Teacher-Affinity` | Zero-based MoE block whose router receives affinity supervision |
| `teacher_affinity_grid_size` | `Teacher-Affinity` | Side length used to pool teacher tokens and soft routing probabilities |
| `teacher_affinity_router_temperature` | `Teacher-Affinity` | Temperature for soft assignments to existing cluster centers |
| `teacher_affinity_relation_temperature` | `Teacher-Affinity` | Temperature for row-wise teacher/router affinity distributions |
| `teacher_affinity_eps` | `Teacher-Affinity` | Numerical epsilon for spatial teacher-feature normalization |
| `spectral_responsibility_block` | `SRSR` | Zero-based aligned MoE block whose shared and routed outputs receive separate targets |
| `spectral_responsibility_reverse` | `SRSR` | Swap the low/high targets to form the reversed causal control |
| `spectral_residual_min_ratio` | `SRSR` | Mask high-pass tokens below this fraction of each image's mean residual norm |
| `spectral_responsibility_eps` | `SRSR` | Numerical epsilon for spatial teacher-feature normalization |
| `expert_geometry_block` | `TCEG` | Zero-based aligned top-1 MoE block whose routed-expert outputs define student centroids |
| `expert_geometry_min_tokens` | `TCEG` | Minimum assigned tokens required for an expert centroid within one conditional image |
| `expert_geometry_min_experts` | `TCEG` | Minimum valid expert centroids required to form a geometry loss for one image |
| `expert_geometry_teacher_roll` | `TCEG` | Two integer shifts on the square teacher-token grid; `[0, 0]` is TCEG and `[7, 11]` is the fixed 16x16 negative control |
| `expert_geometry_eps` | `TCEG` | Numerical epsilon for informative-centroid filtering and normalization |
| `denoising_regret_block` | `FDRR` | Zero-based top-1 MoE block whose existing router receives regret supervision |
| `denoising_regret_probe_interval` | `FDRR` | Training-step interval between sparse regret probes |
| `denoising_regret_token_ratio` | `FDRR` | Fraction of conditional tokens sampled per probed image |
| `denoising_regret_candidate_mode` | `FDRR` | Challenger policy: `runner-up`, `random`, or alternating `mixed` probe slots |
| `denoising_regret_confidence_quantile` | `FDRR` | Quantile of absolute normalized first-order change below which labels are discarded |
| `denoising_regret_temperature` | `FDRR` | Temperature applied to current-versus-challenger router-score margins in pairwise BCE |
| `denoising_regret_warmup_steps` | `FDRR` | First step at which regret probes may become active |
| `denoising_regret_ramp_steps` | `FDRR` | Linear ramp duration for the model-returned regret loss |
| `denoising_regret_label_roll` | `FDRR` | Within-image shift of utility labels; `0` is the positive arm and `1` is the matched roll control |
| `denoising_regret_seed` | `FDRR` | Base seed for the dedicated step/rank-local probe generator |
| `denoising_regret_eps` | `FDRR` | Numerical epsilon for normalized utility and confidence filtering |

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

**Q: Does `global_seed` control MoS-REPA training randomness?**

A: `train_with_MoS_repa.py` uses `global_seed * world_size + rank` for Python, NumPy, CPU/CUDA Torch, and passes `global_seed` to `DistributedSampler`. Matching experiments with the same world size and `global_seed` therefore use paired rank-specific random streams. This is not a cross-hardware bitwise-deterministic mode, and checkpoints do not restore RNG state. The sequential wrapper restarts the process after intermediate evaluation, so comparison arms must use the same phase boundaries; significant results still require multiple seeds.

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
