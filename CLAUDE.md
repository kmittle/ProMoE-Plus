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

### Unit Tests
There are ~40 `unittest` modules (no pytest config, no top-level `tests/` dir) — they
cover the analysis probes/gates, the evaluator, and the DINO route table contract. Run them
from the **repo root** so package imports resolve, using an interpreter that has torch:

```bash
# One module (works for any test home — plain namespace-package path)
python -m unittest analyses.routing_metric.test_phase_metric -v
# One package's suite — `discover -s` requires the start dir to have __init__.py
python -m unittest discover -s analyses/timestep_utility -t .
```
Test homes with `__init__.py` (so `discover -s <dir> -t .` works):
`analyses/{denoising_regret,dino_utility_neighborhood,expert_function,expert_update_budget,phase_default,routing_metric,routing_translation,timestep_utility}/`.
`evaluation/`, `models/` and `preprocess/` hold `test_*.py` but have **no** `__init__.py`, so
`discover -s` there fails with "Start directory is not importable" — name those modules
directly instead (`python -m unittest models.test_models_ProMoE_TC_dino_route`).
`evaluation/test_evaluator_device.py` additionally imports TensorFlow, so it only runs in
the `fid_eval` env. New analysis helper subpackages are expected to ship their own `test_*.py`.

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
| `scripts/MoS_repa/` | MoS, MoS Naive, MoS Naive Choice (block-range sweep, Sep, Blockwise, PerBlock, Fused, Shared-Align, RMSNorm, No-Coeff, proj_coeff sweep), Multi-Align (including TAR, SRSR, TCEG, and FDRR), Cross-Attention MoS, L/XL scale-up |
| `scripts/hierar/` | Hierarchical, Heterogeneous-Expert, NoPenalty, Expert-REPA-Dyna |
| `scripts/mae_align/` | MAE alignment, MAE alignment with projection |
| `scripts/noise_expert/` | Noise expert, Noise expert proj, EMA on noise/shared |
| `scripts/expert_contra/` | Expert contrastive output/param |
| `scripts/expert_choice/` | EC-BC batch-flatten Expert-Choice routing (Base) |
| `scripts/proto_t/` | Proto-T timestep-conditioned prototype: TC and EC-BC, each in residual / direct mode |
| `scripts/structured_batch/` | Structured batch sampling routing ablation: TC and EC-BC |
| `scripts/anchor/` | Anchor (R3-VAE reference-vector removal) routing: `routing` / `replace` modes |
| `scripts/proto_choice/` | Prototype-choice contrastive ratio sweep (`083` / `125`) |
| `scripts/lbcontra/` | Load-balance-aware routing contrastive loss: `reweight` (β), `logit_adjust` (τ), `balance_term` (λ), `soft_only` |
| `scripts/dagfuse/` | DAG-MoE shared↔conditional fusion: `cond_from_shared` / `shared_from_cond` / `bidirectional` |
| `scripts/adepth/` | Adaptive routed-FFN depth (MoD-style) `fixed_q` quota sweep (`q0p1`…`q0p4`) |
| `scripts/lossfree/` | Loss-Free Balancing bias, `bias_update_rate` sweep (`u1e4` / `u1e3` / `u1e2`) |
| `scripts/lsreg/` | Label-smoothing regularization on routing contrastive: `fixed` (ε sweep), `dyn{both,over,under}`, `diag_idea1`/`diag_inv` (diagonal correction, strength sweep) |
| `scripts/dagfuse_shared/` | Shared-expert augmentation: `dense` (prev-Dense-block source), `densenet` (all prev-MoE shared), `sharedroute` (router-selected prev-MoE, top1/top2), `region` (Block-AttnRes; `shared`/`resid` attach × `dag`/`softmax`) |
| `scripts/capacity_combo/` | Capacity-aware expert-responsibility study over **H** (hetero experts), **R** (token-count LS-Reg), **O** (expert-output regularizer), **P** (expert-param regularizer). Only the full combination `HROP` ran, and it failed the 300K gate; the nine never-launched arms (`H`, `HO`, `HP`, `HR`, `HOP`, `HRO`, `HRP`, `HO_norm`, `HROP_norm`) `run_capacity_combo_queue.sh` and the alternative loss variants that only those arms or unlaunched drafts used (normalized output view, cosine/joint signatures, capacity-aware LS-Reg) were deleted on 2026-09-14 pending a redesign. Also holds the `hrop_gate_then_q0p4.sh` supervisor and `capacity_combo_eval_helpers.sh` |
| `scripts/dino_route/` | DINO-assisted load-aware routing: `uncertainty` (s0 / s0p02) and `margin_gate` (v1 / v2-corrected) arms, plus their label-shuffled `permutation_seed` controls |
| `scripts/phase_metric/` | Phase-conditioned routing-metric arms: `base_s0` control, `phase_metric`, and a timestep-shuffled control |
| `scripts/fdrr/` | Teacher-free Base-FDRR (`ProMoE_TC_B_FDRR`) plus its seed-0 control |
| `scripts/ecmix/` | Expert-Choice mixed into TC training (`ProMoE_TC_B_ecmix`): series 1 blends the TC/EC routing-contrastive loss (`loss_w0p05` / `w0p10` / `w0p25` / `w0p50` / `w1p00`, `loss_cos0p50`), series 2 turns a share of training steps into complete EC-BC steps (`step_p0p05` / `p0p10` / `p0p25` / `p0p50`, `step_cos0p50`). 2 GPUs per arm, sampled and evaluated at 500K only; `PROMOE_RESUME=1` continues an interrupted run. Also holds `ecmix_eval_helpers.sh` |
| `scripts/global_center/` | Global-batch routing-contrastive class centers (`ProMoE_TC_B_global_center`), an engineering ablation: each expert's token sum and count are all-reduced so every GPU contrasts the prototypes with the same class centers. 4 GPUs like the fresh baseline, evaluated at 300K and 500K without an automatic stop; `PROMOE_RESUME=1` continues an interrupted run. Also holds `global_center_eval_helpers.sh` |

```bash
# Example: run a MoS experiment end-to-end
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_b3_5_train_sample_eval.sh
```

### VAE Latent Preprocessing (speeds up training)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python preprocess/preprocess_vae.py \
  --latent_save_root "/path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz"
```
`preprocess_vae.py` now **skips images whose `.latent.npz` already exists** (`--skip-existing`, default on) and writes latents **atomically** (`.part` + rename), so an interrupted run only re-encodes the remainder.

### Dataset auto-preparation (run on a fresh server)
`preprocess/prepare_imagenet.py` (+ its `prepare_imagenet.sh` wrapper) makes an experiment runnable on a server that has no local ImageNet: it **downloads full-resolution ImageNet-1K** (HuggingFace first, ModelScope fallback — `ILSVRC/imagenet-1k`, 294 train shards), **materialises** it to `/lustre01/yujie/dataset/imagenet/train/<label:04d>/*.JPEG` (zero-padded label dirs so `CustomImageFolder`'s sorted-dir label == canonical ImageNet label), provisions the SD-VAE, then VAE-encodes everything to `/lustre01/yujie/dataset/imagenet/sd-vae-ft-mse_Latents_256img_npz/`. Idempotent + resume-safe (per-shard + per-file skip, stage sentinels under `/lustre01/yujie/dataset/imagenet/.state/`), cross-process `flock`-locked (two 4-GPU slots on one server won't race), and it **verifies every image has a matching latent** before finishing (a missing latent is silently zero-filled by the training loader, so this gate is mandatory). **Run it once manually (for the shared dataset location), before launching a batch** — it is intentionally NOT wired into the experiment scripts, because two co-scheduled 4-GPU slots (GPU 0-3 and 4-7) would otherwise both trigger download/encode; even with the `flock` (which the tool keeps as a safety net) the second slot would idle-block through the first slot's multi-hour download. Preparing once up front lets both slots find the data ready and start training immediately. It sets `PROMOE_DATA_PATH=/lustre01/yujie/dataset/imagenet/train` internally for its own preprocess step; training reads the same path via `config.py`'s default. Source repo ids are overridable via `PROMOE_HF_DATASET` / `PROMOE_MS_DATASET`, and the gated HF path needs `HF_TOKEN`. The dataset lives outside the repo (`/lustre01/...`), so there is nothing to commit; keep any `PROMOE_DATA_PATH` override `train`-once-safe (no path component but `train/` may contain the substring 'train').
```bash
bash preprocess/prepare_imagenet.sh   # python + 8 GPUs are hardcoded defaults; override with --python / --gpus
```

**Parquet-direct mode (already-downloaded data, no JPEG intermediate).** If the raw HF parquet shards are already present on the server (default `/lustre01/qianyuan/data/ILSVRC/imagenet-1k/data`, override with `PROMOE_PARQUET_DIR` or `--parquet-dir`), `prepare_imagenet.py` **auto-detects** them and encodes VAE latents **directly from parquet** via `preprocess/encode_latents_from_parquet.py` — skipping both the re-download and the ~140GB intermediate JPEG folder. Output is `<latent_root>/<label:04d>/<name>.latent.npz` in the **same** 8-channel `vae.encode(x).latent_dist.parameters` format as `preprocess_vae.py`. Training then reads these directly via the **`LatentFolder`** dataset (opt-in per config: `use_encoded_latents: True`; path from `cfg.latent_data_path` / `PROMOE_LATENT_PATH`) — no image folder and no `str.replace('train', ...)` derivation. Numeric class directories map with `int(<label:04d>)`; copied standard ImageNet synset directories map by the same sorted class order as `ImageFolder`. Latents-only mode is for the non-REPA `train.py` families (the 2026_07_01 batch sets the flag); REPA training still needs raw images, i.e. the JPEG path. The same `prepare_imagenet.sh` command runs whichever mode matches the server (parquet-direct if the shards are there, else download+materialise). `encode_latents_from_parquet.py` is shard-parallel across GPUs and per-file resume-safe.

### Evaluation (separate conda env with TensorFlow)
```bash
conda create -n fid_eval python=3.9 -y && conda activate fid_eval
cd evaluation && pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# The default auto mode uses GPU only when TensorFlow covers its architecture.
CUDA_VISIBLE_DEVICES=0 python run_eval.py /path/to/generated/images
# Pack PNGs to NPZ without running evaluator
python run_eval.py /path/to/generated/images --count 50000 --no-eval
```

`run_eval.py` requires contiguous `img<index>_class<label>.png` names starting
at zero, packs exactly `--count` images even when distributed sampling rounds
the directory up to a larger batch multiple, and propagates evaluator failures
as nonzero exits. Its paths are script-relative, so invoking it from the
repository root also works. The pinned historical TensorFlow build preserves
the prior metric stack and automatically falls back to CPU on newer
incompatible GPU architectures in `--eval-device auto`.

## Architecture

### Configuration System
- `config.py`: Global defaults using EasyDict. Defines base model configs (`DiT_S_config` through `DiT_XL_config`) and MoE-specific configs (`DiffMoE_DiT_*`, `TCDiT_*`, `ECDiT_*`).
- `configs/*.yaml`: Per-experiment overrides deep-merged onto `config.py` defaults at runtime via `deep_update()` in `utils.py`.
- **Config merging flow**: ProMoE models reuse base DiT configs (e.g., `ProMoE_TC_L` maps to `DiT_L_config` in `model_dict`). The YAML adds `MoE_config` as a nested dict under the base config key (e.g., `DiT_L_config.MoE_config`), which `deep_update()` merges in. This means MoE parameters are not in `config.py` for ProMoE — they come entirely from YAML.
- The YAML filename (minus extension) becomes `custom_cfg_name`, which determines the output subdirectory: `outputs/{model_name}/{custom_cfg_name}/` containing `checkpoints/`, `training.log`, `sample.log`, `tensorboard/`, and `sample/step{N}/`.

### Model Registry
`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, and `train_with_mae.py` each define a `model_dict` mapping `model_name` strings to `(ModelClass, config_key)` pairs. `sample.py` merges all four dicts so it can sample from any model variant. Adding a new model requires an entry in the appropriate training script's `model_dict`. Note: `train.py` hosts most model families (base DiT, baselines, ProMoE-TC/EC, ProMoE_EC_BC batch-choice, ProMoE_EC_BC_hetero, proto_t timestep-conditioned-prototype (TC + EC-BC), anchor (R3-VAE), proto_choice (prototype-choice contrastive), lbcontra (load-balance-aware routing contrastive), dagfuse (DAG-MoE shared↔cond fusion), dagfuse_shared (shared-expert augmentation: dense/densenet/sharedroute/region), adepth (adaptive routed-FFN depth), lossfree (loss-free balancing bias), teacher-free Base-FDRR (`ProMoE_TC_B_FDRR`), lsreg (routing-contrastive label smoothing), capacity_combo (H/R/O/P), dino_route (DINO class-uncertainty load-aware routing), ecmix (Expert-Choice loss/steps mixed into TC training), global_center (global-batch routing-contrastive class centers), noise expert variants, expert contrastive); `train_with_MoS_repa.py` hosts MoS, MoS Naive, MoS Naive Choice (B/L/XL), MoS Naive Choice Sep, MoS Naive Choice Fused, MoS Naive Choice Blockwise, MoS Choice PerBlock, Multi-Align, Teacher-Affinity Multi-Align, SRSR Multi-Align, TCEG Multi-Align, FDRR Multi-Align (`ProMoE_TC_REPA_Multi_Align_FDRR_B`), and Cross-Attention variants (both standard REPA and MoS); `train_with_mae.py` only hosts group_align models.

### Model Hierarchy (in `models/`)
All model files follow the `models_*.py` naming convention. Key layers:

- **`modules.py`** — Shared building blocks: `Attention`, `PatchEmbed`, `TimestepEmbedder`, `LabelEmbedder`, `FinalLayer`, `MLP`/`Mlp`, `SwiGLU`, `MoeMLP`, `PrototypeMLP` (timestep-conditioned prototype generator for proto_t variants).
- **`models_DiT.py`** — Dense DiT baseline. `DiTBlock` uses AdaLN-Zero modulation (6-param per-sample conditioning from timestep+class). All ProMoE variants inherit this block structure.
- **Baselines**: `models_TCDiT.py` (Token-Choice MoE), `models_ECDiT.py` (Expert-Choice MoE), `models_DiffMoE.py` (capacity prediction).
- **`models_ProMoE_TC.py`** — Main proposed model. `SparseMoeBlock` implements two-step routing: (1) conditional routing separates uncond tokens (class=1000) to a dedicated expert, (2) prototypical routing assigns cond tokens via cosine similarity to learnable `cluster_centers`. Includes routing contrastive loss via `AddAuxiliaryLoss` autograd trick. `models_ProMoE_EC.py` is the Expert-Choice variant (recommended for DDPM).

**Variant families** (all extend ProMoE-TC):

| Family | Files | Training script | Key difference |
|--------|-------|----------------|----------------|
| Base FDRR | `_denoising_regret.py` | `train.py` | Teacher-free sparse first-order diffusion-MSE utility labels train one existing router's `cluster_centers`; no parameter/buffer or eval-path change. Training returns `(prediction, denoising_regret_loss)`, while eval returns the base tensor. |
| REPA | `_repa.py`, `_repa_shared.py`, `_repa_cond.py` | `train_with_repa.py` | MLP projectors align DiT features with frozen DINOv2 teacher |
| Dynamic REPA | `_repa_dyna.py`, `_repa_dyna_scale.py`, `_repa_dyna_select.py`, `_repa_dyna_only.py` | `train_with_repa.py` | Timestep-dependent REPA loss weighting/selection |
| Router REPA | `_repa_router.py`, `_repa_router_contra.py`, `_repa_routed.py`, `_repa_double_share.py` | `train_with_repa.py` | Alignment at router/prototype level instead of block output |
| MoS REPA | `_repa_MoS.py`, `_MoS_naive.py`, `_MoS_naive_choice.py`, `_MoS_naive_choice_.py` (sep), `_MoS_naive_choice_blockwise.py`, `_MoS_choice_per_block.py` | `train_with_MoS_repa.py` | `BlockRouter` selects which teacher blocks to align with per-token. Config ablation flags: `router_norm_type` (`"layernorm"` default / `"rmsnorm"`), `align_target` (`"block_output"` default / `"shared_expert"` — falls back to block output for dense blocks) |
| Fused MoS | `_MoS_naive_choice_fused.py` | `train_with_MoS_repa.py` | Fuses top-k teacher blocks via routing weights before alignment; shared Transformer+sigmoid `CoeffPredictor` (input: projected student features `z_proj` + conditioning) predicts per-token loss weight |
| Multi-Align | `_repa_multi_align.py`, `_repa_multi_align_affinity.py`, `_repa_multi_align_spectral.py`, `_repa_multi_align_expert_geometry.py`, `_repa_multi_align_denoising_regret.py` | `train_with_MoS_repa.py` | Per-token sigmoid coefficients from `AlignCoefficientPredictor`. Config ablation flag: `use_dynamic_coeff` (`True` default / `False` removes predictor, uses uniform weighting). TAR matches teacher/router affinities; SRSR assigns low/high teacher components to shared/routed branches; TCEG matches routed-expert/teacher centroid geometry; FDRR sparsely uses first-order diffusion-MSE utility labels to train one existing router's `cluster_centers`, with a within-image label roll as its control. |
| Hierarchical | `_hierar.py` | `train.py` | Sub-prototype hierarchical routing (`num_sub_prototype` sub-prototypes per expert, max-over-sub-prototype cos-sim, + sub-prototype diversity loss). Homogeneous experts. |
| Heterogeneous-Expert | `_hetero_expert.py`, `_hetero_expert_repa_dyna.py` | `train.py`, `train_with_repa.py` | Heterogeneous routed-expert widths (1x→3x linear ladder, mean 2x) + optional soft `cost_penalty` FLOPs regularizer. Token-choice routing unchanged. `_repa_dyna` sibling adds dynamic-REPA alignment. |
| Noise Expert | `_noise_expert.py`, `_noise_expert_proj.py`, `_noise_expert_ema.py` | `train.py` | Dedicated noise-level expert; EMA variant calls `update_noise_expert_ema()` after each optimizer step |
| Expert Contrastive | `_expert_contra.py` | `train.py` | Pairwise L2 repulsion on expert outputs or params |
| Group Align | `_group_align.py`, `_group_align_proj.py` | `train_with_mae.py` | Group alignment without REPA |
| Cross-Attention | `_repa_cross_global_pre.py`, `_repa_cross_global_block.py`, `_repa_cross_expert_local.py`, `_repa_cross_proto.py` | `train_with_MoS_repa.py` | Inter-token attention-weighted REPA alignment at different positions (pre-MoE, block, expert-local, prototype) |
| Cross-Attention MoS | `_repa_MoS_naive_choice_cross_global_pre.py`, `..._block.py`, `..._expert_local.py`, `..._proto.py` | `train_with_MoS_repa.py` | MoS + cross-attention alignment (combines block router teacher selection with cross-alignment) |
| Batch-Choice EC | `_EC_batch_choice.py`, `_EC_batch_choice_proto_t.py` | `train.py` | Expert-Choice over the batch-flattened cond-token pool: each expert picks top-k from `B_cond*S` tokens (capacity `k = B_cond*S/E * top_k`) rather than per-image (cf. `models_ProMoE_EC.py`). Dispatch via `torch.gather`/`index_add_`. Keys `ProMoE_EC_BC_B` / `ProMoE_EC_BC_B_proto_t`; forward() returns a plain tensor (`AddAuxiliaryLoss`). EC-family variants follow `models_ProMoE_EC_*.py` and inherit from `ProMoE_EC`, not `ProMoE_TC`. |
| Proto-T (timestep prototype) | `_proto_t.py` (TC), `_EC_batch_choice_proto_t.py` (EC-BC) | `train.py` | Replaces static `cluster_centers` with a per-sample timestep-conditioned prototype from `PrototypeMLP` (one per MoE block); the MoE block forward gains a `t_emb` arg so cos-sim routing runs in a noise-level-aligned space. Config flag `proto_t_update_mode` (default `"residual"` / `"direct"`). Keys `ProMoE_TC_B_proto_t` / `ProMoE_EC_BC_B_proto_t`; forward() returns a plain tensor. |
| Anchor (R3-VAE) | `_anchor.py` | `train.py` | Self-contained `ProMoE_TC` copy; each MoE block holds a learnable `anchor` vector `r`. Cond tokens are orthogonally projected off `r` (`x - ((x·r)/‖r‖²)·r`, R3-VAE Eq 1-2) before prototype cos-sim. Config flag `anchor_apply_mode`: `"routing"` (default; only the router sees `x_new`, experts/shared see `x`) or `"replace"` (router + routed + shared experts all see `x_new`; uncond untouched; the DiTBlock residual carries original `x`). **Not step-0-identical to base ProMoE** (anchor is randomly init'd with no zero-gate) → trained fresh, base checkpoints are non-strict-loadable. Key `ProMoE_TC_B_anchor`; forward() returns a plain tensor (`AddAuxiliaryLoss`). |
| Proto-Choice (contrastive) | `_proto_choice.py` | `train.py` | Self-contained `ProMoE_TC` copy; only the routing-contrastive loss changes (routing/experts/dispatch unchanged). Switches from token-choice to **prototype-choice InfoNCE**: each prototype selects its top-K (`K = ceil(contrastive_proto_choice_ratio · N_cond)`) most cos-similar cond tokens as the positive (their mean), remaining cond tokens as negatives; averaged over prototypes, no prototype-prototype term, no empty-cluster skipping. Sweep configs `_083`/`_125`. Key `ProMoE_TC_B_proto_choice`; forward() returns a plain tensor (`AddAuxiliaryLoss`). |
| LB-Contra (load-balance-aware) | `_lbcontra.py` | `train.py` | Self-contained `ProMoE_TC` copy; only the routing-contrastive loss changes (routing/experts/dispatch unchanged), injecting each codeword's token count to fight expert imbalance over a differentiable soft-assignment base (`q_t=softmax(cos/τ_route)` over all K prototypes; soft counts `ñ` **per-GPU local**). Config `lb_contra_mode`: `soft_only` (base only), `reweight` (per-row InfoNCE weighted by `(ñ_i)^(-lb_reweight_beta)`), `logit_adjust` (add `lb_logit_adj_tau·log(ñ_j)` to candidate logits, balanced-softmax), `balance_term` (add `lb_balance_lambda`·load term). Step-0-identical to base (no new params). Key `ProMoE_TC_B_lbcontra`; forward() plain tensor (`AddAuxiliaryLoss`). |
| DAG-Fuse (shared↔cond fusion) | `_dagfuse.py` | `train.py` | Self-contained `ProMoE_TC` copy; DAG-MoE structural fusion (arXiv 2606.01062, Eq 7-11) between role nodes {C = routed/conditional output, S = shared output} via a K=2 gated-edge `DAGFuseModule` (FusedRMSNorm → down_proj d→`d_g=64` → combined_proj → GELU gate + value sum over keys → **zero-init up_proj** → residual, `fusion_num_iter`=1). Only cond tokens; uncond untouched; shared always re-added. Config `fusion_arm`: `cond_from_shared` / `shared_from_cond` / `bidirectional` (iso-param — the module is identical, direction is the only variable). up_proj zero-init ⇒ step-0-identical (non-strict loadable). Key `ProMoE_TC_B_dagfuse`; forward() plain tensor. |
| Adaptive-Depth (MoD-style) | `_adepth.py` | `train.py` | Self-contained `ProMoE_TC` copy; per-block `depth_gate=Linear(d,1)` scores each cond token, **batch-flattened**, then a fixed quota routes bottom-q to **skip** (0 routed FFN, real compute saving), middle to **normal** (1, =base), top-q to **deepen** (2nd pass through the *same* top-1 expert). `#skip==#deepen` ⇒ routed passes = N_cond (**compute-conserving**). Config `alloc_mode=fixed_q`, `depth_q` (quota, annealed over `depth_warmup`). STE keep-gate (skip) + zero-init `deepen_gain` + zero-init `depth_gate` ⇒ step-0-identical (non-strict); `assert top_k==1`; DDP fake-grad touch avoids unused-param crash. Key `ProMoE_TC_B_adepth`; forward() plain tensor. |
| Loss-Free Balancing | `_lossfree.py` | `train.py` | Self-contained `ProMoE_TC` copy; DeepSeek loss-free balancing (arXiv 2408.15664): a per-prototype `expert_bias` **buffer (non-Parameter, no grad)** is added to the cond-token cos-sim for **top-1 selection only**; expert output weights use the **unbiased** cos-sim (zero interference gradient — orthogonal to the contrastive loss, which is unchanged). After each forward, no-grad update `b_i += bias_update_rate·sign(mean_c − c_i)` with cross-GPU `all_reduce`. Config `use_lossfree_bias`, `bias_update_rate` (u-sweep). Step-0-identical to base (bias init 0), non-strict loadable. Key `ProMoE_TC_B_lossfree`; forward() plain tensor. |
| LS-Reg (label smoothing) | `_lsreg.py` | `train.py` | Self-contained `ProMoE_TC` copy; only the routing-contrastive loss changes. Two `ls_apply` modes: `"label"` smooths the InfoNCE soft target by ε (per-codeword load-dependent when `ls_mode` is dynamic, constant when `fixed`); `"diag"` adds a `.detach()`ed load-proportional offset directly on the similarity-matrix diagonal (idea-1: overloaded codewords pushed +, `ls_diag_sign=-1` inverts). Step-0-identical to base ProMoE (no new params). Key `ProMoE_TC_B_lsreg`; forward() plain tensor (`AddAuxiliaryLoss`). |
| DAG-Fuse Shared (shared-expert augment) | `_dagfuse_dense.py`, `_dagfuse_densenet.py`, `_dagfuse_sharedroute.py`, `_dagfuse_region.py` | `train.py` | Self-contained `ProMoE_TC` copies that augment each MoE block's **shared-expert output** via a zero-init gated `SharedAugmentModule` (`fuse_mech` `dag`/`softmax`, `fuse_dim` d_g=64). Source differs per idea: `dense` = previous Dense block's output (MoE block-entry `x`); `densenet` = all previous MoE blocks' raw shared outputs (forward-local list, not detached); `sharedroute` = per-block router picks top-k (`fuse_top_k` 1/2) previous-MoE shared outputs; `region` = Block-AttnRes over previous fixed-size regions (`region_size`=3), `region_attach` `shared` (augment shared out) or `resid` (zero-init attn-residual on the main stream — structurally-but-not-behaviorally routing-preserving). `fuse_apply` `none`(=base)/`cond`/`all`. up_proj zero-init ⇒ step-0-identical (non-strict loadable). Keys `ProMoE_TC_B_dagfuse_{dense,densenet,sharedroute,region}`; forward() plain tensor. |
| Capacity-Combo (H/R/O/P) | `_capacity_combo.py` | `train.py` | Extends `_expert_contra.py` (kept intact for historical runs) with `CapacityAwareSparseMoeBlock`. Four training-only factors switched on per config: **H** heterogeneous routed widths at the original mean intermediate size, **R** token-count LS-Reg diagonal offset (`ls_balance_mode: token`), **O** expert-**output** exp(-L2/τ) repulsion, **P** width-invariant expert-**param** signature repulsion. A block listed for both O and P scores them separately (`dual_additive`) and weights them (`expert_contrastive_output_lam` / `_param_lam`, independent temperatures). **Inference compute is unchanged.** Only the full `HROP` arm was run, and it failed the 300K gate. Key `ProMoE_TC_B_capacity_combo`. |
| DINO-Route (load-aware) | `_dino_route.py` | `train.py` | DINOv2 is used **offline only** — `preprocess/build_dino_route_table.py` produces one per-ImageNet-class uncertainty scalar; no teacher feature enters the backbone and nothing is feature-aligned. At train time an uncertain class gets a small **detached** preference for experts whose recent conditional load is low; expert output weights keep the original cosine value. The table is contract-checked (`preprocess/dino_route_table_contract.py`: `table_version` + `table_method` + sha256), so a legacy config can never silently consume a corrected table. Arms: `uncertainty` / `margin_gate`, each with a `permutation_seed` label-shuffled control. Key `ProMoE_TC_B_dino_route`. |
| EC-Mix (TC + EC training) | `_ecmix.py` | `train.py` | Only the MoE block inherits from the untouched conference model (`SparseMoeBlock(models_ProMoE_TC.SparseMoeBlock)` reuses its construction, TC routing, TC forward and TC contrastive loss); the DiT, DiTBlock and `AddAuxiliaryLoss` are written out in the file, and `ratio: 0` is bit-identical to `ProMoE_TC_B`. `ecmix_config.mode: loss` routes with TC every step and blends the routing-contrastive loss as (1−ratio)·TC + ratio·EC; `mode: step` makes a ratio share of training steps complete EC-BC steps (each expert takes the rank-local top `T/E` cond tokens; EC contrastive loss). Whether a step is an EC step is a pure function of the global step (train.py passes `training_step`); eval and sampling always run TC. Key `ProMoE_TC_B_ecmix`; forward() plain tensor (`AddAuxiliaryLoss`). |
| Global-Center (engineering) | `_global_center.py` | `train.py` | Only the MoE block inherits the untouched conference `SparseMoeBlock`, and it replaces just the routing-contrastive loss: the forward all-reduces each expert's token sum and count, so every rank contrasts the prototypes with the same global-batch class centers. Other ranks' sums enter as plain values (no gradient crosses processes); the gradient through the rank's own tokens is multiplied by the world size, so after DDP averages gradients the update equals one loss on the whole global batch. Token routing, forward, eval and sampling are unchanged, and a single process is bit-identical to `ProMoE_TC_B`. Key `ProMoE_TC_B_global_center`. |
| Phase-Metric (config flag on base) | `models/phase_metric.py` | `train.py` | **Not a separate model file** — `PhaseConditionedRoutingMetric` is imported by `models_ProMoE_TC.py` and enabled per-config via `MoE_config.phase_metric_config.enabled`. A bounded low-rank trilinear residual over the existing token↔prototype cosine affinity, conditioned on the scalar diffusion phase through Fourier bands. The phase projection is zero-init ⇒ step-0 routing and active-expert count are unchanged. `shuffle_timestep: True` is the built-in control arm. Model key stays `ProMoE_TC_B`. |
| Ablations | `_sigmoid.py`, `_symmetric.py` | `train.py` | Routing gating variants |

**REPA model forward() behavior**: Returns `(pred, zs_proj)` during training (eval returns only `pred`). The `_repa_shared.py` variant aligns shared expert output specifically — requires `encoder_depth` to point to a MoE block.

### Auxiliary Loss Convention
Model `forward()` returns either a plain tensor (DiT) or a tuple for models with auxiliary losses:
- **DiffMoE**: Returns `(pred, "Capacity_Pred", layer_idx_list, ones_list, pred_c_list, loss_weight)`. Training loop computes BCEWithLogitsLoss for capacity prediction.
- **ProMoE**: Uses `AddAuxiliaryLoss` autograd function to inject contrastive loss gradients directly into the forward pass — returns a plain tensor but the auxiliary loss gradient flows through automatically.
- **ProMoE-REPA**: Returns `(pred, zs_proj)` during training. The training loop in `train_with_repa.py` computes `compute_repa_loss(teacher_z, zs_proj)` and adds it weighted by `proj_coeff`. Total loss = MSE + REPA loss * `proj_coeff` + routing contrastive loss (via autograd).
- **ProMoE-MoS-REPA**: Returns `(pred, mos_repa_loss)` during training, where `mos_repa_loss` is a scalar computed inside the model (weighted cosine similarity across selected teacher blocks). The training loop in `train_with_MoS_repa.py` multiplies by `proj_coeff` (default 0.5). Total loss = MSE + mos_repa_loss * `proj_coeff` + routing contrastive loss (via autograd). Note: `teacher_all_z` (all teacher block features) is passed to forward; the model selects which teacher blocks to align with via its router.
- **ProMoE-Multi-Align**: Returns `(pred, repa_loss)` during training. Similar to MoS-REPA but aligns with teacher last layer only; `AlignCoefficientPredictor` produces per-token sigmoid coefficients that weight the alignment loss. When `use_dynamic_coeff=False`, the predictor is removed and alignment uses uniform weighting (plain mean of negative cosine similarity). The TAR, SRSR, TCEG, and FDRR siblings return a third `teacher_affinity_loss`, `spectral_responsibility_loss`, `expert_geometry_loss`, or `denoising_regret_loss`. The outer loop dispatches that value by registered family and weights it with top-level `teacher_affinity_coeff`, `spectral_responsibility_coeff`, `expert_geometry_coeff`, or `denoising_regret_coeff`; at most one may be positive in an experiment arm. All are trained via `train_with_MoS_repa.py`. FDRR model-level controls are the `denoising_regret_*` fields for block, probe interval/token ratio, challenger mode, confidence quantile, temperature, warm-up/ramp, label roll, seed, and epsilon. FDRR suppresses `AddAuxiliaryLoss` only during its inner diffusion-MSE gradient query; normal outer backward keeps routing-contrastive gradients. Its official launch gate uses probe v4, the exact six-case `fdrr_gate_v1` manifest, a Base Multi-Align checkpoint at step >=10K, and matching loaded/canonical checkpoint steps; the separate Base-shape overhead/memory gate is still mandatory.
- **ProMoE-Fused-MoS-REPA**: Returns `(pred, mos_repa_loss)` during training. Like MoS-REPA but fuses top-k teacher block features (weighted sum using routing weights, no re-normalization) before computing cosine similarity. A shared `CoeffPredictor` (Transformer + sigmoid, input = projected student features `z_proj` + conditioning `c`) predicts per-token loss weight. Trained via `train_with_MoS_repa.py`.
- **ProMoE-Cross-Attention**: Returns `(pred, cross_align_loss)` during training. The cross-alignment loss uses attention weights (global, block-level, expert-local, or prototype-based) to weight the cosine similarity between student projections and teacher features. Computed inside the model via `compute_cross_align_loss()`. Trained via `train_with_MoS_repa.py` with same `proj_coeff` weighting.
- **ProMoE-EC-BC / proto_t**: Both keep the `AddAuxiliaryLoss` plain-tensor return convention. proto_t additionally threads the timestep embedding through every block (`block(x, c, labels, t)` → MoE `forward(..., t_emb)`) so prototypes are regenerated per-timestep; preserve this `t_emb` arg in any proto_t-derived variant.

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
- Iterates `named_parameters()` each step for grad-norm stats grouped by param-name substring (attn_modules / projectors / block_router / coeff_predictor / capacity_predictor / cluster_centers / shared_expert / moe_experts / backbone). Frozen params (`requires_grad=False`, e.g. `pos_embed`, `noise_expert_ema` params) are skipped.
- `cluster_centers` is a parameter inside each `SparseMoeBlock` (NOT on the top-level DiT), so stats are aggregated via `named_parameters()` traversal, not a top-level attribute lookup.
- `ExpertLocalAttention` uses masked softmax + `nan_to_num(0)`, which produces fully-zero rows for uncond tokens (labels==1000) by design (~10% under CFG). The attention hook filters to "active rows" (`row_sum > 1e-4`) before computing min/max, and reports `inactive_frac` as a benign diagnostic instead of an alert.
- Dense DiT / non-MoE / non-REPA models degrade gracefully — no hooks are installed, only backbone grad norms are reported.
- Every stat path is wrapped in try/except so a monitoring bug cannot take the training run down.

### Key MoE Parameters (in YAML `MoE_config`)
Core parameters: `num_routed_experts` (typically 12), `top_k` (experts per token, default 1), `use_shared_expert`/`use_uncond_expert`, `interleave` (alternate MoE/dense layers).

- `proto_t_update_mode` (proto_t variants only): `"residual"` (default; `prototype_t = cc + MLP(concat(cc, t_emb))`, fc2 zero-init so step 0 == base ProMoE) or `"direct"` (`proto_proj(cc) + MLP(...)`, identity-init `proto_proj`, adds ~3.5M params). Both are step-0-identical to base ProMoE.
- `anchor_apply_mode` (anchor variant only): `"routing"` (default; anchor-removed tokens feed only the router) or `"replace"` (anchor-removed tokens feed router + routed + shared experts). The per-block `anchor` parameter is random-init'd (no zero-gate), so this variant is **not** step-0-identical to base ProMoE — train fresh, don't resume from base ProMoE checkpoints. `anchor_eps` (default `1e-6`) stabilizes the `‖r‖²` denominator.
- `contrastive_proto_choice_ratio` (proto_choice variant only, default `0.1`): fraction of cond tokens each prototype claims as its positive set, `K = ceil(ratio · N_cond)`. Sweep configs use `0.083`/`0.125` (filename suffixes `083`/`125`). Step-0-identical to base ProMoE (only the contrastive loss formulation differs).
- `lb_contra_mode` (lbcontra variant only): `soft_only` | `reweight` (param `lb_reweight_beta`, e.g. 0.25/0.5/1/2) | `logit_adjust` (param `lb_logit_adj_tau`, e.g. 0.5/1/2/4) | `balance_term` (param `lb_balance_lambda`, e.g. 0.001…1). Only the routing-contrastive loss changes; step-0-identical to base ProMoE.
- `fusion_arm` (dagfuse variant only): `cond_from_shared` (default) | `shared_from_cond` | `bidirectional`; `fusion_num_iter` (default `1`) is the number of gated-edge iterations. up_proj is zero-init so step-0-identical to base ProMoE (non-strict loadable, adds `DAGFuseModule` params).
- `alloc_mode` / `depth_q` / `depth_warmup` (adepth variant only): `alloc_mode=fixed_q`; `depth_q` is the skip=deepen quota fraction (e.g. 0.1…0.4, annealed from 0 over `depth_warmup` steps, default 5000). `depth_gate`+`deepen_gain` zero-init ⇒ step-0-identical (non-strict). Requires `top_k==1`.
- `use_lossfree_bias` / `bias_update_rate` (lossfree variant only): `use_lossfree_bias` (default `False`) enables the per-prototype `expert_bias` buffer on top-1 selection; `bias_update_rate` is the DeepSeek update step `u` (sweep `1e-4`/`1e-3`/`1e-2`, paper-best ≈`1e-3`). Buffer is non-trainable (no grad), step-0-identical to base ProMoE (non-strict loadable).
- `ls_mode` / `ls_apply` (lsreg variant only): `ls_apply` `"label"` (default; soft-target label smoothing on the InfoNCE) with `ls_mode` `off`/`fixed`(const ε=`ls_eps_base`)/`dyn_both`/`dyn_under`/`dyn_over` (load-dependent ε from `ls_slope`, `ls_eps_cap`, `ls_load_map` `linear`/`invsqrt`, `ls_warmup`), or `ls_apply` `"diag"` (idea-1: `.detach()`ed load-proportional offset on the similarity diagonal, `ls_diag_strength` sweep, `ls_diag_sign` `+1` original / `-1` inverse). `train.py` logs realized mean ε to TensorBoard (`lsreg/mean_eps`). Step-0-identical to base ProMoE (no new params).
- `fuse_apply` / `fuse_mech` / `fuse_dim` (dagfuse_shared variants only): `fuse_apply` `none`(=base)/`cond`/`all` gates which tokens the shared-output augmentation touches; `fuse_mech` `dag`/`softmax` picks the `SharedAugmentModule` combiner; `fuse_dim` (d_g, default 64) is the gating bottleneck. `sharedroute` adds `fuse_top_k` (1/2); `region` adds `region_size` (default 3) and `region_attach` (`shared`/`resid`). up_proj zero-init ⇒ step-0-identical to base ProMoE (non-strict loadable).
- `phase_metric_config` (any `ProMoE_TC`-derived model): `enabled` (default `False`), `rank` (8), `num_fourier_bands` (4), `num_train_timesteps` (1000), `scale` (0.25), `init_seed` (1729), plus `shuffle_timestep` (control arm that permutes the per-sample phase within the batch). Zero-init phase projection ⇒ step-0-identical to base ProMoE.
- `dino_route_config` (dino_route variant only): `enabled`, `mapping` (`correct` vs the legacy arm), `table_path` + `table_version` + `table_method` (the three must satisfy `preprocess/dino_route_table_contract.SUPPORTED_TABLE_CONTRACTS`; declaring one of version/method without the other is an error, and omitting both pins the legacy contract), `num_classes`, `strength` (e.g. 0.08), `ema_decay` (0.99, recent conditional-load EMA), `permutation_seed` (label-shuffled control).
- `ecmix_config` (ecmix variant only): `mode` (`loss` / `step`), `schedule` (`constant` / `cosine`), `ratio` (EC loss weight or EC-step share; the start value of the cosine schedule), `anneal_steps` (cosine only; the ratio reaches 0 there). A constant share spaces EC steps exactly evenly (5% = one step in every 20). train.py logs `ecmix_ratio`, `ecmix_ec_step`, the TC-assignment load CV / max share / active experts, and both contrastive losses.
- capacity_combo factors (capacity_combo variant only): **H** `hetero_expert` + `hetero_min_ratio`/`hetero_max_ratio`; **R** `ls_balance_mode` (`"off"` or `"token"` = historical diagonal LS-Reg) + `ls_diag_sign`/`ls_diag_strength`/`ls_ema_beta`; **O/P** `expert_output_blocks`/`expert_param_blocks` choose the blocks for each view (a block in both lists runs `dual_additive`, a block only in the param list runs `param_signature`), with `expert_contrastive_output_lam`/`_param_lam` (0.5/0.5), independent `expert_contrastive_output_temperature`/`_param_temperature` (0.5/0.7), and `expert_contrastive_signature_bins` for the P signature length.

Constraints to know:
- For the `repa_router` model, `routing_contrastive_lam` defaults to **0** (a default kwarg, not an assert) — the model expects the contrastive term off since alignment happens at the router level. For `repa_router_contra`, it is the total budget shared between REPA alignment and contrastive via linear handoff over `router_loss_decay_steps`.
- `expert_contrastive_blocks` must all be MoE blocks (asserted at init).
- `noise_expert_ema` model: noise expert params are `requires_grad=False` and updated via EMA — excluded from optimizer.

### Training Pipeline (`train.py`)
- PyTorch DDP for multi-GPU distributed training via `mp.spawn`. The YAML `gpu_ids` field sets `CUDA_VISIBLE_DEVICES` and determines the DDP world size (`gpus_per_machine`).
- Training hyperparameters (`lr`, `total_train_batch_size`, `weight_decay`, `num_steps`) are per-YAML — `config.py` only has framework defaults like `max_grad_norm`, `betas`, `weighting_scheme`.
- Logit-normal timestep sampling (SD3-style) with Rectified Flow objective
- Mixed precision with bfloat16; gradient clipping at `max_grad_norm=0.5`
- EMA model maintained for stable generation
- Supports both raw image loading and pre-computed VAE latents (`use_pre_latents=True`)
- Optional structured batch sampling (opt-in, default off, `train.py` only): top-level `structured_batch_sampling: True` swaps `DistributedSampler` for `StructuredDistributedBatchSampler` (utils.py). Each batch is one of two cases, mixed by `structured_batch_case1_prob` (default 0.5): case 1 = random classes + a single broadcast timestep (varies class, fixed t); case 2 = a single random class + per-sample timesteps (fixed class, varies t). The loop detects case 1 via labels-not-all-equal and broadcasts the per-image t/sigma. Reuses the `ProMoE_TC_B` / `ProMoE_EC_BC_B` models (no new model class); default False keeps training bit-identical. Configs `004_ProMoE_B{,_EC_BC}_structbatch.yaml`, scripts under `scripts/structured_batch/`.
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
- VAE loading uses `load_vae()` from `utils.py`: if `--vae-path`/`vae_path` is given it loads directly from that path; otherwise it checks `pretrained_ckpt/vae/{repo_id-with-/-replaced-by---}/` (e.g. `stabilityai/sd-vae-ft-mse` → `pretrained_ckpt/vae/stabilityai--sd-vae-ft-mse/`) for a local copy first, and if absent downloads from HuggingFace and `save_pretrained()`s it there for reuse.
- All training entry points (`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, `train_with_mae.py`, `sample.py`, `preprocess/preprocess_vae.py`) use this cached loading path.
- REPA teacher encoders (DINOv2) are cached to `pretrained_ckpt/encoder/` after first download via torch.hub.

### Analysis Tools (`analyses/`)
~30 `run_*.py` entrypoints, each with a matching `analyses/<basename>.md` usage guide and
shared logic in an `analyses/<topic>/` subpackage (most of which ship `test_*.py`).
Consult `analyses/README.md` for the authoritative per-script list; the groups are:

- **Visualization / accounting** — `run_compute_flops.py` (FLOPs, activated params, expert frequency), `run_tokenwise_tsne.py` / `run_samplewise_pooled_tsne.py` / `run_imagewise_tsne.py` (routing t-SNE at three granularities), `run_repa_dyna_heatmap.py`, `run_token_choice_expert_heatmap.py`, `run_mos_routing_analysis.py` (MoS teacher-block selection: histograms, timestep evolution, token variance, routing entropy; auto-detects global/blockwise/per_block/mos router types). Helpers: `t_SNE/`, `heatmap/`, `flops/`, `mos_routing/`.
- **Causal routing probes** — `run_routing_translation_probe.py` / `run_routing_flip_probe.py` / `run_routing_translation_stratified_probe.py` (do top-1 routes follow transported content or absolute coordinates?), `run_expert_function_consistency_probe*.py`, `run_affinity_responsibility_probe.py`, `run_cfg_route_inversion_probe.py`, `run_phase_default_probe.py`, `run_phase_metric_checkpoint_probe.py`. Helpers: `routing_translation/`, `expert_function/`, `phase_default/`, `routing_metric/`.
- **Exact-counterfactual utility probes** — `run_timestep_utility_probe*.py`, `run_count_preserving_cycle_probe_batch.py`, `run_compute_exchange_probe_batch.py` / `run_compute_exchange_deployability_gate.py`, `run_denoising_regret_probe*.py` (FDRR evidence), `run_dino_utility_neighborhood.py`. Helpers: `timestep_utility/`, `denoising_regret/`, `dino_utility_neighborhood/`.
- **Checkpoint / longitudinal audits and gates** — `run_expert_update_budget_audit.py`, `run_learning_credit_balance_probe_batch.py` / `run_learning_credit_balance_cross_checkpoint.py`. Helpers: `expert_update_budget/`.

Many of these are written as **pre-registered / sealed gates**: they lock a checkpoint, a
case manifest, a discovery/confirmation split, and a bootstrap decision rule up front.
When extending one, preserve the lock — do not loosen a gate to make a result pass.

## Coding Conventions
- 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Model files follow `models_*.py` naming pattern. Preserve numeric experiment prefixes in config names (e.g., `004_ProMoE_L.yaml`).
- **1-indexed naming convention**: Script and config filenames use 1-indexed block numbers (e.g., `b3_5` means blocks 3-5 human-readable), while YAML `align_blocks` uses 0-indexed Python indices (e.g., `[2, 3, 4]`). Always maintain this distinction.
- No formatter or linter is configured — match surrounding style in the file you edit.
- Validate changes with `python -m py_compile <file>` for syntax, the relevant `unittest` module(s) (see "Unit Tests"), then a targeted smoke test (short training run, sample pass). Analysis-helper subpackages carry their own `test_*.py` — extend them rather than adding an ad-hoc script.

### Shell Script Convention
- **All new three-in-one (train + sample + eval) `.sh` experiment scripts must follow the `scripts/template.sh` pattern** — otherwise the other experiment server cannot run them. (Legacy split-purpose scripts under `scripts/repa/` such as `train_repa_B.sh` / `sample_and_eval_repa_B.sh` predate the template and are exempt; new work should not introduce more of them.)
- **Sequential pipeline pattern**: The template implements a train→stop→sample+eval→resume loop. For each step in `step_list_for_sample`: (1) generate a temp config with `num_steps` set to the checkpoint step + 1 and `resume_checkpoint: True`, (2) train until that step then exit, (3) sample + eval with GPUs fully free, (4) resume for the next step. The final step uses the original `num_steps` from config. This avoids concurrent training + sampling, which can exceed GPU memory for XL-scale models.
- Key template patterns: `set -euo pipefail`, locate repo root via `SCRIPT_DIR`/`REPO_ROOT`, parse `model_name`/`gpu_ids`/`num_fid_samples`/`step_list_for_sample`/`orig_num_steps` from YAML using inline Python, call training/sampling/evaluation with absolute python paths, and `find ... -name images | sort -V` for evaluation directory traversal. Never use `conda activate`.
- **Interpreters come from `scripts/_python_env.sh`.** `template.sh` still hardcodes them, but newer scripts `source "${REPO_ROOT}/scripts/_python_env.sh"` and use `$PROMOE_TRAIN_PYTHON` / `$PROMOE_EVAL_PYTHON`. That helper pins the experiment-server paths and **fails the script before any output bucket is touched** when they are absent, unless the caller explicitly sets `PROMOE_ALLOW_LOCAL_FALLBACK=1`. Prefer sourcing it over re-hardcoding paths.
- **Combined run logs go to `logs/`.** Write `LOG="${REPO_ROOT}/logs/log_<name>.log"` and `mkdir -p "$(dirname "$LOG")"`. Per-experiment `training.log` / `sample.log` stay under the experiment's own output dir. No ANSI colour codes in files; retired logs move to `logs/archived/<date>/`. `*.log` is gitignored; `logs/README.md` is tracked.
- **Gate/metric predicates live in `scripts/_eval_metric_helpers.sh`** (`promoe_eval_file_metrics_valid`, `promoe_eval_file_fid`, …). They parse the evaluator's `FID:` / `Inception Score:` lines strictly so a malformed value can never be coerced to `0` and read as a passing gate. Reuse them instead of writing a fresh `awk` one-liner.
- **Queue / supervisor scripts** (e.g. `scripts/capacity_combo/hrop_gate_then_q0p4.sh`) chain arms across GPU slot pairs: they require an attached tmux session, hold a kernel-owned `flock` so only one supervisor runs per repo, open one tmux window per experiment, and advance only when the preceding wrappers exit. They never background with `&`.
- **When creating a new script**, only two things need changing from template.sh: the `CONFIG` path and `LOG` filename. Also change the training entrypoint in the train step (`train_with_repa.py`, `train_with_MoS_repa.py`, or `train.py`) to match the model family.
- End-to-end scripts under `scripts/` use **absolute python paths** (e.g., `/mnt/workspace/yujie/.conda/envs/promoe/bin/python`) instead of `conda activate` for company server compatibility.
- Training/sampling uses the `promoe` env; evaluation uses the `fid_eval` env.
- **Run-time GPU-slot grouping (`scripts/_run_times/<date>/`)**: experiments are launched through thin per-date wrapper scripts so that two 4-GPU runs share one physical 8-GPU server without hand-editing `gpu_ids` from `0-3` to `4-7`. Two-step flow whenever you "write an experiment script":
  1. Create the semantic script `scripts/<family>/run_<...>.sh` as usual (template.sh-based) — unchanged from before.
  2. Allocate a launch slot with `scripts/_run_times/new_run.sh --script scripts/<family>/run_<...>.sh [--date YYYY_MM_DD] [--gpus 2|4|8] [--dry-run]`. It computes the next free slot, patches that experiment's YAML `gpu_ids` to match, and writes `scripts/_run_times/<date>/<slot>-<desc>.sh` (a wrapper that `exec`s the semantic script; GPU assignment lives entirely in the YAML).
  - **Slot naming = physical 8-GPU server map**: `X.1` → GPU `0-3`, `X.2` → GPU `4-7` (two 4-GPU jobs fill server `X`); a full 8-GPU job (e.g. XL) is named `X` with no sub-index → GPU `0-7`, consumes a whole server, so the next group starts at `X+1`. The leaf `gpu_ids` is written into each experiment's own YAML (`.1`→`[0,1,2,3]`, `.2`→`[4,5,6,7]`, full→`[0,1,2,3,4,5,6,7]`). A 2-GPU job takes a quarter `X.1`–`X.4` → GPU `0-1` / `2-3` / `4-5` / `6-7`; a server is split into quarters or halves, never both, and each wrapper header records its slot's GPUs so the allocator can tell them apart.
  - **Scope = one date dir only**: allocation reads and writes within the `--date` directory exclusively (it never inspects other date dirs), and within a date dir the assigned slots use disjoint GPUs by construction. The cross-dir caveat is operational, not automatic: don't run two date dirs' jobs on the same physical GPUs at once. Date format is `YYYY_MM_DD` (e.g. `2026_06_20`); `--date` defaults to today.
  - **Continue numbering from existing files**: a 4-GPU job takes the lowest open half in the date dir — this backfills a `.2` half (whose `.1` is an earlier 4-GPU job) that a later 8-GPU job skipped when it jumped to a fresh whole server. An 8-GPU job takes a fresh server `max_major + 1` (a whole `X`, no halves). `--desc` defaults to the semantic script's distinguishing name (`run_B_xxx_train_sample_eval.sh` → `B_xxx`). Use `--dry-run` to preview the slot without writing anything.
  - **Every date dir carries a `commands.md` command table** (user rule, 2026-09-14). The file holds **only the table** — no title, intro paragraphs or notes. It has the header `| 实验描述 | git分支 | 启动命令 | 输出位置 |` and one row per `<slot>-<desc>.sh` wrapper in slot order, for example `| Slot 1.1 · GPU 0-1 · <中文实验描述> | repa | bash scripts/_run_times/<date>/<slot>-<desc>.sh | outputs/<model_name>/<custom_cfg_name>/ |`. Every cell is plain text with no quotes or backticks, so it can be copied directly. `scripts/_run_times/2026_09_14/commands.md` shows the row format; copy its table, not its intro paragraphs. Update the table whenever a wrapper in that dir is added, renamed or removed. This is separate from the `/command-table` skill's `commands.csv`.

## Experiment Discipline & Provenance Gates
The project enforces a scientific protocol in code, not just in docs. `doc/design-todo.md`
is the live roadmap and states the current rules; the mechanisms below implement them.

**The 300K dual-CFG FID gate.** Every candidate trains **from scratch (step 0, empty output
dir)** to 300K, then samples 50K images at CFG 1.0 **and** 1.5 and scores them with the
OpenAI evaluator. It must beat the *fresh* ProMoE-TC baseline's 300K FID at **both** CFG
values; a candidate that fails is abandoned on the spot and never continued to 500K. Only a
gate-passing arm earns 500K plus routing/expert-specialization analysis. Corollaries that
matter when writing scripts:
- A mid-training checkpoint continuation is **not** a substitute for an independent
  experiment, and cannot be reported as a method result.
- Arms in a factorial study are **independent hypotheses**: a full-combination failure does
  not disprove a partial-combination or single-point arm, so each arm runs its own gate.
- Comparable arms must match on seed, init, data order, global batch 256, lr 1e-4, and
  training length. Only the factor under test may differ.
- The 2026-09-14 `ecmix` batch is exempt by the user's decision: it trains straight to 500K
  and is sampled and evaluated only there, because its annealed arms are still mid-schedule
  at 300K. It runs on 2 GPUs per arm, so it is not comparable with the 4-GPU fresh baseline.
- The 2026-09-14 `global_center` arm is evaluated at 300K and 500K and, by the user's decision, is
  not stopped by its 300K result. It matches the fresh baseline's 4-GPU setup, so its 300K FIDs
  compare directly with 30.58 / 9.59.

**A run that starts from another experiment's weights is not an experiment.** Project rule,
no exceptions: 接着别的实验的权重继续往下训练的实验一律不承认，因为因素混杂，这种实验根本不算是干净的
消融。A continuation cannot separate the method's effect from the borrowed checkpoint's, so it
is not evidence of anything and must not enter the repo — delete its config, its launcher, and
any model file exclusive to it rather than archiving them. **The exception is the same
trajectory extended**: one model trained to 300K and then continued to 500K in its *own*
output bucket is the normal gate-passing path, not a continuation. The mechanical enforcement
is `train.py`'s `_validate_strict_output_bucket()` (a fresh run demands an empty bucket) plus
the absence of any "start from this checkpoint" parameter — `load_latest_checkpoint()` reads
only the run's own `checkpoints/` dir, and the former `initial_checkpoint_path` seeding path
was removed. Do not add one back.

**Never write experiment results outside the repository.** `output_dir` must stay inside the
working tree — the `config.py` default `outputs/` is the intended value, and a config should
normally not set it at all. Checkpoints, `training.log` / `sample.log`, samples, evaluator
outputs, TensorBoard data, metrics, manifests and analysis artifacts all belong under
`outputs/{model_name}/{custom_cfg_name}/` (or another repo-tracked experiment dir). A result
written to an absolute path outside the repo (`/home/dev/promoe-runs`, `/mnt/cubefs/...`,
`/tmp`, …) is invisible to this repo's tooling, cannot be traced back from the config, and is
orphaned as soon as that server is recycled. **Inputs are the exception and are not affected**:
the ImageNet latents / images (`data_path`, `latent_data_path`), the VAE and the DINOv2 teacher
caches are large shared read-only dependencies that legitimately live outside the repo — being
allowed to *read* from `/home/dev` or `/lustre01` never authorizes *writing* results there.
No config sets an out-of-repo `output_dir` any more: the former `/home/dev/promoe-runs` and
`/home/dev/promoe-probes` contents were moved into `outputs/`, `outputs/archived_outputs/` and
`analyses/archvied_analyses/`, and the last two configs that pointed outside
(`004_ProMoE_B_dino_route_margin_gate_v2_{correct,shuffled}_s0.yaml`) dropped the line, so they
resolve to the in-repo buckets that now hold their data. Never add an absolute `output_dir`.
Analysis runners enforce the same rule for `--output-dir` through the argparse type
`analyses/timestep_utility/repository_output.repository_output_dir`, which also requires the
directory to be git-ignored so the sealed runners' clean-tree checks keep passing; use it in new runners.

**Archived configs hard-fail.** `train.py:main()` raises immediately when a config carries
`archived_experiment: True`. No config carries it today (the three
`004_ProMoE_B_credit_rate_*_301k_20k.yaml` continuations that did were deleted under the rule
above); the guard stays as the trip-wire for any future one. Do not strip the flag or retarget
`num_steps` to pass an archived continuation off as a fresh run.

**Removed research line.** `research_on_expert_learning_signal_balance/` once held the MoE
"learning-credit redistribution" hypothesis (per-expert suffix-gradient credit rate rather
than token count). Its only experiments were 301K→321K continuations, so the whole study —
configs, the scripts/credit_redistribution/ launchers, the run_credit_redistribution_gate
entrypoint, and the controller / protocol / evaluator / orchestration modules — was deleted. The directory now
holds **only `git_provenance.py`**, which is unrelated to the hypothesis and pinned by path +
sha256 in both `train.py`'s `STRICT_PROVENANCE_SOURCE_PATHS` and
`analyses/expert_update_budget/audit.py`'s `LOCKED_TRAINING_SOURCE_PATHS` — so do not move or
rename it. Its README records the hypothesis and the step-0 protocol a clean revival would
need; the read-only `run_learning_credit_balance_*` probes under `analyses/` are unaffected and
still valid.

**Strict training provenance** (`PROMOE_STRICT_PROVENANCE=1`, exactly `0` or `1`) makes a run
self-certifying. **No script enables it today** — the audited `expert_contra` arm that did was
removed unrun; set the env var yourself when a run must be self-certifying. It
requires CUDA, a **clean working tree**, `HEAD == origin/repa` with zero divergence, and
sha256 verification of a per-model source manifest (`STRICT_PROVENANCE_SOURCE_PATHS` in
`train.py`, registered today for `ProMoE_TC_B`, `ProMoE_TC_B_expert_contra`,
`ProMoE_TC_B_capacity_combo`). Adding a strict-provenance model means adding its manifest
entry. `PROMOE_RUN_ID` (16-128 chars of `[A-Za-z0-9_-]`) tags the run. On resume, a
strict-provenance checkpoint's recorded provenance must equal this invocation's exactly
(`_training_provenance_matches`) — there is no escape hatch, so a changed pinned source
file means a fresh run, not a patched constant.

## Adding a New Experiment

### Config-driven ablation (no new model file)
If the ablation is controlled by an existing config flag (e.g., `router_norm_type`, `align_target`, `use_dynamic_coeff`), only a new YAML config and shell script are needed — the `model_name` stays the same. Add the flag with a default that preserves backward compatibility. Allocate its launch slot via `scripts/_run_times/new_run.sh` as well (see "Run-time GPU-slot grouping").

### New model variant
1. **Model**: Create `models/models_ProMoE_TC_<variant>.py` (EC-family variants follow `models_ProMoE_EC_<variant>.py` and inherit from `ProMoE_EC` instead). Inherit from the closest existing variant. Follow `forward()` return conventions (see Auxiliary Loss Convention above).
2. **Register**: Add a `(ModelClass, config_key)` entry to `model_dict` in the appropriate training script (`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`). `sample.py` merges all dicts automatically.
3. **Config**: Create `configs/004_ProMoE_<size>_<variant>.yaml`. Set `model_name` to match the registered key. Add `MoE_config` and/or `repa_config` under the model config key as needed.
4. **Shell script**: Copy `scripts/template.sh` to `scripts/<family>/run_<size>_<variant>_train_sample_eval.sh`. Update `CONFIG` and `LOG` variables. Choose the correct training entrypoint in the train step.
5. **Validate**: `python -m py_compile models/models_ProMoE_TC_<variant>.py` then a short training run. Also run `python scripts/check_output_dir.py --config configs/004_ProMoE_<size>_<variant>.yaml` — the output dir is `outputs/{model_name}/{custom_cfg_name}` (config filename), and the guard fails if that dir already exists or is claimed; on a hit, bump the config (and its script/wrapper) to the suggested `_vN` name. **Re-running an experiment whose model code changed**: do not reuse the old config name — use `/rerun-experiment` (or follow it manually) to `_vN`-bucket the {config, script, wrapper} so the new run gets a clean output dir.
6. **Run-time slot**: allocate a launch slot with `scripts/_run_times/new_run.sh --script scripts/<family>/run_<size>_<variant>_train_sample_eval.sh [--gpus 2|4|8]` instead of hand-setting `gpu_ids` (see "Run-time GPU-slot grouping" above).

## Important Notes
- **Experiments run on other servers; this server is primarily for writing code.** Training/sampling/evaluation are launched elsewhere — on this machine expect to author models, configs, scripts, and run-time slot wrappers, not to run real training. Experiment results (per-experiment FID/IS at 300K/500K, cfg 1.0/1.5) live on a dashboard at `https://video-generation-wulanchabu.oss-cn-wulanchabu.aliyuncs.com/yujie/projects/promoe_plus/index.html`. The **WebFetch tool is blocked** for this domain, but plain `curl` from the server reaches it fine — pull and parse locally: `curl -sS -L "<url>" -o /tmp/promoe_dashboard.html` then read `/tmp/promoe_dashboard.html` (the results are one HTML `<table class="cmp">`).
- All paper results use `qk_norm=False`. Enable `qk_norm=True` for training beyond 2M steps.
- Token-Choice routing is default; use Expert-Choice for DDPM training. Two EC variants exist: per-image (`models_ProMoE_EC.py`) and batch-flatten (`models_ProMoE_EC_batch_choice.py`, key `ProMoE_EC_BC_B`).
- Evaluation requires a separate TensorFlow environment and the reference batch `VIRTUAL_imagenet256_labeled.npz` from OpenAI's guided-diffusion. `evaluation/download_ref_batches.py` can auto-download these.
- `cfg.data_path` in `config.py` is the ImageNet train directory. It defaults to the shared `/lustre01/yujie/dataset/imagenet/train` (where `prepare_imagenet.sh` materialises the data) and is overridable via the `PROMOE_DATA_PATH` env var; train.py has no `--data-path` CLI flag. This absolute default is `train`-once-safe — none of `/lustre01`, `yujie`, `dataset`, `imagenet` contains the substring 'train', only the `train/` dir does — which train.py's `str.replace('train', ...)` latent derivation requires (keep any override the same way). See "Dataset auto-preparation".
- Multi-GPU sampling produces different random sequences than single-GPU (different class label ordering).
- REPA training requires raw images (not just pre-computed latents) since the teacher encoder operates on pixel space. The dataset returns `(path, label, latent, raw_image)` when `load_raw_image=True`.
- Offline/air-gapped training: all training scripts and `sample.py` accept `--vae-path`; `train_with_repa.py` and `train_with_MoS_repa.py` also accept `--repa-enc-path`. See `doc/ProMoE-REPA.md` for details.
- `preprocess/image_paths_cache.txt` caches the dataset file list (shared by `train.py` and `preprocess_vae.py`); delete and rebuild it after switching datasets or reorganizing files. `prepare_imagenet.py` rebuilds it deterministically (sorted, atomic) so DDP ranks don't race to regenerate it.
- When `use_pre_latents=True`, the latent directory must be a sibling of `train/` named `sd-vae-ft-mse_Latents_256img_npz` — the code derives latent paths by replacing `train` in image paths.
- `model.py` at the repo root is an unrelated reference file (not imported anywhere in the project). Ignore it when navigating the codebase — the project's models live in `models/`.
- **`output_dir` is a per-config top-level key** (default `outputs/`), so an output bucket is `{output_dir}/{model_name}/{custom_cfg_name}/`. It must stay inside the repo — see "Never write experiment results outside the repository" above. **Input paths are separate**: an experiment-server config may read latents from `/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz` with `use_encoded_latents: True`, so do not assume the `/lustre01/...` layout when reading a config. `scripts/check_output_dir.py` reads only **top-level** (zero-indent) `model_name`/`output_dir` scalars on purpose.
- `_previous_results/` holds the archived experiment table (`results_all_experiments_2026_06_21_to_08_05.md` + an HTML render). No script reads it any more (its only reader, the capacity-combo queue, was deleted), but keep it as archived data rather than a scratch note.
- `sample.py` loads checkpoints through a restricted unpickler where the installed torch supports it (`weights_only=True` plus `safe_globals([EasyDict, TorchVersion])`), falling back only when the runtime lacks `safe_globals`. Keep new checkpoint metadata inside that allowlist rather than widening the fallback.
- Three skill directories mirror each other: `.claude/skills/` (Claude Code), `.agents/skills/` (the active Codex definitions — `check-cc`/`inspect-cc` are the Codex-side names), and `.codex/skills/`. When editing a shared workflow, update the mirrors together.

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
- `doc/ProMoE-REPA.md` — Detailed guide for all REPA variant workflows, configuration reference, and FAQ.
- `AGENTS.md` — Full project structure reference, output layout, testing guidelines, and commit conventions.
- `analyses/README.md` — Overview of analysis entrypoints; per-script usage in `analyses/<basename>.md` files.
- `plans/` — Implementation plans for Cross-Attention variants (`plan_01` through `plan_08`), covering both standard REPA and MoS cross-alignment designs.
- `doc/implementation-plan.md` — Draft plan (Chinese) for a future "attention-weighted same-expert same-image alignment" experiment family. Not yet implemented; reference for forthcoming work, not current code.
- `doc/design-todo.md` — **The live MoE-mainline roadmap (Chinese).** Per-improvement-group status (implemented / verified / pushed / abandoned), the decision logs, and the 300K dual-CFG gate ordering rule. Read this first to learn what the current priority is; keep it updated when an arm's status changes.
- `doc/todo.md` — Short-lived launch queue rendered as a command table (slot · GPUs · branch · command · output dir).
- `doc/load-balance-design.md`, `doc/contrastive-label-smoothing.md`, `doc/shared-expert-augmentation-plan.md` — Design notes behind the lbcontra / lossfree, lsreg, and dagfuse_shared families.
- `doc/output-table-template.md`, `command-tables/command-table-template.csv` — Templates for result tables and for the `/command-table` CSV output.
- `research_on_expert_learning_signal_balance/README.md` — Why the learning-credit-redistribution line was deleted (its only experiments were checkpoint continuations), what a clean revival must satisfy, and why `git_provenance.py` must stay put.
- `logs/README.md` — The `logs/` logging convention (Chinese).
- `collapse_smoking_test/crash_diagnosis_report.md` — The cross-alignment crash investigation behind the stability constraints above.

## Project-Local Skills (`.claude/skills/`)
Eight project-specific slash commands live under `.claude/skills/`. They encode the project-aware checks (model_dict ↔ models/ ↔ configs/ ↔ scripts/ four-way consistency, cross-alignment stability invariants, TrainingMonitor hook integrity, output-dir collision avoidance) so future Claude instances don't have to re-derive them.

- `/inspect [n]` — Carpet-style code-quality loop on **the entire codebase**: scan → fix → commit (per iteration) → smoke test. Terminates after `n` consecutive iterations with zero findings, or hard caps at 20 iterations; `n` is an optional argument (integer 1–20, default 2), e.g. `/inspect 3` or `/inspect n=3`. Smoke test is `py_compile` + import check only — never starts real training/sampling. Each iteration produces its own commit (`chore(inspect): iter N — ...`); never amends, force-pushes, or pushes.
- `/check [n]` — Same loop shape and `n` argument as `/inspect`, but the scan is **scoped to the current uncommitted diff** (modified + staged + untracked relative to HEAD). Does **not** commit during the loop — leaves the validated WIP dirty for the user to commit themselves. Stops immediately if the working tree is already clean.
- `/inspect-codex [n]` — Codex-augmented `/inspect`: each iteration also briefs an **independent Codex reviewer** (headless **`codex exec`**, xhigh reasoning, launched **yolo / no sandbox** via `--dangerously-bypass-approvals-and-sandbox` in a new tmux window that **closes itself when the review finishes** — `codex exec` exits on completion, so nothing lingers and no decision is needed; findings read from the `-o` capture, with the session rollout `last_agent_message` as fallback) run **in parallel** with Claude's own scan, then aggregates both finding sets, adjudicates the real problems, fixes them, smoke-tests, and commits. Codex only reviews — kept review-only by instruction **plus** a snapshot/checksum-revert guard (any file Codex writes is reverted to the pre-Codex snapshot, since it runs unsandboxed); Claude is the sole fixer. Same `n`-consecutive-clean (default 2) / 20-iteration termination. Billed (Codex xhigh quota); requires being inside tmux.
- `/check-codex [n]` — Codex-augmented `/check`: the same diff-scoped loop and `n` argument with the parallel yolo Codex reviewer (same headless `codex exec` launch in a tmux window that closes itself when the review finishes + dirty-set backup / checksum-revert guard), but **never commits** (leaves the vetted WIP dirty). Stops immediately if the working tree is clean.
- `/new-experiment` — Scaffolds a complete experiment end-to-end and allocates its GPU run-time slot: (new variant) model file + `model_dict` registration + config, or (ablation) config only → `template.sh`-based train+sample+eval run script → `py_compile` + four-way-consistency validation → `scripts/_run_times/new_run.sh` slot allocation (patches the experiment YAML's `gpu_ids`, writes the per-date `<slot>-<desc>.sh` wrapper). **Auto-fires** on "write/add an experiment" requests; always previews the slot with `--dry-run` before writing. Validation now includes a **mandatory output-dir collision guard** (`python scripts/check_output_dir.py --config <cfg>`) so a new experiment never targets an existing run's `outputs/{model_name}/{custom_cfg_name}` dir. Unlike the four review loops it is a one-shot scaffolder, not an iterate-to-clean loop — and it never launches the run (hands off a tmux command instead). After writing the wrapper it auto-invokes `/describe-experiment` to drop a `<slot>-<desc>-describe.txt` beside it. See "Run-time GPU-slot grouping" and "Adding a New Experiment".
- `/rerun-experiment` — Re-buckets an **existing** experiment to a fresh `_vN` output dir before re-running it after a model-code change (crash fix, init/normalization/architecture edit). Output dirs are `outputs/{model_name}/{custom_cfg_name}`, so a naive re-run with the same config name silently collides with — or resumes from — the prior (crashed/stale) run's checkpoints. Traces the experiment's full {config, semantic run script, run-time wrapper} set, `git mv`s all three in lock-step to a `_vN` name (default via `scripts/check_output_dir.py --suggest-version`), updates every in-file reference (`CONFIG=`/`LOG=`/`exec`), and validates the chain — **keeping the existing GPU slot + `gpu_ids` (no `new_run.sh` call)**. The old run's on-disk data is preserved under the old name. It `git mv`s the companion `<slot>-<desc>-describe.txt` to the `_vN` name and auto-invokes `/describe-experiment` to regenerate it. A one-shot rename helper, not an iterate-to-clean loop; never launches a run. Distinct from `/new-experiment` (first-time creation) — use this for "改完代码重跑 / 重新跑" an experiment.
- `/command-table` — Organizes the launch wrappers in one `scripts/_run_times/<date>/` directory into a CSV command table (Notion/Excel-importable). Reads each `<slot>-<desc>.sh` wrapper, traces it (wrapper → semantic run script → config) to fill the columns (`实验描述,git分支,启动命令,输出位置`), renders using `command-tables/command-table-template.csv` (RFC 4180 quoting, plain-text cells — no backticks), and writes `commands.csv` into that same date dir. **Auto-fires** on requests to organize/summarize the run-time commands into a table — e.g. "把 `scripts/_run_times/<date>` 中的指令整理为命令表格" / "整理/生成命令表格". A one-shot generator (like `/new-experiment`), not an iterate-to-clean loop; reads/writes only `commands.csv` and never launches a run.
- `/describe-experiment` — Writes a plain-text description for a run-time wrapper. Traces it (wrapper → semantic run script → config → model file / CLAUDE.md variant table) and writes `<slot>-<desc>-describe.txt` (the wrapper basename minus `.sh`, plus `-describe.txt`) **in the same date dir**: a numbered 1/2/3 list of the experiment's core changes relative to the baseline, **most-important change first**, bilingual (中文为主 + English terms). "Baseline" is written against **both** the base ProMoE model (TC/EC) **and** the immediate parent variant (inheritance parent or sibling flag value). **Auto-fires** right after a wrapper is created/re-bucketed — `/new-experiment` and `/rerun-experiment` invoke it at the end — and on requests like "为 `scripts/_run_times/<date>/<slot>-<desc>.sh` 写实验描述" / "describe this experiment". A one-shot generator (like `/command-table`), not an iterate-to-clean loop; reads source files and writes only the `*-describe.txt` and never launches a run.

All eight skills explicitly refuse to: push/force-push/amend, run real training/sampling, edit runtime artifact dirs (`outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`, `collapse_smoking_test*/`), or touch the vendored `REPA/` (uppercase) subproject. The two `-codex` variants additionally run Codex **yolo / unsandboxed** headless via `codex exec` (`--dangerously-bypass-approvals-and-sandbox`), review-only, inside a new tmux window that **closes itself when the review finishes** (no manual step), with a snapshot/checksum-revert guard that reverts anything Codex writes to the pre-Codex snapshot — Claude remains the sole writer.
