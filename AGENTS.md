# Repository Guidelines

## Project Structure & Module Organization
Core entrypoints are at repository root:

- `train.py`: baseline and non-REPA training (DiT, TCDiT, ECDiT, DiffMoE, ProMoE, hierarchical and expert variants).
- `train_with_repa.py`: REPA-enabled training (REPA / REPA-Shared / REPA-Cond / REPA-DYNA / REPA-DYNA-SELECT / REPA-DYNA-SCALE / REPA-DYNA-ONLY / REPA-Router / REPA-Router-Contra / REPA-Routed / REPA-Double-Share), including teacher-feature alignment loss.
- `train_with_MoS_repa.py`: MoS-REPA and MoS-REPA-Naive training with teacher-block routing and per-block REPA projectors.
- `sample.py`: sampling/inference entrypoint. It merges model registries from `train.py`, `train_with_repa.py`, and `train_with_MoS_repa.py`, so one script can sample all registered families.

Shared defaults and helpers:

- `config.py`: global defaults (`cfg`) and model templates (`DiT_*_config`, `DiffMoE_*`, etc.).
- `utils.py`: `deep_update`, `find_free_port`, VAE caching/loading (`load_vae`), CLI list parsers, and Inception utilities.
- `ProMoE-REPA.md`: repo-level guide for current REPA / MoS-REPA workflows and variants.

Main code layout:

- `models/`: architecture implementations; keep the `models_*.py` naming convention.
- `repa/`: REPA helper package used by `train_with_repa.py` and `train_with_MoS_repa.py` (`encoder.py`, `loss.py`).
- `preprocess/`: VAE latent preprocessing and shared cache file `preprocess/image_paths_cache.txt`.
- `evaluation/`: OpenAI-style evaluation pipeline (`run_eval.py`, `evaluator.py`, `download_ref_batches.py`).
- `scripts/repa/`: REPA-B / REPA-Shared-B / REPA-Cond-B helpers plus router / routed / double-share train + sample + eval wrappers.
- `scripts/MoS_repa/`: MoS-REPA and MoS-REPA-Naive train + sample + eval wrappers following `scripts/template.sh`.
- `scripts/hierar/`: B-scale hierarchical/expert train + infer/eval wrappers.
- `scripts/dynamic_repa/`: REPA-DYNA-B, REPA-DYNA-SELECT-B, REPA-DYNA-SCALE-B, and REPA-DYNA-ONLY-B train + sample + eval pipelines (including select-ratio variants r25/r75).
- `compute_FLOPs/`: FLOPs/statistics utilities.
- `REPA/` (uppercase): separate upstream-style subproject with its own docs and `AGENTS.md`.

Top-level wrappers:

- `scripts/run_all_infer_eval_500K.sh`: batch sample+eval for multiple configs at 500K.
- `scripts/eval_B_hierar_expert.sh`, `scripts/eval_B_hierar_expert_NoPenalty.sh`: eval-only wrappers.

Outputs follow:

`outputs/<model_name>/<config_stem>/`

with:

- `checkpoints/ckpt_step_*.pth`
- `training.log`, `sample.log`
- `tensorboard/`
- `sample/step<step>/img<...>/images`

## Build, Test, and Development Commands
Create training env:

```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```

Run standard training:

```bash
python train.py --config configs/004_ProMoE_L.yaml
```

Run REPA training (optional local teacher checkpoint):

```bash
python train_with_repa.py \
  --config configs/004_ProMoE_B_repa.yaml \
  --repa-enc-path /path/to/state_dict.pth
```

Run MoS-REPA training (optional local teacher checkpoint):

```bash
python train_with_MoS_repa.py \
  --config configs/004_ProMoE_B_repa_MoS.yaml \
  --repa-enc-path /path/to/state_dict.pth
```

Optional local VAE path (skips auto-download/cache in train/sample):

```bash
python train.py --config configs/004_ProMoE_B.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_MoS_repa.py --config configs/004_ProMoE_B_repa_MoS.yaml --vae-path /path/to/sd-vae-ft-mse
python sample.py --config configs/004_ProMoE_B_repa.yaml --vae-path /path/to/sd-vae-ft-mse
```

Precompute VAE latents:

```bash
python preprocess/preprocess_vae.py \
  --latent_save_root /path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz
```

Sampling (explicit step/CFG/num samples):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python sample.py \
  --config configs/004_ProMoE_B_repa.yaml \
  --step_list_for_sample 300000,500000 \
  --guide_scale_list 1.0,1.5 \
  --num_fid_samples 50000
```

Common wrappers:

```bash
bash scripts/repa/train_repa_B.sh
bash scripts/repa/train_repa_shared_B.sh
bash scripts/repa/train_repa_cond_B.sh

bash scripts/repa/sample_and_eval_repa_B.sh
bash scripts/repa/sample_and_eval_repa_shared_B.sh
bash scripts/repa/sample_and_eval_repa_cond_B.sh
bash scripts/repa/run_B_repa_router_train_sample_eval.sh
bash scripts/repa/run_B_repa_router_contra_train_sample_eval.sh
bash scripts/repa/run_B_repa_routed_train_sample_eval.sh
bash scripts/repa/run_B_repa_double_share_train_sample_eval.sh

bash scripts/MoS_repa/run_B_repa_mos_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_train_sample_eval.sh

bash scripts/dynamic_repa/run_B_repa_dyna_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_r25_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_r75_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_scale_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_only_train_sample_eval.sh

bash scripts/hierar/run_B_hierar_train.sh
bash scripts/hierar/run_B_hierar_infer_eval.sh
bash scripts/hierar/run_B_hierar_expert_train.sh
bash scripts/hierar/run_B_hierar_expert_infer_eval.sh
bash scripts/hierar/run_B_hierar_expert_NoPenalty_train.sh
bash scripts/hierar/run_B_hierar_expert_NoPenalty_infer_eval.sh

bash scripts/run_all_infer_eval_500K.sh
```

When adding any new train + sample + eval all-in-one `.sh` wrapper, follow the structure and execution pattern of `scripts/template.sh` rather than inventing a new style; this is required for compatibility with another experiment server.

Create evaluation env (TensorFlow-based):

```bash
conda create -n promoe_eval python=3.9 -y
conda activate promoe_eval
cd evaluation
pip install -r requirements.txt
python run_eval.py /path/to/generated/images --count 50000
```

Pack PNGs to NPZ without running evaluator:

```bash
python run_eval.py /path/to/generated/images --count 50000 --no-eval
```

## Coding Style & Naming Conventions
- Follow existing style in touched files: 4-space indentation, `snake_case` functions/variables, `PascalCase` classes.
- Preserve existing import grouping and logging-heavy style.
- Keep config filenames in the numeric-prefixed style (for example `004_*.yaml`).
- Add overrides in YAML and rely on `deep_update` to only replace intended keys.

## Testing Guidelines
No dedicated `tests/` directory. Use smoke checks aligned to your change surface:

- Training changes: `python train.py --config ...`, `python train_with_repa.py --config ...`, or `python train_with_MoS_repa.py --config ...`.
- Sampling changes: `python sample.py --config ...` (with `--step_list_for_sample` / `--guide_scale_list` as needed).
- Evaluation changes: run from inside `evaluation/`: `python run_eval.py ...`.
- Syntax checks: `python -m py_compile <modified_python_files>`.
- End-to-end REPA-DYNA smoke check: run one wrapper in `scripts/dynamic_repa/` and verify train/sample/eval logs are produced.
- End-to-end MoS-REPA smoke check: run `bash scripts/MoS_repa/run_B_repa_mos_train_sample_eval.sh` and verify train/sample/eval logs are produced.
- Before using one-click wrappers under `scripts/repa/` or `scripts/MoS_repa/`, check whether they hard-code local Python interpreter paths and adjust them if needed; the direct Python entrypoints are the portability baseline.
- When writing a new training + sampling + evaluation three-in-one shell script, start from `scripts/template.sh` and preserve its pattern; otherwise the script may fail on the other experiment server.

If you touch dataset traversal, latent mapping, or preprocessing logic, clear/regenerate `preprocess/image_paths_cache.txt` before re-running checks.

## Commit & Pull Request Guidelines
Use concise, imperative, single-scope commit subjects. In PR descriptions, include:

- Affected model family/config(s).
- Whether the path is `train.py`, `train_with_repa.py`, or `train_with_MoS_repa.py`.
- Dataset layout assumptions (especially `train/` path and latent sibling path).
- GPU assumptions (`gpu_ids`, world size, sampling GPUs).
- Evidence of behavior change (logs, sample folders, metric outputs).

## Configuration Notes
- Set `cfg.data_path` (or YAML override) to ImageNet `train/` root before preprocess/train.
- `custom_cfg_name` is auto-injected from `--config` filename stem and used in output path construction.
- Training uses `gpu_ids` from YAML to set `CUDA_VISIBLE_DEVICES` when provided.
- Sampling uses `sample_gpu_ids` only if provided; otherwise it uses all visible GPUs.
- `train_with_repa.py` and `train_with_MoS_repa.py` read REPA behavior from top-level `repa_config`; model-level REPA knobs live under `DiT_*_config.repa_config` in YAML.
- MoS-REPA configs (for example `004_ProMoE_B_repa_MoS.yaml`) additionally set `DiT_*_config.repa_config.num_teacher_blocks`; keep it aligned with the chosen teacher encoder depth.
- Dynamic-select configs (`004_ProMoE_B_repa_dyna_select*.yaml`) control token selection via `DiT_B_config.repa_config.repa_select_ratio`.
- Router configs use model-level REPA knobs such as `router_repa_coeff` (`004_ProMoE_B_repa_router.yaml`) or `router_loss_decay_steps` (`004_ProMoE_B_repa_router_contra.yaml`).
- `004_ProMoE_B_repa_dyna_only.yaml` also carries a model-level `DiT_B_config.repa_config.proj_coeff` for the capped dynamic-weight ablation.
- Most provided YAMLs set `resume_checkpoint: True`; when no checkpoint exists the loader logs an error and training starts from step 0.
- `sample.py` behavior:
  - if `step_list_for_sample` is set, it loads only those checkpoints;
  - otherwise it scans `checkpoints/` and loads steps divisible by `sample_every_step`;
  - `--num_fid_samples` also updates `save_img_num`.

Latent mode and cache behavior:

- With `use_pre_latents=True`, both training and preprocessing rely on `preprocess/image_paths_cache.txt`.
- Latent path rule in training is string-based: image path replaces `train` with `sd-vae-ft-mse_Latents_256img_npz`, extension becomes `.latent.npz`.
- Keep dataset naming aligned with this replacement rule, or update the code.
- In REPA training with `use_pre_latents=True`, the dataset additionally loads raw images for teacher feature extraction.
- In MoS-REPA training, teacher features are extracted from all teacher blocks and aligned block-wise against every DiT block.

Weight caching:

- VAE auto-cache path: `pretrained_ckpt/vae/<hf_repo_id_with_slash_replaced>/` (unless `--vae-path` is passed).
- REPA teacher cache path: `pretrained_ckpt/encoder/<hub_name>/state_dict.pth` (unless `--repa-enc-path` is passed).
- For REPA and MoS-REPA, rank 0 performs initial teacher download/cache, then other ranks load from local cache after barrier.

Evaluation notes:

- `evaluation/run_eval.py` always calls `ensure_ref_batches()` to auto-download missing reference NPZs.
- It expects generated PNG names containing class suffix like `_class123.png`.
- It writes `<image_folder>.npz`, and when evaluation runs it also writes `<image_folder>_eval_openai.txt`.
- Run it from inside `evaluation/` so relative `evaluator.py` lookup succeeds.
