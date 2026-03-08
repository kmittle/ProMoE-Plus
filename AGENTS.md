# Repository Guidelines

## Project Structure & Module Organization
Core entrypoints are at repository root:

- `train.py`: baseline and non-REPA training (DiT, TCDiT, ECDiT, DiffMoE, ProMoE, hierarchical and expert variants).
- `train_with_repa.py`: REPA-enabled training (REPA / REPA-Shared / REPA-Cond / REPA-DYNA / REPA-DYNA-SELECT), including teacher-feature alignment loss.
- `sample.py`: sampling/inference entrypoint. It merges model registries from both `train.py` and `train_with_repa.py`, so one script can sample all registered families.

Shared defaults and helpers:

- `config.py`: global defaults (`cfg`) and model templates (`DiT_*_config`, `DiffMoE_*`, etc.).
- `utils.py`: `deep_update`, `find_free_port`, VAE caching/loading (`load_vae`), CLI list parsers, and Inception utilities.

Main code layout:

- `models/`: architecture implementations; keep the `models_*.py` naming convention.
- `repa/`: REPA helper package used by `train_with_repa.py` (`encoder.py`, `loss.py`).
- `preprocess/`: VAE latent preprocessing and shared cache file `preprocess/image_paths_cache.txt`.
- `evaluation/`: OpenAI-style evaluation pipeline (`run_eval.py`, `evaluator.py`, `download_ref_batches.py`).
- `scripts/repa/`: REPA-B / REPA-Shared-B / REPA-Cond-B train + infer/eval wrappers.
- `scripts/hierar/`: B-scale hierarchical/expert train + infer/eval wrappers.
- `scripts/dynamic_repa/`: REPA-DYNA-B and REPA-DYNA-SELECT-B train + sample + eval pipelines (including select-ratio variants r25/r75).
- `compute_FLOPs/`: FLOPs/statistics utilities.
- `REPA/` (uppercase): separate upstream-style subproject with its own docs and `AGENTS.md`.

Root wrappers:

- `run_all_infer_eval_500K.sh`: batch sample+eval for multiple configs at 500K.
- `eval_B_hierar_expert.sh`, `eval_B_hierar_expert_NoPenalty.sh`: eval-only wrappers.

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

Optional local VAE path (skips auto-download/cache in train/sample):

```bash
python train.py --config configs/004_ProMoE_B.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml --vae-path /path/to/sd-vae-ft-mse
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

bash scripts/dynamic_repa/run_B_repa_dyna_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_r25_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_r75_train_sample_eval.sh

bash scripts/hierar/run_B_hierar_train.sh
bash scripts/hierar/run_B_hierar_infer_eval.sh
bash scripts/hierar/run_B_hierar_expert_train.sh
bash scripts/hierar/run_B_hierar_expert_infer_eval.sh
bash scripts/hierar/run_B_hierar_expert_NoPenalty_train.sh
bash scripts/hierar/run_B_hierar_expert_NoPenalty_infer_eval.sh

bash run_all_infer_eval_500K.sh
```

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

- Training changes: `python train.py --config ...` or `python train_with_repa.py --config ...`.
- Sampling changes: `python sample.py --config ...` (with `--step_list_for_sample` / `--guide_scale_list` as needed).
- Evaluation changes: run from inside `evaluation/`: `python run_eval.py ...`.
- Syntax checks: `python -m py_compile <modified_python_files>`.
- End-to-end REPA-DYNA smoke check: run one wrapper in `scripts/dynamic_repa/` and verify train/sample/eval logs are produced.

If you touch dataset traversal, latent mapping, or preprocessing logic, clear/regenerate `preprocess/image_paths_cache.txt` before re-running checks.

## Commit & Pull Request Guidelines
Use concise, imperative, single-scope commit subjects. In PR descriptions, include:

- Affected model family/config(s).
- Whether the path is `train.py` or `train_with_repa.py`.
- Dataset layout assumptions (especially `train/` path and latent sibling path).
- GPU assumptions (`gpu_ids`, world size, sampling GPUs).
- Evidence of behavior change (logs, sample folders, metric outputs).

## Configuration Notes
- Set `cfg.data_path` (or YAML override) to ImageNet `train/` root before preprocess/train.
- `custom_cfg_name` is auto-injected from `--config` filename stem and used in output path construction.
- Training uses `gpu_ids` from YAML to set `CUDA_VISIBLE_DEVICES` when provided.
- Sampling uses `sample_gpu_ids` only if provided; otherwise it uses all visible GPUs.
- `train_with_repa.py` reads REPA behavior from top-level `repa_config`; model-level REPA knobs live under `DiT_*_config.repa_config` in YAML.
- Dynamic-select configs (`004_ProMoE_B_repa_dyna_select*.yaml`) control token selection via `DiT_B_config.repa_config.repa_select_ratio`.
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

Weight caching:

- VAE auto-cache path: `pretrained_ckpt/vae/<hf_repo_id_with_slash_replaced>/` (unless `--vae-path` is passed).
- REPA teacher cache path: `pretrained_ckpt/encoder/<hub_name>/state_dict.pth` (unless `--repa-enc-path` is passed).
- For REPA, rank 0 performs initial teacher download/cache, then other ranks load from local cache after barrier.

Evaluation notes:

- `evaluation/run_eval.py` always calls `ensure_ref_batches()` to auto-download missing reference NPZs.
- It expects generated PNG names containing class suffix like `_class123.png`.
- It writes `<image_folder>.npz`, and when evaluation runs it also writes `<image_folder>_eval_openai.txt`.
- Run it from inside `evaluation/` so relative `evaluator.py` lookup succeeds.
