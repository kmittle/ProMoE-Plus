# Repository Guidelines

## Project Structure & Module Organization
Core training and sampling entrypoints are at the repository root:

- `train.py`: standard/baseline training path (DiT, TCDiT, ECDiT, DiffMoE, ProMoE, hierarchical variants).
- `train_with_repa.py`: REPA-enabled training path, including REPA / REPA-Shared / REPA-Cond model variants and teacher-feature alignment loss.
- `sample.py`: inference entrypoint; merges model registries from both `train.py` and `train_with_repa.py`, so one script can sample all supported model families.

Shared defaults and helpers:

- `config.py`: global defaults plus model config templates (`DiT_*_config`, `DiffMoE_*`, etc.).
- `utils.py`: recursive config merge (`deep_update`), VAE loading/caching, free-port discovery, CLI list parsers, and Inception utilities.

Model code layout:

- `models/`: architecture implementations. Keep the `models_*.py` naming style.
- REPA model variants currently include `models/models_ProMoE_TC_repa.py`, `models/models_ProMoE_TC_repa_shared.py`, and `models/models_ProMoE_TC_repa_cond.py`.

Data and evaluation:

- `preprocess/`: latent preprocessing and shared image path cache (`preprocess/image_paths_cache.txt`).
- `evaluation/`: OpenAI-style metric pipeline (`run_eval.py`, `evaluator.py`, reference-batch downloader).

Workflow scripts:

- `scripts/repa/`: train + sample/eval wrappers for REPA-B, REPA-Shared-B, REPA-Cond-B.
- `scripts/hierar/`: train + sample/eval wrappers for ProMoE hierarchical B variants.
- Root wrappers: `run_all_infer_eval_500K.sh`, `eval_B_hierar_expert.sh`, `eval_B_hierar_expert_NoPenalty.sh`.

Other subprojects:

- `repa/` (lowercase): helper package used by `train_with_repa.py` (`encoder.py`, `loss.py`).
- `REPA/` (uppercase): independent upstream-style subproject with its own entrypoints and its own `AGENTS.md`.

Outputs are organized as:

`outputs/<model_name>/<config_stem>/`

with checkpoints under `checkpoints/`, logs under `training.log` / `sample.log`, TensorBoard under `tensorboard/`, and generated images under `sample/`.

## Build, Test, and Development Commands
Create the main training environment:

```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```

Run standard training:

```bash
python train.py --config configs/004_ProMoE_L.yaml
```

Run REPA-enabled training (optional local teacher checkpoint):

```bash
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml --repa-enc-path /path/to/state_dict.pth
```

Optional local VAE path (skip auto-download/cache):

```bash
python train.py --config configs/004_ProMoE_B.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml --vae-path /path/to/sd-vae-ft-mse
```

Precompute VAE latents:

```bash
python preprocess/preprocess_vae.py --latent_save_root /path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz
```

Run sampling:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python sample.py \
  --config configs/004_ProMoE_B_repa.yaml \
  --step_list_for_sample 300000,500000 \
  --guide_scale_list 1.0,1.5 \
  --num_fid_samples 50000
```

Use provided wrappers when they match your target run:

```bash
bash scripts/repa/train_repa_B.sh
bash scripts/repa/sample_and_eval_repa_B.sh

bash scripts/hierar/run_B_hierar_train.sh
bash scripts/hierar/run_B_hierar_infer_eval.sh

bash run_all_infer_eval_500K.sh
```

Create a separate TensorFlow evaluation environment:

```bash
conda create -n promoe_eval python=3.9 -y
conda activate promoe_eval
cd evaluation
pip install -r requirements.txt
python run_eval.py /path/to/generated/images --count 50000
```

`evaluation/run_eval.py` can also skip evaluator execution and only pack PNGs into NPZ:

```bash
python run_eval.py /path/to/generated/images --count 50000 --no-eval
```

## Coding Style & Naming Conventions
- Follow existing Python style in the touched file: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Preserve current import grouping and logging-heavy style.
- Keep config filenames in the existing numeric-prefixed pattern (for example `004_*.yaml`).
- Add new config fields in YAML and rely on `deep_update` to merge only the keys you intend to override.

## Testing Guidelines
There is no dedicated `tests/` directory. Use targeted smoke checks based on your change surface:

- Training-path changes: run `python train.py --config ...` or `python train_with_repa.py --config ...`.
- Sampling-path changes: run `python sample.py --config ...` with explicit step/scale overrides if needed.
- Evaluation-path changes: run `python run_eval.py ...` from inside `evaluation/`.
- Lightweight syntax checks: `python -m py_compile <modified_python_files>`.

If you modify dataset traversal, latent mapping, or preprocessing behavior, regenerate/remove `preprocess/image_paths_cache.txt` before re-running smoke checks so stale cache data does not mask regressions.

## Commit & Pull Request Guidelines
Use concise, imperative, single-scope commit subjects (matching current history style). In PR descriptions, include:

- Affected model family/config(s).
- Whether the path is `train.py` or `train_with_repa.py`.
- Dataset layout assumptions (especially `train/` path and latent sibling path).
- GPU setup assumptions (`gpu_ids`, world size).
- Evidence for behavior changes (logs, sample folders, metric outputs).

## Configuration Notes
- Set `cfg.data_path` in `config.py` (or YAML override) to the ImageNet training root before preprocessing/training.
- At runtime, config stem is injected as `custom_cfg_name` from `--config` filename and used in output path construction.
- Training scripts read `gpu_ids` from YAML and set `CUDA_VISIBLE_DEVICES` internally when present.
- Sampling uses `sample_gpu_ids` only if provided in merged config; otherwise it uses all visible GPUs.

Latent mode and cache behavior:

- With `use_pre_latents=True`, training/preprocessing both use `preprocess/image_paths_cache.txt`.
- Latent file resolution is path-based: each image path replaces the `train` segment with `sd-vae-ft-mse_Latents_256img_npz` and switches extension to `.latent.npz`.
- Keep dataset directory naming aligned with this replacement rule, or update the code accordingly.

Weight caching:

- VAE auto-cache path: `pretrained_ckpt/vae/<hf_repo_id_with_slash_replaced>/` (unless `--vae-path` is provided).
- REPA teacher cache path: `pretrained_ckpt/encoder/<hub_name>/state_dict.pth` (unless `--repa-enc-path` is provided).

Evaluation notes:

- `evaluation/run_eval.py` auto-downloads missing reference batches via `download_ref_batches.py`.
- Run `run_eval.py` from inside `evaluation/` so relative `evaluator.py` lookup works.
