# Repository Guidelines

## Project Structure & Module Organization
Core entrypoints live at the repository root: `train.py` for distributed training, `sample.py` for image generation, `train_with_repa.py` for REPA-enabled training, `config.py` for defaults, and `utils.py` for shared helpers. Model definitions are in `models/`, typically one variant per file (for example, `models/models_ProMoE_TC.py`). Experiment configs are stored in `configs/` and selected with `--config`. Data preparation scripts live in `preprocess/`, while TensorFlow-based evaluation code is isolated in `evaluation/`. Generated checkpoints, samples, and logs are written under `outputs/`. The `REPA/` directory is a separate subproject; treat changes there as scoped work.

## Build, Test, and Development Commands
Create the main environment and install dependencies:
```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```
Run a standard training job:
```bash
python train.py --config configs/004_ProMoE_L.yaml
```
Run a local sampling smoke test:
```bash
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml
```
Optional: precompute VAE latents before long runs:
```bash
python preprocess/preprocess_vae.py --latent_save_root /path/to/latents
```
For metrics, use a separate environment in `evaluation/` and run `python evaluation/run_eval.py /path/to/generated/images`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Keep model files on the established `models_*.py` naming pattern. Preserve numeric experiment prefixes in config names such as `004_ProMoE_L.yaml`. No formatter or linter config is checked in, so match the surrounding import order, logging style, and argument patterns in the file you edit.

## Testing Guidelines
There is no dedicated `tests/` directory or enforced coverage target. Validate changes with targeted smoke tests: run the affected training entrypoint, execute a short `sample.py` pass, and use `evaluation/run_eval.py` only if evaluation code changed. For config-only edits, test the exact YAML you modified.

## Commit & Pull Request Guidelines
Recent commits use short, direct subjects such as `update`, `fix bug`, and `init main`. Keep commit titles concise, imperative, and focused on one change. Pull requests should name the affected model or config, note dataset and GPU assumptions, and include evidence for behavior changes such as sample outputs, metric deltas, or log excerpts.

## Configuration Notes
Set `cfg.data_path` in `config.py` before training. Keep TensorFlow evaluation dependencies separate from the main PyTorch environment to avoid version conflicts.
