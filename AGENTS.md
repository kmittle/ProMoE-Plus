# Repository Guidelines

## Project Structure & Module Organization
`train.py`, `sample.py`, and `config.py` are the main entry points for training, sampling, and shared runtime configuration. Model implementations live in `models/` (`models_ProMoE_*`, `models_DiT.py`, `models_DiffMoE.py`), experiment YAMLs live in `configs/`, preprocessing lives in `preprocess/preprocess_vae.py`, and metric code lives in `evaluation/`. Treat `pretrained_ckpt/` and `outputs/` as artifact directories for checkpoints and generated results, not as source directories.

## Build, Test, and Development Commands
Create the training environment with `conda create -n promoe python=3.10 -y` and install dependencies with `pip install -r requirements.txt`. Start training with `python train.py --config configs/004_ProMoE_L.yaml`. Run sampling with `CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml`. For evaluation, install the isolated metric dependencies from `evaluation/requirements.txt`, then run `CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py /path/to/generated/images`.

## Coding Style & Naming Conventions
Follow standard Python style: 4-space indentation, `snake_case` for modules/functions/variables, `CamelCase` for classes, and uppercase for constants. Keep new model files consistent with the existing `models/models_<Variant>.py` pattern. Name config files with the current numbered convention, for example `004_ProMoE_B_hierar.yaml`. Prefer small, explicit config changes over hidden defaults, and keep CLI flags aligned with `config.py`.

## Testing Guidelines
There is no committed `tests/` directory yet, so validate changes with targeted smoke tests. For training changes, run a short `train.py` launch with the affected config. For inference changes, run `sample.py` with reduced sample counts or a single checkpoint step. For metric changes, verify `evaluation/run_eval.py` against a small generated image set. If you introduce automated tests, add `pytest`-style files under a new `tests/` directory and use `test_*.py` filenames.

## Commit & Pull Request Guidelines
Recent commits use short, lowercase subjects such as `update hierar` and `modify hierar routing`. Keep that style concise and imperative, but make the scope clearer when possible, for example `add EC routing config`. Pull requests should state which model or config changed, list the commands you ran, note expected training or metric impact, and include sample outputs or evaluation deltas when behavior changes.

## Configuration & Data Hygiene
Do not commit machine-specific dataset paths, cached latents, generated images, or large checkpoints. Keep `cfg.data_path` updates local to your environment, and avoid hardcoding absolute paths outside local experiments.
