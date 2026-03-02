# Repository Guidelines

## Project Structure & Module Organization
Core entrypoints live at the repository root: `train.py` for distributed training, `sample.py` for image generation, `config.py` for defaults, and `utils.py` for shared helpers. Model implementations are in `models/`, with one file per variant (for example `models/models_ProMoE_TC.py`). Experiment YAMLs live in `configs/` and are loaded with `--config`. Data preparation helpers are in `preprocess/`, and TensorFlow-based metrics code is isolated in `evaluation/`. Runtime artifacts are written under `outputs/{model_name}/{config_name}/`.

## Build, Test, and Development Commands
Create the main environment and install dependencies:
```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```
Run training with a checked-in experiment config:
```bash
python train.py --config configs/004_ProMoE_L.yaml
```
Generate samples for a smoke test:
```bash
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml
```
Optional: precompute VAE latents before long runs:
```bash
python preprocess/preprocess_vae.py --latent_save_root /path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz
```
For metrics, use the separate `evaluation/requirements.txt` environment, then run `python evaluation/run_eval.py /path/to/generated/images`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Keep model files on the established `models_*.py` pattern and preserve numeric experiment prefixes in config names such as `004_ProMoE_L.yaml`. No formatter or linter config is checked in, so match the surrounding file style and keep imports and logging patterns consistent with the current scripts.

## Testing Guidelines
There is no dedicated `tests/` directory or enforced coverage gate. Validate changes with targeted smoke tests: launch `train.py` with the affected config, run a short `sample.py` pass, and verify `evaluation/run_eval.py` in the separate evaluation environment if metrics code changed. For config edits, test the exact YAML you touched.

## Commit & Pull Request Guidelines
Recent history uses short, direct subjects (`update`, `fix bug`, `init main`). Keep commit titles concise, imperative, and focused on one change. Pull requests should state the affected model or config, note required data or GPU assumptions, and include evidence for behavior changes (sample outputs, metric deltas, or log snippets). Link related issues or experiment notes when available.

## Configuration & Environment Notes
Set `cfg.data_path` in `config.py` before training. Keep TensorFlow-based evaluation dependencies separate from the main PyTorch environment to avoid version conflicts.
