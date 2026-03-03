# Repository Guidelines

## Project Structure & Module Organization
The main training and inference entrypoints live at the repository root: `train.py` handles the standard ProMoE and baseline workflows, `train_with_repa.py` adds REPA projection loss and teacher-encoder loading, and `sample.py` serves both by merging the model registries from the two training scripts. Shared defaults are defined in `config.py`, and reusable helpers such as config merging, VAE loading, and CLI parsers live in `utils.py`. Model implementations stay in `models/`; keep the existing `models_*.py` naming pattern, including the REPA variants in `models/models_ProMoE_TC_repa.py` and `models/models_ProMoE_TC_repa_shared.py`. Experiment YAMLs are in `configs/`, with the config stem becoming `cfg.custom_cfg_name` and part of the output path.

Data preprocessing code is under `preprocess/`, and both preprocessing and training reuse `preprocess/image_paths_cache.txt` as a cached image list. Evaluation code is isolated in `evaluation/`; `evaluation/run_eval.py` can package generated PNGs into an `.npz` and then call the TensorFlow evaluator. Generated checkpoints, TensorBoard logs, and sampled images are written under `outputs/<model_name>/<config_stem>/`. REPA-specific helper code used by `train_with_repa.py` lives in the lowercase `repa/` package, while the uppercase `REPA/` directory is a separate upstream-style subproject with its own entrypoints and its own `AGENTS.md`. Cached teacher encoder weights are stored under `pretrained_ckpt/encoder/`, and `scripts/` contains shell wrappers for the current REPA-B workflow.

## Build, Test, and Development Commands
Create the main training environment and install dependencies:
```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```

Run standard training with a YAML config:
```bash
python train.py --config configs/004_ProMoE_L.yaml
```

Run REPA-enabled training (optionally passing a local teacher checkpoint to skip `torch.hub` download):
```bash
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml --repa-enc-path /path/to/dinov2_state_dict.pth
```

Precompute VAE latents before long runs:
```bash
python preprocess/preprocess_vae.py --latent_save_root /path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz
```

Run sampling for one or more checkpoint steps:
```bash
CUDA_VISIBLE_DEVICES=0 python sample.py --config configs/004_ProMoE_L.yaml --step_list_for_sample 500000 --guide_scale_list 1.0,1.5
```

Use the canned REPA-B helpers when they match the target workflow:
```bash
bash scripts/train_repa_B.sh
bash scripts/sample_and_eval_repa_B.sh
```

For TensorFlow-based metrics, create a separate environment:
```bash
conda create -n promoe_eval python=3.9 -y
conda activate promoe_eval
cd evaluation
pip install -r requirements.txt
python run_eval.py /path/to/generated/images --count 50000
```

## Coding Style & Naming Conventions
Follow the existing Python style in the touched file: 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Preserve the current import grouping and the repository's logging-heavy style instead of rewriting files into a new format. YAML configs use numeric prefixes such as `004_ProMoE_L.yaml`; keep that convention for new experiment files. When you add config fields, mirror the current pattern where YAML values are merged into the global `cfg` object via `deep_update`, so only override the keys that actually need to change.

## Testing Guidelines
There is no dedicated `tests/` directory. Validate changes with targeted smoke checks that match the edited surface area: run `python train.py --config ...` or `python train_with_repa.py --config ...` for training-path changes, run `sample.py` against the affected config for inference changes, and run `python run_eval.py ...` from inside `evaluation/` only when the evaluation pipeline changes. For lightweight sanity checks after Python edits, use `python -m py_compile` on the modified modules. If you change dataset traversal or preprocessing behavior, refresh `preprocess/image_paths_cache.txt` before re-running the smoke test so the cache does not hide a regression.

## Commit & Pull Request Guidelines
Recent commits are short and direct, for example `init repa-shared`, `update pretraind model assigning & add ProMoE-REPA.md`, and `update naive repa`. Keep commit subjects concise, imperative, and scoped to one change. In pull requests, call out the affected model family or config, whether the path is standard or REPA training, which dataset layout and GPU count were assumed, and include evidence such as training logs, sample folders, or metric outputs when behavior changes.

## Configuration Notes
Set `cfg.data_path` in `config.py` to the ImageNet training root before running training or preprocessing. The training scripts honor `gpu_ids` from the YAML by exporting `CUDA_VISIBLE_DEVICES` internally, so the config file is the source of truth unless you intentionally override it outside the script. Output directories are built as `outputs/<model_name>/<config_stem>/`, and `sample.py` reads checkpoints from that tree automatically.

If `cfg.use_pre_latents=True`, keep the latent directory naming aligned with the code path: the training dataset derives latent files by replacing the `train` segment in each image path with `sd-vae-ft-mse_Latents_256img_npz`, so `--latent_save_root` should normally point to that sibling directory. Both preprocessing and training reuse `preprocess/image_paths_cache.txt`; delete or rebuild it after switching datasets or reorganizing files. VAE weights can be provided with `--vae-path` to avoid an automatic download from Hugging Face, and REPA teacher weights can be provided with `--repa-enc-path`; otherwise `train_with_repa.py` will cache DINOv2 weights into `pretrained_ckpt/encoder/` on first use. `evaluation/run_eval.py` auto-downloads the OpenAI reference batch into `evaluation/` if it is missing, so keep the evaluation environment separate from the main PyTorch environment to avoid dependency conflicts.
