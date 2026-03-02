# Repository Guidelines

## Project Structure & Module Organization
This repository is a PyTorch training codebase for REPA. Core training and sampling entrypoints live at the repo root: `train.py`, `train_t2i.py`, `generate.py`, `generate_t2i.py`, `samplers.py`, and `samplers_t2i.py`. Shared training logic is in `dataset.py`, `loss.py`, and `utils.py`. Model definitions are grouped under `models/` (`sit.py`, `mmdit.py`, encoder backbones). Data preparation utilities live in `preprocessing/`, including a separate `preprocessing/README.md`. There is currently no dedicated `tests/` directory.

## Build, Test, and Development Commands
Set up the environment with:
```bash
conda create -n repa python=3.9 -y
conda activate repa
pip install -r requirements.txt
```
Run ImageNet training with `accelerate launch train.py --data-dir <path> --output-dir exps --exp-name <name>`. Run MS-COCO text-to-image training with `accelerate launch train_t2i.py --data-dir <path> --output-dir exps --exp-name <name>`. Run sampling or FID generation with `torchrun --nnodes=1 --nproc_per_node=8 generate.py --ckpt <checkpoint> ...`. Use `python -m py_compile train.py train_t2i.py` as a quick syntax check before submitting changes.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions, variables, and module filenames, and `PascalCase` for classes. Keep new training or utility scripts descriptively named at the repository root only when they are true entrypoints; otherwise place reusable code in `models/` or shared modules. Match the current PyTorch-first style and keep argument names consistent with existing CLI flags. No formatter or linter is configured here, so keep edits minimal and consistent.

## Testing Guidelines
There is no formal automated test suite in this repository yet. For code changes, prefer lightweight smoke checks: compile modified files, run a short local launch on a small dataset slice, and confirm logs/checkpoints are written under `exps/`. If you add tests, place them under a new `tests/` directory and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent history uses short, direct commit subjects such as `Update generate.py`, `fix typo in unpatchify`, and `512x512 support`. Use concise, imperative commit messages focused on one change. Pull requests should state the training or sampling behavior affected, list exact commands used for validation, and note any dataset, checkpoint, or preprocessing assumptions. Do not commit datasets, model weights, API tokens, or machine-specific absolute paths.
