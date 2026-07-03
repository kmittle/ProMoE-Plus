# Repository Guidelines

## Project Overview
ProMoE-Plus implements ProMoE, a Mixture-of-Experts framework for scaling Diffusion Transformers on ImageNet class-conditional generation. The core routing design combines conditional routing for cond/uncond token separation with prototypical routing through learnable cluster centers, plus routing contrastive guidance for expert specialization.

## Project Structure & Module Organization
Core entrypoints are at repository root:

- `train.py`: baseline and non-REPA training (DiT, TCDiT, ECDiT, DiffMoE, ProMoE, hierarchical, expert-choice/batch-choice, structured-batch, proto-t, anchor, proto-choice, noise-expert, and expert-contrastive variants).
- `train_with_repa.py`: REPA-enabled training (REPA / REPA-Shared / REPA-Cond / REPA-DYNA / REPA-DYNA-SELECT / REPA-DYNA-SCALE / REPA-DYNA-ONLY / REPA-Router / REPA-Router-Contra / REPA-Routed / REPA-Double-Share / hierarchical-expert REPA-DYNA), including teacher-feature alignment loss.
- `train_with_MoS_repa.py`: MoS-REPA, MoS-Naive / Naive-Choice, separate-projector / per-block / blockwise / fused / multi-align, and both standard-REPA + MoS cross-alignment training with teacher-block routing and per-block REPA projectors.
- `train_with_mae.py`: MAE/group-alignment training for `group_align` and `group_align_proj` variants.
- `sample.py`: sampling/inference entrypoint. It merges model registries from `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, and `train_with_mae.py`, so one script can sample all registered families.

Shared defaults and helpers:

- `config.py`: global defaults (`cfg`) and model templates (`DiT_*_config`, `DiffMoE_*`, etc.).
- `utils.py`: `deep_update`, `find_free_port`, VAE caching/loading (`load_vae`), `TrainingMonitor`, CLI list parsers, and Inception utilities.
- `scripts/check_output_dir.py`: output-directory collision guard for new or rerun experiments.
- `scripts/template.sh`: required base pattern for new train + sample + eval experiment wrappers.
- `ProMoE-REPA.md`: repo-level guide for current REPA / MoS-REPA workflows and variants.

Main code layout:

- `models/`: architecture implementations; keep the `models_*.py` naming convention.
- `repa/`: REPA helper package used by `train_with_repa.py` and `train_with_MoS_repa.py` (`encoder.py`, `loss.py`).
- `preprocess/`: VAE latent preprocessing and shared cache file `preprocess/image_paths_cache.txt`.
- `evaluation/`: OpenAI-style evaluation pipeline (`run_eval.py`, `evaluator.py`, `download_ref_batches.py`).
- `analyses/`: analysis entry scripts live directly under this directory. Current entrypoints include `run_tokenwise_tsne.py`, `run_samplewise_pooled_tsne.py`, `run_imagewise_tsne.py`, `run_repa_dyna_heatmap.py`, `run_token_choice_expert_heatmap.py`, `run_compute_flops.py`, and `run_mos_routing_analysis.py`. Reusable helpers live under `analyses/t_SNE/`, `analyses/heatmap/`, `analyses/flops/`, and `analyses/mos_routing/`; `analyses/README.md` should remain a brief directory-level overview only, while detailed usage belongs in per-entry Markdown files that share the same basename as each entry `.py`.
- `scripts/repa/`: REPA-B / REPA-Shared-B / REPA-Cond-B helpers plus router / routed / double-share, cross-alignment, and L/XL scale-up train + sample + eval wrappers.
- `scripts/MoS_repa/`: MoS-REPA, naive / naive-choice, per-block / blockwise / fused, multi-align, cross-alignment, and B/L/XL block-range sweep wrappers following `scripts/template.sh`.
- `scripts/hierar/`: B-scale hierarchical/expert train + infer/eval wrappers.
- `scripts/dynamic_repa/`: REPA-DYNA-B, REPA-DYNA-SELECT-B, REPA-DYNA-SCALE-B, and REPA-DYNA-ONLY-B train + sample + eval pipelines (including select-ratio variants r25/r75).
- `scripts/mae_align/`: group-align and group-align-proj train + sample + eval wrappers.
- `scripts/noise_expert/`: noise-expert, proj, EMA-on-shared, and EMA-on-noise train + sample + eval wrappers.
- `scripts/expert_contra/`: expert-contrastive output/param train + sample + eval wrappers, including B4 ablations.
- `scripts/expert_choice/`: expert-choice batch-choice (`EC_BC`) train + sample + eval wrappers.
- `scripts/structured_batch/`: token-choice and expert-choice structured-batch train + sample + eval wrappers.
- `scripts/proto_t/`: token-choice and expert-choice proto-t direct/residual train + sample + eval wrappers.
- `scripts/anchor/`: anchor-routing and anchor-replace train + sample + eval wrappers.
- `scripts/proto_choice/`: contrastive proto-choice ratio sweep train + sample + eval wrappers.
- `scripts/_run_times/`: timestamped launch indirection for scheduled experiment batches.
- `collapse_smoking_test/`, `collapse_smoking_test_10k/`: crash-diagnosis smoke configs, logs, summaries, and rerun helpers for cross-alignment stability work.
- `tb_smoke_200/`, `tb_smoke_500/`: TensorBoard/`TrainingMonitor` smoke harnesses for selected cross-alignment configs.
- `REPA/` (uppercase): separate upstream-style subproject with its own docs and `AGENTS.md`.

Top-level wrappers:

- `scripts/run_all_infer_eval_500K.sh`: batch sample+eval for multiple configs at 500K.
- `scripts/eval_B_hierar_expert.sh`, `scripts/eval_B_hierar_expert_NoPenalty.sh`: eval-only wrappers.

Companion documentation:

- `ProMoE-REPA.md`: detailed REPA / MoS-REPA workflow, configuration reference, and FAQ.
- `analyses/README.md`: analysis entrypoint overview; keep per-script usage in matching `analyses/<basename>.md` files.
- `plans/`: implementation plans for standard REPA and MoS cross-alignment variants.
- `implementation-plan.md`: Chinese draft plan for a future attention-weighted same-expert same-image alignment family; reference only, not current code.

Outputs follow:

`outputs/<model_name>/<config_stem>/`

with:

- `checkpoints/ckpt_step_*.pth`
- `training.log`, `sample.log`
- `tensorboard/`
- `sample/step<step>/img<...>/images`
- `sample/step<step>/<analysis_name>/` for analysis artifacts such as `flops_eval/` and t-SNE outputs

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

Run MAE/group-alignment training:

```bash
python train_with_mae.py --config configs/004_ProMoE_B_mae_align.yaml
```

Optional local VAE path (skips auto-download/cache in train/sample):

```bash
python train.py --config configs/004_ProMoE_B.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_MoS_repa.py --config configs/004_ProMoE_B_repa_MoS.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_mae.py --config configs/004_ProMoE_B_mae_align.yaml --vae-path /path/to/sd-vae-ft-mse
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

Run FLOPs / activated-parameter / expert-frequency analysis:

```bash
python analyses/run_compute_flops.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --num-samples-per-class 5 \
  --guide-scale 1.0 \
  --save-every-steps 50
```

Run MoS routing analysis:

```bash
python analyses/run_mos_routing_analysis.py \
  --ckpt outputs/ProMoE_TC_REPA_MoS_Naive_Choice_B/004_ProMoE_B_repa_MoS_naive_choice_b3_5/checkpoints/ckpt_step_500000.pth
```

Common wrappers:

```bash
bash scripts/repa/train_repa_B.sh
bash scripts/repa/train_repa_shared_B.sh
bash scripts/repa/train_repa_cond_B.sh

bash scripts/repa/sample_and_eval_repa_B.sh
bash scripts/repa/sample_and_eval_repa_shared_B.sh
bash scripts/repa/sample_and_eval_repa_cond_B.sh
bash scripts/repa/run_B_repa_cross_global_pre_train_sample_eval.sh
bash scripts/repa/run_B_repa_cross_global_block_train_sample_eval.sh
bash scripts/repa/run_B_repa_cross_expert_local_train_sample_eval.sh
bash scripts/repa/run_B_repa_cross_proto_train_sample_eval.sh
bash scripts/repa/run_B_repa_router_train_sample_eval.sh
bash scripts/repa/run_B_repa_router_contra_train_sample_eval.sh
bash scripts/repa/run_B_repa_routed_train_sample_eval.sh
bash scripts/repa/run_B_repa_double_share_train_sample_eval.sh

bash scripts/MoS_repa/run_B_repa_mos_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_b3_5_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_sep_b3_5_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_choice_per_block_b3_5_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_blockwise_b3_5_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_b3_5_fused_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_multi_align_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_multi_align_no_dynamic_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_cross_global_pre_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_cross_global_block_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_cross_expert_local_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_cross_proto_train_sample_eval.sh

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
bash scripts/hierar/run_B_hierar_expert_repa_dyna_train_sample_eval.sh

bash scripts/mae_align/run_B_mae_align_train_sample_eval.sh
bash scripts/mae_align/run_B_mae_align_proj_train_sample_eval.sh
bash scripts/noise_expert/run_B_noise_expert_train_sample_eval.sh
bash scripts/noise_expert/run_B_noise_expert_proj_train_sample_eval.sh
bash scripts/noise_expert/run_B_noise_expert_ema_on_shared_train_sample_eval.sh
bash scripts/noise_expert/run_B_noise_expert_ema_on_noise_train_sample_eval.sh
bash scripts/expert_choice/run_B_ec_bc_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_output_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_output_b4_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_param_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_param_b4_train_sample_eval.sh
bash scripts/structured_batch/run_B_tc_structbatch_train_sample_eval.sh
bash scripts/structured_batch/run_B_ec_bc_structbatch_train_sample_eval.sh
bash scripts/proto_t/run_B_tc_proto_t_direct_v2_train_sample_eval.sh
bash scripts/proto_t/run_B_tc_proto_t_residual_v2_train_sample_eval.sh
bash scripts/proto_t/run_B_ec_bc_proto_t_direct_v2_train_sample_eval.sh
bash scripts/proto_t/run_B_ec_bc_proto_t_residual_v2_train_sample_eval.sh
bash scripts/anchor/run_B_anchor_routing_train_sample_eval.sh
bash scripts/anchor/run_B_anchor_replace_train_sample_eval.sh
bash scripts/proto_choice/run_B_proto_choice_083_train_sample_eval.sh
bash scripts/proto_choice/run_B_proto_choice_125_train_sample_eval.sh

bash tb_smoke_200/run_all.sh
bash tb_smoke_500/run_all.sh

bash scripts/run_all_infer_eval_500K.sh
```

When adding any new train + sample + eval all-in-one `.sh` wrapper, follow the structure and execution pattern of `scripts/template.sh` rather than inventing a new style; this is required for compatibility with another experiment server. All such experimental `.sh` wrappers must launch Python exactly in the template style: use `/mnt/workspace/yujie/.conda/envs/promoe/bin/python` for training/sampling and `/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python` for evaluation, and do not rely on `conda activate` at runtime.

Template-specific requirements:

- `scripts/template.sh` uses a sequential train-stop-sample/eval-resume loop. For each `step_list_for_sample` item, it trains to that checkpoint, exits, frees GPUs for sample/eval, then resumes training for the next step.
- New scripts should only change `CONFIG`, `LOG`, and the training entrypoint (`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`) unless the experiment genuinely needs extra logic.
- Preserve `set -euo pipefail`, repo-root discovery via `SCRIPT_DIR` / `REPO_ROOT`, inline Python YAML parsing, absolute Python interpreter paths, and `find ... -name images | sort -V` evaluation traversal.
- Legacy split scripts such as `scripts/repa/train_repa_B.sh` and `scripts/repa/sample_and_eval_repa_B.sh` predate the template; do not introduce new split-purpose experiment scripts.

Runtime GPU-slot grouping:

- After creating a semantic experiment wrapper under `scripts/<family>/`, allocate its launch wrapper with `scripts/_run_times/new_run.sh --script scripts/<family>/run_<...>.sh [--date YYYY_MM_DD] [--gpus 4|8] [--dry-run]` rather than hand-editing `gpu_ids`.
- `scripts/_run_times/new_run.sh` patches the experiment YAML `gpu_ids` and writes `scripts/_run_times/<date>/<slot>-<desc>.sh`; GPU assignment lives in YAML, not the wrapper.
- Slot names map to one physical 8-GPU server: `X.1` means GPUs `0-3`, `X.2` means GPUs `4-7`, and full-slot `X` means GPUs `0-7`.
- Allocation is scoped to one date directory only. Do not run jobs from different date directories on the same physical GPUs unless you have checked the assignments manually.
- Use `--dry-run` first when scheduling a new run-time wrapper so the slot and YAML patch are visible before writing.

Create evaluation env (TensorFlow-based):

```bash
conda create -n promoe_eval python=3.9 -y
conda activate promoe_eval
cd evaluation
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
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
- Wrapper/config names like `b3_5` are human-readable 1-indexed block ranges; YAML `align_blocks` stays 0-indexed Python-style (for example `b3_5` pairs with `align_blocks: [2, 3, 4]`).
- Add overrides in YAML and rely on `deep_update` to only replace intended keys.

## Adding or Rerunning Experiments
For config-only ablations, create a new YAML and a template-based shell wrapper while keeping `model_name` unchanged. Add config flags with defaults that preserve existing behavior, and allocate a run-time slot through `scripts/_run_times/new_run.sh`.

For new model variants:

1. Add the model under `models/models_ProMoE_TC_<variant>.py` or, for Expert-Choice family work, `models/models_ProMoE_EC_<variant>.py`.
2. Register the `model_name` in the appropriate `model_dict` in `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`; `sample.py` merges all registries automatically.
3. Add `configs/004_ProMoE_<size>_<variant>.yaml`, keeping `model_name`, config filename, and wrapper name aligned.
4. Create a `scripts/<family>/run_<size>_<variant>_train_sample_eval.sh` wrapper from `scripts/template.sh`.
5. Validate with `python -m py_compile <modified_python_files>` and `python scripts/check_output_dir.py --config <config>`.
6. Allocate the launch wrapper with `scripts/_run_times/new_run.sh --script <wrapper> [--gpus 4|8]`.

Before rerunning an experiment after model-code changes, do not reuse the old config name if an output directory already exists. Use `python scripts/check_output_dir.py --config <config> --suggest-version` and move the config, semantic script, and run-time wrapper together to a fresh `_vN` name so the new run writes to a clean `outputs/<model_name>/<custom_cfg_name>/` directory.

## Testing Guidelines
No dedicated `tests/` directory. Use smoke checks aligned to your change surface:

- Training changes: `python train.py --config ...`, `python train_with_repa.py --config ...`, `python train_with_MoS_repa.py --config ...`, or `python train_with_mae.py --config ...`.
- Sampling changes: `python sample.py --config ...` (with `--step_list_for_sample` / `--guide_scale_list` as needed).
- Evaluation changes: run from inside `evaluation/`: `python run_eval.py ...`.
- Analysis changes: run the relevant entry script under `analyses/`; for FLOPs/statistics, use `python analyses/run_compute_flops.py --help` for import/CLI validation or a checkpoint-backed smoke run with `--ckpt ...`.
- Syntax checks: `python -m py_compile <modified_python_files>`.
- End-to-end REPA-DYNA smoke check: run one wrapper in `scripts/dynamic_repa/` and verify train/sample/eval logs are produced.
- End-to-end MoS-REPA smoke check: run `bash scripts/MoS_repa/run_B_repa_mos_train_sample_eval.sh` and verify train/sample/eval logs are produced.
- Cross-alignment / monitoring changes: run `bash tb_smoke_200/run_all.sh` or `bash tb_smoke_500/run_all.sh` and verify `monitor/` TensorBoard scalars are emitted.
- One-click experimental wrappers should keep the hard-coded interpreter launch style from `scripts/template.sh`: `/mnt/workspace/yujie/.conda/envs/promoe/bin/python` for training/sampling and `/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python` for evaluation.
- When writing a new training + sampling + evaluation three-in-one shell script, start from `scripts/template.sh`, preserve its interpreter-launch pattern, swap the training entrypoint as needed (`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`), and do not replace it with `conda activate`; otherwise the script may fail on the experiment server.
- For new analysis entrypoints, add a matching `analyses/<basename>.md` usage guide and keep shared logic in a subpackage under `analyses/` rather than embedding everything in the root script.

If you touch dataset traversal, latent mapping, or preprocessing logic, clear/regenerate `preprocess/image_paths_cache.txt` before re-running checks.

## Workflow Rules
- Clean up smoke-test artifacts immediately after a smoke or sanity run finishes, whether it succeeds or fails. Remove temporary configs, temporary scripts, and smoke-only output directories such as `outputs/<model>/<smoke_cfg>/`; do not delete long-lived real outputs, `pretrained_ckpt/`, `training_logs/`, or project-level caches unless explicitly requested.
- Run long-lived background processes in a new tmux window of the current session. Do not use `command &`, `nohup`, or detached shell backgrounding for training, sampling, long evals, watch loops, or dev servers.
- Before launching a long-lived tmux command, require an attached tmux session:

```bash
test -n "${TMUX:-}" || { echo "not inside tmux - attach first"; exit 1; }
tmux new-window -t "$(tmux display-message -p '#S')" -n <name> '<command>'
```

- If `$TMUX` is unset, abort and ask the user to attach to tmux first. Short synchronous commands such as `ls`, `rg`, `py_compile`, and `git status` should still run in the foreground.
- Treat the vendored `REPA/` uppercase subproject as separate scope. Do not edit it unless the user explicitly asks for that subproject.

## Commit & Pull Request Guidelines
Use concise, imperative, single-scope commit subjects. In PR descriptions, include:

- Affected model family/config(s).
- Whether the path is `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`.
- Dataset layout assumptions (especially `train/` path and latent sibling path).
- GPU assumptions (`gpu_ids`, world size, sampling GPUs).
- Evidence of behavior change (logs, sample folders, metric outputs).

When the user asks to push, treat that as permission to commit relevant current WIP first, then push. Still stage explicit paths only; do not use `git add -A` or `git add .`, do not stage secrets or `*.local.json`, do not bypass hooks with `--no-verify`, and do not force-push to `main` or `master`.

## Configuration Notes
- Set `cfg.data_path` (or the `PROMOE_DATA_PATH` env var, or a YAML override) to the ImageNet `train/` root before preprocess/train; it defaults to the shared `/lustre01/yujie/dataset/imagenet/train`. This absolute path is `train`-once-safe — no component but the `train/` dir contains the substring 'train' — which train.py's latent derivation via `str.replace('train', ...)` requires (keep any override the same way).
- On a fresh server, run `preprocess/prepare_imagenet.sh` **once manually** before launching a batch (it is intentionally NOT auto-invoked by the run scripts — two co-scheduled 4-GPU slots would otherwise both trigger download/encode, and the second would idle-block on the `flock`). It auto-downloads full-resolution ImageNet-1K (HuggingFace→ModelScope), materialises it to `/lustre01/yujie/dataset/imagenet/train/<label:04d>/`, provisions the VAE, and VAE-encodes everything — idempotent, resume-safe, `flock`-locked as a safety net, and it verifies every image has a latent before finishing. Training then reads `/lustre01/yujie/dataset/imagenet/train` via `config.py`'s default. The data lives outside the repo (nothing to commit).
- Parquet-direct mode: if raw HF parquet shards already exist (default `/lustre01/qianyuan/data/ILSVRC/imagenet-1k/data`, override `PROMOE_PARQUET_DIR` / `--parquet-dir`), `prepare_imagenet.py` auto-detects them and encodes latents DIRECTLY from parquet via `preprocess/encode_latents_from_parquet.py` (shard-parallel, per-file resume), skipping re-download and the intermediate JPEG folder. Output `<latent_root>/<label:04d>/*.latent.npz` uses the same 8-channel `.latent_dist.parameters` format as `preprocess_vae.py`. Training reads it via the `LatentFolder` dataset when a config sets `use_encoded_latents: True` (label = `int(<label:04d> dir)`; no image folder, no `replace('train',...)`); latent path from `cfg.latent_data_path` / `PROMOE_LATENT_PATH`. The 2026_07_01 non-REPA configs set the flag; REPA still needs the JPEG path. `preprocess/latent_paths_cache.txt` is the LatentFolder equivalent of the image cache (prepare rebuilds it sorted).
- `custom_cfg_name` is auto-injected from `--config` filename stem and used in output path construction.
- Training uses `gpu_ids` from YAML to set `CUDA_VISIBLE_DEVICES` when provided.
- Sampling uses `sample_gpu_ids` only if provided; otherwise it uses all visible GPUs.
- `train_with_repa.py` and `train_with_MoS_repa.py` read REPA behavior from top-level `repa_config`; model-level REPA knobs live under `DiT_*_config.repa_config` in YAML.
- MoS-REPA configs (for example `004_ProMoE_B_repa_MoS.yaml`) additionally set `DiT_*_config.repa_config.num_teacher_blocks`; keep it aligned with the chosen teacher encoder depth.
- Dynamic-select configs (`004_ProMoE_B_repa_dyna_select*.yaml`) control token selection via `DiT_B_config.repa_config.repa_select_ratio`.
- Router configs use model-level REPA knobs such as `router_repa_coeff` (`004_ProMoE_B_repa_router.yaml`) or `router_loss_decay_steps` (`004_ProMoE_B_repa_router_contra.yaml`).
- `004_ProMoE_B_repa_dyna_only.yaml` also carries a model-level `DiT_B_config.repa_config.proj_coeff` for the capped dynamic-weight ablation.
- Cross-alignment configs use model names such as `ProMoE_TC_REPA_CROSS_GLOBAL_PRE_B`, `ProMoE_TC_REPA_CROSS_GLOBAL_BLOCK_B`, `ProMoE_TC_REPA_CROSS_EXPERT_LOCAL_B`, `ProMoE_TC_REPA_CROSS_PROTO_B`, and their `ProMoE_TC_REPA_MoS_*` counterparts.
- MoS-REPA block-range configs may vary `align_blocks`, projector sharing, router norm, fused routing, dynamic coefficients, or `proj_coeff`; keep wrapper names and YAML values aligned.
- `proto_t` configs choose `proto_t_update_mode` (`direct` or `residual`) under `DiT_B_config`; the `EC_BC` proto-t configs use the expert-choice batch-choice model registry key.
- Anchor configs set `anchor_apply_mode` (`routing` or `replace`) under `DiT_B_config`.
- Proto-choice ratio configs set `contrastive_proto_choice_ratio`; wrapper/config suffixes such as `083` and `125` refer to the ratio sweep values.
- Load-balance-contrastive (lbcontra) configs set `lb_contra_mode` (`reweight`/`logit_adjust`/`balance_term`/`soft_only`) and the mode's scalar (`lb_reweight_beta` / `lb_logit_adj_tau` / `lb_balance_lambda`); only the routing-contrastive loss changes, so step-0-identical to base ProMoE.
- DAG-Fuse (dagfuse) configs set `fusion_arm` (`cond_from_shared`/`shared_from_cond`/`bidirectional`) and `fusion_num_iter`; the fusion module's up_proj is zero-init, so step-0-identical (non-strict checkpoint load).
- Adaptive-depth (adepth) configs set `alloc_mode=fixed_q`, `depth_q`, and `depth_warmup`; requires `top_k==1`, and zero-init gates keep it step-0-identical (non-strict).
- Loss-free (lossfree) configs set `use_lossfree_bias` and `bias_update_rate` (`u`); the per-prototype bias is a non-trainable buffer added only to top-1 selection, step-0-identical (non-strict).
- Most provided YAMLs set `resume_checkpoint: True`; when no checkpoint exists the loader logs an error and training starts from step 0.
- `sample.py` behavior:
  - if `step_list_for_sample` is set, it loads only those checkpoints;
  - otherwise it scans `checkpoints/` and loads steps divisible by `sample_every_step`;
  - `--num_fid_samples` also updates `save_img_num`.

Model registry and forward conventions:

- `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, and `train_with_mae.py` each define a `model_dict` from `model_name` to `(ModelClass, config_key)`. Add new registrations in the training script that owns the family.
- `train.py` hosts base DiT/baselines, ProMoE TC/EC, EC batch-choice, proto-t, anchor, proto-choice, lbcontra (load-balance-aware routing contrastive), dagfuse (DAG-MoE shared↔cond fusion), adepth (adaptive routed-FFN depth), lossfree (loss-free balancing bias), structured-batch, noise-expert, and expert-contrastive families.
- `train_with_repa.py` hosts standard REPA, REPA shared/cond, dynamic REPA, router/routed/double-share REPA, and hierarchical-expert REPA-DYNA families.
- `train_with_MoS_repa.py` hosts MoS, naive/choice/separate/blockwise/per-block/fused variants, multi-align, and standard REPA + MoS cross-alignment families.
- `train_with_mae.py` hosts `group_align` and `group_align_proj` families.
- Plain DiT and most ProMoE variants return a tensor. REPA variants return `(pred, zs_proj)` during training. MoS-REPA, multi-align, fused MoS, and cross-alignment variants return `(pred, alignment_loss)` during training.
- ProMoE routing contrastive loss generally flows through the `AddAuxiliaryLoss` autograd wrapper even when `forward()` returns a plain tensor.

REPA config scope:

- Top-level `repa_config` belongs to the training loop and controls teacher loading, such as `enc_type`, and global alignment weighting such as `proj_coeff`.
- Nested `DiT_*_config.repa_config` belongs to the model and controls projectors, `encoder_depth`, `z_dims`, `align_blocks`, `num_teacher_blocks`, router REPA knobs, and dynamic/select behavior.
- Keep `enc_type` and teacher block depth consistent across both scopes for REPA and MoS-REPA configs.

Cross-alignment stability notes:

- In cross-alignment loss code, clamp cosine-similarity matrices to `[-1, 1]` after normalization and `torch.bmm`; bf16 precision can otherwise create values slightly outside the valid range and trigger loss spikes.
- For block-wise cross-alignment weight predictors (`cross_global_block` and `cross_expert_local`, including MoS counterparts), feed detached block outputs into the attention/weight module while keeping the projection path on the original tensor. This prevents competing gradient paths into the same DiT block.
- `cross_global_pre` is exempt because attention is applied before DiT blocks; `cross_proto` is exempt because weights come from MoE routing rather than a dedicated weight predictor.
- Use `TrainingMonitor` from `utils.py` for cross-alignment crash diagnosis. Wire it after `backward()` and gradient clipping, before `zero_grad()`, and pass the existing TensorBoard writer when available so `monitor/*` scalars are emitted.

Variant-specific notes:

- `anchor_apply_mode` is `routing` or `replace`; anchor variants are not step-0-identical to base ProMoE because anchors are randomly initialized, so train them fresh.
- `contrastive_proto_choice_ratio` controls proto-choice positive-set size; suffixes such as `083` and `125` map to ratios such as `0.083` and `0.125`.
- `proto_t_update_mode` supports `direct` and `residual`; script/config names remain human-readable and 1-indexed while YAML `align_blocks` remains 0-indexed.
- `noise_expert_ema` parameters are frozen and updated through EMA after optimizer steps; they should stay excluded from the optimizer.

Latent mode and cache behavior:

- With `use_pre_latents=True`, both training and preprocessing rely on `preprocess/image_paths_cache.txt`. `prepare_imagenet.py` rebuilds this cache deterministically (sorted, atomic write) before preprocessing so DDP ranks don't race to regenerate it; `preprocess_vae.py` also skips images whose `.latent.npz` already exists and writes latents atomically.
- Latent path rule in training is string-based: image path replaces `train` with `sd-vae-ft-mse_Latents_256img_npz`, extension becomes `.latent.npz`.
- Keep dataset naming aligned with this replacement rule, or update the code.
- In REPA training with `use_pre_latents=True`, the dataset additionally loads raw images for teacher feature extraction.
- In MoS-REPA training, teacher features are extracted from all teacher blocks and aligned block-wise against every DiT block.

Weight caching:

- VAE auto-cache path: `pretrained_ckpt/vae/<hf_repo_id_with_slash_replaced>/` (unless `--vae-path` is passed).
- REPA teacher cache path: `pretrained_ckpt/encoder/<hub_name>/state_dict.pth` (unless `--repa-enc-path` is passed).
- For REPA and MoS-REPA, rank 0 performs initial teacher download/cache, then other ranks load from local cache after barrier.

Analysis notes:

- `analyses/run_compute_flops.py` resolves the YAML config from the checkpoint path, reads `gpu_ids` from that YAML, and sets `CUDA_VISIBLE_DEVICES` before spawning workers.
- The FLOPs analysis entrypoint accepts both `--ckpt` and the legacy positional checkpoint argument; hyphenated and underscore flag spellings are both supported for its sampling/reporting options.
- FLOPs/statistics outputs are written under `outputs/<model_name>/<config_stem>/sample/step<ckpt_step>/flops_eval/`, including `flops_result.txt`, expert-frequency plots, and optional per-step reports.
- `analyses/run_mos_routing_analysis.py` infers the YAML config from `--ckpt`, does not require a teacher encoder, and writes plots plus `metadata.yaml` under `outputs/<model_name>/<config_stem>/sample/step<ckpt_step>/mos_routing/`.

Evaluation notes:

- `evaluation/run_eval.py` always calls `ensure_ref_batches()` to auto-download missing reference NPZs.
- It expects generated PNG names containing class suffix like `_class123.png`.
- It writes `<image_folder>.npz`, and when evaluation runs it also writes `<image_folder>_eval_openai.txt`.
- Run it from inside `evaluation/` so relative `evaluator.py` lookup succeeds.
