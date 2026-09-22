---
name: new-experiment
description: Scaffold a complete ProMoE experiment end-to-end and create its run-time wrapper. For a new model variant it creates the model file, registers it in the correct model_dict, and writes the config; for a config-driven ablation it writes only the config; then it creates the template.sh-based train+sample+eval run script and calls scripts/_run_times/new_run.sh to write the per-date run-time wrapper (which claims its GPUs at launch; the YAML's gpu_ids is only a placeholder fixing the DDP world size). Always previews with new_run.sh --dry-run before writing. Use when the user asks to add/write/create a ProMoE experiment, a new model variant, a config-driven ablation, or a train+sample+eval run script — including phrasings like "写一个 XXX 实验脚本" / "加一个 XXX 消融". Does NOT run real training/sampling/eval, and does NOT commit, push, or amend.
---

# /new-experiment — Scaffold a ProMoE experiment + create its run-time wrapper

Standardizes the full "add an experiment" flow so it is done the same way every time:
**model side (if needed) → semantic run script → validate → create run-time wrapper**, with a
mandatory `--dry-run` preview before anything is written. The mechanical GPU-count/`gpu_ids` handling is
delegated to `scripts/_run_times/new_run.sh`; the run-script shape is delegated to
`scripts/template.sh`. This skill is the orchestration + project-aware checks around them.

This skill **scaffolds and validates only**. It never launches real training, sampling, or
evaluation, and never commits.

## Inputs to gather or infer
Before doing anything, pin down (ask only if genuinely ambiguous — otherwise use the default):
- **Variant + family** — which model and which training entrypoint (`train.py` /
  `train_with_repa.py` / `train_with_MoS_repa.py` / `train_with_mae.py`). See CLAUDE.md "Model Registry".
- **Size** — `B` / `L` / `XL` (maps to `DiT_<size>_config`).
- **Config knobs** — any `MoE_config` / `repa_config` flags the experiment sets.
- **GPU count** — how many GPUs this experiment uses for DDP; any positive integer. Default `4`, **XL defaults to `8`**. Honor an explicit override, and keep every arm of one comparison on the same count (per-rank quantities change with the world size).
- **Date directory** — `scripts/_run_times/<date>/`, format `YYYY_MM_DD`, default **today**.

## Step 0 — Classify the experiment
- **New model variant** → do Step 1A, 1B, 1C.
- **Config-driven ablation** (controlled by an existing flag, `model_name` unchanged) → do Step 1C only.
- **Already defined** (model + config exist, just needs a run script + wrapper) → skip to Step 2.

## Step 1 — Model side

### 1A. Model file
Create `models/models_ProMoE_TC_<variant>.py` (EC-family variants follow
`models_ProMoE_EC_<variant>.py` and inherit from `ProMoE_EC`). Inherit from the closest existing
variant. Follow the `forward()` return conventions (CLAUDE.md "Auxiliary Loss Convention").
If the variant is a cross-alignment model, preserve both "Cross-Alignment Stability Constraints"
(clamp `cos_sim` to `[-1, 1]` after `bmm`; `x.detach()` into block-wise weight predictors).

### 1B. Register in model_dict
Add a `(ModelClass, config_key)` entry to the **correct** training script's `model_dict`
(per family above). `sample.py` merges all four dicts automatically.

### 1C. Config
Create `configs/004_ProMoE_<size>_<variant>.yaml`. Set `model_name` to the registered key.
Add `MoE_config` / `repa_config` nested under the `DiT_<size>_config` key as needed
(REPA has the two-level `repa_config` gotcha — see CLAUDE.md). For a config-driven ablation,
add the new flag with a **default that preserves backward compatibility**.

## Step 2 — Semantic run script
Copy `scripts/template.sh` to `scripts/<family>/run_<size>_<variant>_train_sample_eval.sh`.
Per CLAUDE.md "Shell Script Convention", change **only**: the `CONFIG=` path, the `LOG=` filename,
and the **training entrypoint** in the train step to match the model family. Keep everything else
(`set -euo pipefail`, `SCRIPT_DIR`/`REPO_ROOT`, the sequential train→sample→eval loop) intact.

## Step 3 — Validate (no real runs)
- `python -m py_compile` on every new/edited `.py` (model file + any edited training script).
- `python -c "from models.models_<X> import *"` for each new model file (catches import-time errors). This needs the `promoe` env (it imports `torch`); if `torch` isn't importable in the current shell, fall back to `py_compile`-only and say so rather than reporting a false failure.
- **Four-way consistency**: the new `model_name` exists in exactly the intended `model_dict`; its
  `config_key` is defined in `config.py`; the config's `model_name` matches; the run script's
  `CONFIG=` resolves and its training entrypoint matches the family.
- **Output-dir collision guard (mandatory):** run
  `python scripts/check_output_dir.py --config configs/004_ProMoE_<size>_<variant>.yaml`.
  It derives `outputs/{model_name}/{custom_cfg_name}` (train.py:434/723) and **fails (exit 1)** if
  that dir already exists on local disk or is claimed by another config. On a hit, rename the config
  to the suggested `_vN` name — keeping the run script + wrapper names in lock-step — before
  continuing. **Never point a new experiment at an existing run's output dir.** The local-disk check
  can't see the training server, so if this experiment was already launched there (or it re-uses a
  name whose model code has since changed), bump to `_vN` regardless — that re-run case is exactly
  what `/rerun-experiment` automates.
- Do NOT start training/sampling/eval.

## Step 4 — Create the run-time wrapper (preview gate)
1. **Preview first (mandatory):**
   ```
   scripts/_run_times/new_run.sh --script scripts/<family>/run_<...>.sh \
       [--date <YYYY_MM_DD>] --gpus <N> --dry-run
   ```
   Echo the plan it prints: GPU count, placeholder `gpu_ids`, target config, wrapper path.
2. **Then write** (drop `--dry-run`): patches the experiment YAML's placeholder `gpu_ids`,
   and writes `scripts/_run_times/<date>/<desc>.sh`.
3. **Validate what it wrote**: `bash -n` the generated wrapper, and confirm its `exec` target and
   the config it names both exist. The wrapper sources `claim_gpus.sh` and holds the claiming
   logic, so a syntax check is worth more than it was under the old fixed-id wrappers.
4. Pause for confirmation **only** if intent is ambiguous (GPU count, date dir, or new-vs-existing
   variant). Otherwise the preview is the checkpoint — proceed to write.

GPU selection is owned by `new_run.sh` + `claim_gpus.sh` (it needs `--gpus N`; the wrapper claims N idle GPUs at launch, so no GPU ids are baked in). Do not re-implement that math here.

## Step 5 — Auto-write the experiment description
Immediately after the wrapper is written, invoke **`/describe-experiment`** on it. That skill traces
the wrapper → semantic script → config → model/CLAUDE.md and writes
`scripts/_run_times/<date>/<desc>-describe.txt` — a numbered list of this experiment's core
changes vs the baseline (base ProMoE **and** the immediate parent variant), most-important change
first, bilingual. It is read-only tracing + one `.txt` write; it launches nothing and never commits.

## Step 6 — Report
List every file created or modified (model, config, run script, wrapper, `*-describe.txt`), and the
GPU count the experiment needs. Give the launch command but **do not run it** — per the project rule, the user (or a
later request) starts it in a new tmux window:
```
tmux new-window -t "$(tmux display-message -p '#S')" -n <name> 'bash scripts/_run_times/<date>/<desc>.sh'
```

## Workflow rules (project-wide, see CLAUDE.md)
- **Background processes go to a new tmux window in the current session.** Never `command &`,
  `nohup`, or `run_in_background=true`. If `$TMUX` is unset, abort and ask the user to attach first.
  This skill does not launch runs itself, but any run it hands off must follow this.
- **Clean up smoke-test artifacts immediately.** This skill's validation is `py_compile` + import
  only. If a check ever creates a temp config/output/cache, delete it as soon as the check finishes.

## What this skill must NOT do
- **No real training / sampling / evaluation runs** — scaffold + validate only.
- **No git commits, push, force-push, or amend.** Leave all created files dirty for the user to commit.
- No `git add -A` / `git add .` — if the user later asks to commit, stage explicit paths only.
- No edits to runtime artifact dirs (`outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`,
  `collapse_smoking_test*/`) or the vendored `REPA/` (uppercase) subproject.
- Do not re-implement GPU selection or `gpu_ids` handling — always delegate to `new_run.sh`.
- Do not auto-fire on unrelated shell-script requests or on review/check tasks (those are `/inspect` / `/check`).
