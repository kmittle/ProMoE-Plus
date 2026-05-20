---
name: check
description: Run a project-wide carpet-style code-quality loop on ProMoE-Plus — for each iteration do (scan → fix → commit → smoke test), repeating until 5 consecutive iterations find zero issues. Use when the user invokes /check or asks for a thorough sweep before a milestone.
---

# /check — Iterative carpet check loop

Sweep the ProMoE-Plus codebase for issues, fix them, commit the fixes, then run a smoke test. Repeat until **5 consecutive iterations** produce zero findings AND a passing smoke test. Hard cap: **20 iterations total** — if hit, stop and report what remains.

## State to maintain across iterations
- `iter`: 1-indexed iteration counter
- `consecutive_clean`: resets to 0 on any finding or smoke-test failure
- `commits[]`: SHAs of commits this skill has produced this run
- `findings_history[]`: short notes per iteration, for the final report

## One iteration

### 1. Carpet scan
Run in parallel where possible. Surface every issue found; do not silently skip.

**Syntax + import graph**
- `python -m py_compile` every `.py` under `models/`, `repa/`, `analyses/`, plus top-level `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, `train_with_mae.py`, `sample.py`, `utils.py`, `config.py`, `preprocess/preprocess_vae.py`.
- For every `models/models_*.py` file, attempt `python -c "import importlib; importlib.import_module('models.<basename>')"` to catch import-time errors that py_compile misses (missing symbols, circular imports).

**Cross-reference drift** (the high-signal checks for this repo)
- For each entry in `model_dict` of `train.py` / `train_with_repa.py` / `train_with_MoS_repa.py` / `train_with_mae.py`: verify the imported `ModelClass` exists in the named module and the `config_key` is defined in `config.py`. Flag orphans either direction (model class with no `model_dict` entry, or `model_dict` entry whose model file is gone).
- For each YAML in `configs/`: verify `model_name` matches a registered key across the four training scripts (the union — `sample.py` merges them).
- For each shell script in `scripts/**/*.sh` whose name ends `_train_sample_eval.sh`: verify the `CONFIG=` path resolves to an existing YAML, and the training entrypoint (`train*.py`) matches the model family for that YAML's `model_name`.
- Path references in `CLAUDE.md`, `AGENTS.md`, `ProMoE-REPA.md`, `analyses/README.md`, `analyses/*.md`: every file path mentioned must exist. Flag broken paths.

**Cross-alignment stability constraints** (from CLAUDE.md "Cross-Alignment Stability Constraints" section)
- For each of the 8 cross-alignment model files (`models_ProMoE_TC_repa_cross_*.py`, `models_ProMoE_TC_repa_MoS_naive_choice_cross_*.py`), confirm:
  1. Every `torch.bmm(z_proj_norm, teacher_norm...)` is followed by `.clamp(-1.0, 1.0)` (or `.clamp(min=-1.0, max=1.0)`).
  2. In `cross_global_block` and `cross_expert_local` variants (both standard and MoS — 4 files total), the attention module is invoked with `x.detach()`, while the projection path uses unwrapped `x`.
- Violations are blocking findings.

**TrainingMonitor hook integrity**
- Class names referenced by `TrainingMonitor` in `utils.py` (`ExpertLocalAttention`, `BlockAlignAttention`, `GlobalPreAttention`, `CoeffPredictor`, `AlignCoefficientPredictor`, `BlockRouter`, `PerBlockRouter`, `AdaLNRouter`) must each exist in at least one `models/*.py`. A monitored class with no defining file = dead hook.

**Code hygiene** (lower priority, only flag if obvious)
- Unused imports, unreachable code, leftover `print(` debug statements outside `if rank == 0:` blocks, dangling `TODO/FIXME` without context.
- Do NOT flag style preferences (line length, blank lines, etc.) — no formatter is configured for this repo.

### 2. Fix
For each finding, make the smallest correction that resolves it. **Do not refactor unrelated code** — bug fixes don't get surrounding cleanup (CLAUDE.md rule).

If a finding is ambiguous (e.g., "is this dead code, or staged for a future variant?"), surface it to the user and pause the iteration — do not guess, do not auto-commit, do not skip. Resume on user input.

### 3. Commit
If anything was fixed in step 2, create one commit per iteration. Use:

```
chore(check): iter N — <one-line summary>

<bulleted list of fixes if more than one>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Rules:
- Stage **only** files actually touched this iteration (`git add <path> <path>`). Never `git add -A` / `.`.
- Never amend. Never `--no-verify`. Never push.
- If no files were touched, do not produce an empty commit — proceed to smoke test.
- Record the new SHA into `commits[]`.

### 4. Smoke test
Run, in order. Any failure resets `consecutive_clean` to 0 and becomes a finding for the next iteration.

a. **Full py_compile sweep** of all source listed in step 1.
b. **Import check for touched modules.** For each `models/models_<X>.py` touched this iteration, run `python -c "from models.models_<X> import *"` to catch import-time regressions. Skip for non-model files.

Do NOT start a real training run, sample run, or anything that occupies a GPU — that is out of scope for /check.

### 5. Bookkeeping
- If `findings.count == 0` AND smoke test passed: `consecutive_clean += 1`.
- Else: `consecutive_clean = 0`.
- Emit a one-line summary: `iter N: K findings, M fixed, smoke=<ok|fail>, consecutive_clean=X/5`.

## Termination

- **Success:** `consecutive_clean == 5`. Print final summary: total iterations, total findings fixed, list of commit SHAs created, and "5/5 consecutive clean — codebase is in a clean state for this scope."
- **Cap hit:** `iter == 20` without reaching 5/5. Print final summary and the outstanding findings from the last iteration.
- **Ambiguity pause:** if step 2 surfaced something to the user, halt with the question and the current state (`iter`, `consecutive_clean`, pending finding). Resume on user input.

## Workflow rules (project-wide, see CLAUDE.md)
- **Clean up smoke-test artifacts immediately.** If this skill ever extends the smoke test beyond `py_compile` + import (e.g., a short training/sampling dry run), delete the temp scripts, generated configs, output dirs (`tb_smoke_*/`, `collapse_smoking_test*/`, `outputs/<model>/<smoke_cfg>/`), and any caches created solely for the smoke test, as soon as that step finishes. Never let debug-only artifacts accumulate.
- **Background processes go to a new tmux window in the current session.** Never use `command &`, `nohup`, or `run_in_background=true` for long-running processes. Use `tmux new-window -t "$(tmux display-message -p '#S')" -n <name> '<command>'`. If `$TMUX` is unset, **abort and ask the user to attach to a tmux session first** — do not silently fall back to backgrounding. Short synchronous commands stay in the foreground.

## What this skill must NOT do
- Do not push to remote, force-push, or amend prior commits.
- Do not delete files or directories without surfacing the rationale to the user first.
- Do not invoke `--no-verify` or bypass any pre-commit hooks.
- Do not run real training / sampling / evaluation — smoke test is compile + import only.
- Do not edit `CLAUDE.md`, `AGENTS.md`, `README.md`, `ProMoE-REPA.md` unless the scan finds factual drift (broken path, missing model variant, wrong file reference). Even then, the doc fix is in scope only if it directly addresses a finding from the same iteration.
- Do not modify `outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`, `collapse_smoking_test*/` — these are runtime artifacts, not source.
- Do not touch `REPA/` (uppercase) — that is a vendored standalone subproject, out of scope per CLAUDE.md.
