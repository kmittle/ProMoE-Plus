---
name: debug
description: Like /check, but the carpet scan is scoped to the current uncommitted diff only (modified + staged + untracked relative to HEAD). Iterate (scan → fix → smoke test) until 5 consecutive iterations on the dirty working tree find zero issues. Use when the user invokes /debug or wants to validate WIP before committing.
---

# /debug — Iterative carpet check on uncommitted changes

Same loop shape as `/check`, but **scoped to uncommitted changes** and **does not commit during the loop**. The deliverable is a clean working tree that the user can then commit themselves.

Hard cap: **20 iterations total** — if hit, stop and report what remains.

## State to maintain across iterations
- `iter`: 1-indexed iteration counter
- `consecutive_clean`: resets to 0 on any finding or smoke-test failure
- `findings_history[]`: short notes per iteration, for the final report

## Defining "the dirty set" each iteration
Re-compute at the start of every iteration (fixes change the diff):

```
git status --porcelain
```

The dirty set = files reported as `M`, `A`, `??`, `R`, `C`, or any combination — i.e. anything other than clean. Track both the **paths** and, for `M`/`A`/staged, the **hunks** via `git diff HEAD -- <path>` so the scan can focus on the changed lines, not the whole file.

If the dirty set is empty at the start of an iteration: stop immediately with "nothing to debug — working tree is clean."

## One iteration

### 1. Carpet scan (scoped to dirty set)
Run in parallel where possible. Findings only count if they involve dirty files or are caused by them.

**Syntax + import graph** (dirty files only)
- `python -m py_compile <each .py in dirty set>`.
- For every dirty `models/models_*.py`, run `python -c "import importlib; importlib.import_module('models.<basename>')"` to catch import-time errors py_compile misses.
- If a dirty file is imported by another (still-clean) file in the project, run py_compile on those importers too — they may break even though they look untouched.

**Cross-reference drift caused by the diff** (the high-signal checks)
- If `model_dict` in any of `train.py` / `train_with_repa.py` / `train_with_MoS_repa.py` / `train_with_mae.py` is in the dirty set: verify every `ModelClass` referenced exists in the named module and every `config_key` is defined in `config.py`.
- If a `models/models_*.py` file is dirty: verify it is still registered in at least one `model_dict` (or, if intentionally not, surface as ambiguous to the user).
- If a YAML in `configs/` is dirty: verify `model_name` matches a registered key (union across the four training scripts), and any `MoE_config` / `repa_config` keys it sets are actually read by the matching model file.
- If a script in `scripts/**/*.sh` is dirty: verify the `CONFIG=` path resolves to an existing YAML, the training entrypoint matches that YAML's model family, and the script follows `scripts/template.sh` pattern (see CLAUDE.md "Shell Script Convention").
- If a doc (`CLAUDE.md`, `AGENTS.md`, `ProMoE-REPA.md`, `analyses/*.md`) is dirty: every file path mentioned in the new/changed lines must exist.

**Cross-alignment stability constraints** (CLAUDE.md "Cross-Alignment Stability Constraints")
- If any of the 8 cross-alignment model files (`models_ProMoE_TC_repa_cross_*.py`, `models_ProMoE_TC_repa_MoS_naive_choice_cross_*.py`) is in the dirty set, verify on the post-diff content:
  1. Every `torch.bmm(z_proj_norm, teacher_norm...)` is followed by `.clamp(-1.0, 1.0)` (or equivalent).
  2. In `cross_global_block` and `cross_expert_local` variants (both standard and MoS), the attention module is invoked with `x.detach()`; the projection path uses unwrapped `x`.
- Violations are blocking findings even if pre-existing — the diff is the reason this file is being looked at.

**TrainingMonitor hook integrity** (only if `utils.py` is dirty)
- Class names referenced by `TrainingMonitor` must each exist in at least one `models/*.py`.

**Code hygiene** (lower priority, only on changed hunks)
- Unused imports introduced by the diff, leftover `print(` debug statements outside `if rank == 0:` blocks, dangling `TODO/FIXME` without context, contradictory comments vs new code.
- Do NOT flag pre-existing code that the diff didn't touch.
- Do NOT flag style preferences — no formatter is configured.

### 2. Fix
For each finding, make the smallest correction that resolves it. Confine fixes to the dirty set when possible — if a fix requires touching a previously clean file (e.g., updating an importer), that's allowed, but call it out in the iteration summary.

If a finding is ambiguous (e.g., "is this dead import staged for an upcoming change?"), surface it to the user and pause the iteration — do not guess, do not skip. Resume on user input.

### 3. Smoke test
Run, in order. Any failure resets `consecutive_clean` to 0 and becomes a finding for the next iteration.

a. `python -m py_compile` on every `.py` currently in the dirty set.
b. For each dirty `models/models_<X>.py`: `python -c "from models.models_<X> import *"`.
c. If `train*.py`, `sample.py`, `utils.py`, `config.py`, or `repa/*.py` is dirty: also py_compile that file plus its direct importers among the other train/sample entrypoints.

Do NOT start real training, sampling, or evaluation — out of scope.

### 4. Bookkeeping (NO commit)
- If `findings.count == 0` AND smoke test passed: `consecutive_clean += 1`.
- Else: `consecutive_clean = 0`.
- Emit a one-line summary: `iter N: dirty=<count>, K findings, M fixed, smoke=<ok|fail>, consecutive_clean=X/5`.
- **Do not run `git add`, `git commit`, or `git stash`.** The working tree stays dirty by design — the goal is to hand a clean WIP back to the user.

## Termination

- **Success:** `consecutive_clean == 5`. Print final summary: total iterations, total findings fixed, current dirty set (`git status --short`), and one line suggesting the user commit when ready. **Do not commit on the user's behalf** — leave that to them.
- **Cap hit:** `iter == 20` without reaching 5/5. Print final summary, outstanding findings, and the current dirty set.
- **Clean tree at start of an iteration:** stop with "nothing to debug — working tree is clean. Did you mean /check?"
- **Ambiguity pause:** halt with the question and the current state (`iter`, `consecutive_clean`, dirty set, pending finding). Resume on user input.

## Workflow rules (project-wide, see CLAUDE.md)
- **Clean up smoke-test artifacts immediately.** If this skill ever extends the smoke test beyond `py_compile` + import (e.g., a short training/sampling dry run), delete the temp scripts, generated configs, output dirs (`tb_smoke_*/`, `collapse_smoking_test*/`, `outputs/<model>/<smoke_cfg>/`), and any caches created solely for the smoke test, as soon as that step finishes. Never let debug-only artifacts accumulate.
- **Background processes go to a new tmux window in the current session.** Never use `command &`, `nohup`, or `run_in_background=true` for long-running processes. Use `tmux new-window -t "$(tmux display-message -p '#S')" -n <name> '<command>'`. If `$TMUX` is unset, **abort and ask the user to attach to a tmux session first** — do not silently fall back to backgrounding. Short synchronous commands stay in the foreground.

## What this skill must NOT do
- **No git commits.** Not per-iteration, not at the end, not even "just one for the fixes." Committing is the user's call.
- No `git stash`, `git reset`, `git checkout --`, or any operation that drops/hides the user's WIP.
- No push, force-push, or amend (irrelevant here, but stated for symmetry with /check).
- No real training / sampling / evaluation runs.
- No edits to `outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`, `collapse_smoking_test*/`, or `REPA/` (uppercase vendored subproject).
- No edits to clean (non-dirty) files unless required to fix a finding caused by the dirty set — and in that case, mention it explicitly in the iteration summary so the user knows their commit-to-be will grow.
