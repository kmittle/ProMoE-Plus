---
name: rerun-experiment
description: Re-bucket an existing ProMoE experiment to a fresh _vN output directory before re-running it after a model-code change (e.g. a crash fix, init/normalization/architecture edit). Output dirs are outputs/{model_name}/{custom_cfg_name}; since model_name and the config filename usually don't change after a code fix, a naive re-run silently collides with — or resumes from — the previous (crashed/stale) run's checkpoints. This skill renames the experiment's config + semantic run script + run-time wrapper in lock-step to a _vN name (so the new run gets a clean bucket while the old run's data is preserved under the old name), updates every in-file reference, and validates the chain. Use when the user asks to re-run / 重跑 an experiment after fixing its code, to give an experiment a fresh output dir, or to de-conflict a "code changed but output path didn't" situation. Does NOT run training/sampling/eval, allocate a new GPU slot, commit, push, or amend.
---

# /rerun-experiment — Re-bucket an experiment to a fresh `_vN` output dir

When an experiment's **model code** changes (crash fix, init correction, normalization, architecture
edit) but its `model_name` and config filename stay the same, its output dir
`outputs/{model_name}/{custom_cfg_name}` (train.py:434, `custom_cfg_name` = config basename at
train.py:723) is unchanged. Re-running then **collides with or resumes from the previous run's
checkpoints** — mixing old (buggy) weights with new code, or clobbering crashed-run artifacts. This
is the bug that motivated the skill.

The fix is mechanical and always the same: bump the experiment's config + run script + wrapper to a
fresh `_vN` name (default `_v2`, then `_v3`, …) in lock-step, so the new run writes to a clean
`..._vN/` bucket while the old run's data stays put under the old name. This skill standardizes that
so it is never done half-way (the classic failure: renaming the config but leaving the script's
`CONFIG=` pointing at the old name, or vice-versa).

This skill **renames + validates only**. It never launches training/sampling/eval, never allocates a
new GPU slot (the experiment keeps its existing slot/`gpu_ids`), and never commits.

## Step 0 — Identify the experiment's full file set
For each target experiment, resolve the complete chain (any one entry point reaches the rest):
- **config** — `configs/004_ProMoE_<...>.yaml`
- **semantic run script** — the `scripts/<family>/run_<...>_train_sample_eval.sh` whose `CONFIG=`
  points at that config (reverse-lookup with `grep -rl 'CONFIG="configs/<name>.yaml"' scripts`,
  excluding `scripts/_run_times/`).
- **run-time wrapper(s)** — the `scripts/_run_times/<date>/<slot>-<desc>.sh` whose `exec bash`
  targets that semantic script (`grep -rl '<run script basename>' scripts/_run_times`).
A target may be given as a config, a run script, a wrapper path, or a `model_name`+variant phrase —
trace outward to the full {config, script, wrapper(s)} set before touching anything. Handle multiple
experiments in one invocation (e.g. a whole proto_t batch).

## Step 1 — Confirm a re-bucket is warranted (and document why)
Re-bucketing is for **code changes that invalidate prior checkpoints**. Confirm and record the cause:
- `git log --oneline -- models/<the model file(s) this config uses>` — show what changed since the
  experiment was first added (e.g. init fix + LayerNorm). Note it for the report.
- Sanity-check the **config content is unchanged** by the fix (the change should be in model code,
  not the YAML). If the YAML itself changed meaningfully, say so — a rename still works, but call it out.
If the code did **not** change and the user just wants a parallel run, this is really a new ablation —
redirect to `/new-experiment` instead.

## Step 2 — Pick the new version
Default: `python scripts/check_output_dir.py --suggest-version configs/<name>.yaml` → next free
`_vN` (it strips any existing `_vN` so `_v2` → `_v3`, never `_v2_v2`). Honor an explicit suffix the
user gives. Keep the **same suffix across the whole batch** so paired experiments stay aligned.

## Step 3 — Rename in lock-step (config → script → wrapper)
Default is **rename** (`git mv`), not copy: it removes the old colliding name from HEAD so the
footgun can't be re-launched, while the old run's on-disk data (untracked) is preserved under the old
name. (Copy only if the user explicitly wants the pre-fix variant to stay launchable — then warn that
the old config still points at the colliding dir.) Insert `_vN` at the variant position, keeping the
fixed prefixes/suffixes:
- `configs/004_..._<variant>.yaml` → `..._<variant>_vN.yaml`
- `scripts/<fam>/run_<size>_<variant>_train_sample_eval.sh` → `..._<variant>_vN_train_sample_eval.sh`
- `scripts/_run_times/<date>/<slot>-<desc>.sh` → `<slot>-<desc>_vN.sh` (**keep the `<slot>` prefix** —
  same GPU assignment; only the desc gains `_vN`)
- `scripts/_run_times/<date>/<slot>-<desc>-describe.txt` (the companion experiment description, if it
  exists) → `<slot>-<desc>_vN-describe.txt` — `git mv` it too so it doesn't dangle under the old name
  (its content is regenerated in Step 5)

Then update every **in-file reference** (this is the step that's easy to half-do):
- semantic script: `CONFIG="configs/..._vN.yaml"` **and** `LOG="log_..._vN_..._train_sample_eval.log"`
- wrapper: the `exec bash "${REPO_ROOT}/scripts/<fam>/..."` path → the `_vN` script
- The wrapper's `gpu_ids` and the config's `gpu_ids` are **unchanged** — same slot, same GPUs. Do
  **not** call `new_run.sh` (no new slot is allocated; this is the same experiment re-bucketed).

## Step 4 — Validate (no real runs)
- `bash -n` every renamed semantic script and wrapper.
- **Chain integrity:** wrapper `exec` target exists → its `CONFIG` resolves → derive its output dir;
  confirm each hop lands on a `_vN` file that exists.
- `python scripts/check_output_dir.py --config configs/..._vN.yaml` → expect `RESULT: OK` (fresh
  bucket). If it still reports a collision, the chosen `_vN` was taken — bump again.
- **No stale refs:** confirm nothing outside git history still points at the renamed-away names.
  A bare `grep '<stem>'` is **wrong** — it also matches the new names (`proto_t_direct` ⊂
  `proto_t_direct_v2`). Grep the **exact old basenames** (e.g.
  `grep -rn 'run_B_<variant>_train_sample_eval.sh' scripts configs`), or exclude the new suffix:
  `grep -rnE '<old-stem>(\.yaml|\.sh|")' scripts configs | grep -v '_v[0-9]'` → expect no hits.
- Do NOT start training/sampling/eval, and do NOT touch `outputs/`.

## Step 5 — Regenerate the experiment description
After the renames validate, invoke **`/describe-experiment`** on the `_vN` wrapper to (re)write
`<slot>-<desc>_vN-describe.txt`. The change list is about the *variant*, so it matches the old name —
but regenerate it freshly (the model code changed, which is why the experiment was re-bucketed). If a
`<old-stem>-describe.txt` was `git mv`'d in Step 3, this overwrites the moved file's contents; if none
existed, it creates the description for the `_vN` stem. Read-only tracing + one `.txt` write.

## Step 6 — Report
List, per experiment, the `git mv` renames (config / script / wrapper / `*-describe.txt`) and the in-file edits,
the **unchanged** slot + `gpu_ids`, and the new output dir `outputs/{model_name}/{cfg}_vN/`. Give the
launch command but **do not run it**:
```
tmux new-window -t "$(tmux display-message -p '#S')" -n <name> 'bash scripts/_run_times/<date>/<slot>-<desc>_vN.sh'
```
If the date dir has a `commands.md`, note it is now stale — offer to regenerate it with `/command-table`.

## Workflow rules (project-wide, see CLAUDE.md)
- **Background processes go to a new tmux window in the current session.** Never `command &`,
  `nohup`, or `run_in_background=true`. This skill does not launch runs; any hand-off must follow this.
- **Stage explicit paths only** if the user later asks to commit — never `git add -A` / `git add .`.

## What this skill must NOT do
- **No real training / sampling / evaluation runs** — rename + validate only.
- **No new GPU-slot allocation** — the experiment keeps its existing slot; do not call `new_run.sh`.
- **No git commits, push, force-push, or amend.** Leave the renames/edits dirty for the user to commit.
- No edits to runtime artifact dirs (`outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`,
  `collapse_smoking_test*/`) or the vendored `REPA/` (uppercase) subproject.
- Do not auto-fire on first-time experiment creation — that is `/new-experiment`.
