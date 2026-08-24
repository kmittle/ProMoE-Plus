---
name: rerun-experiment
description: Re-bucket an existing ProMoE experiment into a fresh _vN output directory before rerunning it after model-code changes. Trace a config, semantic run script, or dated run-time wrapper to the complete experiment file set; rename the config, script, wrappers, and descriptions in lock-step without staging; update exact references; retain the same GPU slots; validate the new output path; and regenerate descriptions. Use for explicit $rerun-experiment requests, requests to 重跑 an experiment after a fix, or output collisions caused by unchanged model_name and config filenames. Never launch the run, allocate a new slot, touch old outputs, or commit changes.
---

# Re-Bucket a ProMoE Experiment

Move an already-defined experiment to a fresh versioned config basename so a run after code changes cannot resume from or overwrite stale checkpoints. Preserve old on-disk output and keep the existing slot and GPU assignment.

## Establish Context

1. Work from the repository root.
2. Read root `AGENTS.md` completely before acting. Read relevant `CLAUDE.md` sections only as supplemental model documentation.
3. Read `scripts/check_output_dir.py`, the target config, semantic wrapper, run-time wrapper, model registry entry, and implementation.
4. Inspect `git status --short` plus relevant staged and unstaged diffs. Preserve unrelated changes; incorporate overlapping user edits into the renamed files.
5. Use `rg` for searches when available and fall back to `grep` or `find`.

## Resolve the Full File Set

Accept a config, semantic script, run-time wrapper, or unambiguous model/variant phrase as the starting point. Resolve:

- `configs/004_ProMoE_<...>.yaml`;
- every semantic run script whose top-level `CONFIG=` points to it, normally a `scripts/<family>/run_<...>_train_sample_eval.sh` or a documented legacy split pair;
- every `scripts/_run_times/<date>/<slot>-<desc>.sh` whose `exec bash` target is one of those semantic scripts;
- each adjacent `<slot>-<desc>-describe.txt`, when present.

Use exact-path searches. Do not treat a substring match against a newer `_vN` file as an old reference. Before editing, report the resolved set when more than one semantic script or run-time wrapper is involved.

Require at least one dated run-time wrapper across the resolved source and any compatible versioned chain before re-bucketing because this workflow must preserve an existing slot. If none exists, explain that same-slot rerun is impossible, read [`../new-experiment/SKILL.md`](../new-experiment/SKILL.md) completely, and, after the user accepts a new slot and supplies or approves the date and supported GPU count, follow its missing-runtime rerun handoff. Do not rename any file before that handoff.

When multiple semantic scripts point to the same config, compare their training entrypoint and orchestration before editing. Only a documented legacy train/infer split pair may move together automatically. For every other multi-script case, whether the pipelines are equivalent aliases or materially different, report all candidates and require the user to identify the intended target or grouping. Do not rename the shared config until every confirmed launchable group has an explicit distinct config basename and output bucket plan.

## Confirm Re-Bucketing Is Appropriate

Use both committed and working-tree evidence:

- inspect `git log --oneline -- <model-files>` for relevant historical changes;
- inspect `git diff` and `git diff --cached` for uncommitted model changes;
- compare the target config with its base or prior version to identify any meaningful YAML change.

Record why old checkpoints are incompatible or why the output name is otherwise unsafe. If neither code nor config semantics changed and the user only wants an independent ablation, read [`../new-experiment/SKILL.md`](../new-experiment/SKILL.md) completely and follow its config-driven ablation path. For a parallel seed or equivalent independent copy, read [`../new-experiment/SKILL.md`](../new-experiment/SKILL.md) completely and follow its independent-parallel-run mode, handing off the resolved config, semantic script, date, supported GPU count, and requested seed or suffix; that mode must preserve the originals and allocate a distinct output bucket and slot. Honor an explicit user request for a fresh bucket even when local Git history cannot prove the remote run state.

## Resume or Advance a Versioned Chain

Before suggesting another suffix, enumerate the source stem's existing `_vN` configs and their semantic scripts, run-time wrappers, descriptions, and effective output directories. Validate present artifacts independently, then compare the code/config change that motivated the request with the change already represented by each chain.

Determine partial versus complete execution state from config, semantic script, and run-time wrapper only. Treat a missing or stale description as a repairable generated sidecar: it never advances the suffix or changes a slot and is regenerated after the target wrapper validates, regardless of whether the output has run.

- If exactly one compatible chain is an unlaunched partial attempt for the current change, continue its remaining lock-step renames, reference updates, validation, and description in place. Do not advance the suffix.
- If the current execution chain is complete and no later model/config change or explicit new-generation request exists, report it idempotently and make no execution-chain change, regardless of whether it has run; regenerate its description when needed.
- Advance to the next suffix only for a later incompatible code/config change, an explicit request for another fresh generation, or an output collision that cannot be attributed to the compatible chain being resumed or reported.
- If multiple chains could represent the request, or launch/change provenance is uncertain, report the evidence and require an explicit target before editing.

## Choose One Fresh Version

Enter this section only when the preflight above establishes that a new generation is required. A compatible chain being resumed or reported keeps its current suffix.

Run:

```bash
python scripts/check_output_dir.py --suggest-version configs/<name>.yaml
```

Use the suggested `_vN` unless the user gave an explicit suffix. The helper strips an existing suffix, so `_v2` advances to `_v3` rather than `_v2_v2`. Preserve any intentional top-level `output_dir`: it is a root prefix, while runtime appends `{model_name}/{config_basename}`, so the `_vN` config basename creates the fresh output leaf. For a batch, choose one suffix for which every config name and effective output path is free so paired names stay aligned.

## Rename in Lock-Step

Rename instead of copy by default so the stale launcher cannot be reused accidentally. Preserve the old output directory; it is not part of the repository rename.

Apply the same suffix to:

- config: `configs/004_..._<variant>.yaml` to `configs/004_..._<variant>_vN.yaml`;
- semantic script: `run_<size>_<variant>_train_sample_eval.sh` to `run_<size>_<variant>_vN_train_sample_eval.sh`;
- run-time wrapper: `<slot>-<desc>.sh` to `<slot>-<desc>_vN.sh`, preserving the slot prefix;
- description: `<slot>-<desc>-describe.txt` to `<slot>-<desc>_vN-describe.txt`, when it exists.

Do not use `git mv`, because it updates the Git index. Use `apply_patch` to add each new text file and delete its old path, preserving all current content and edits. Restore executable mode on renamed shell files when the old files were executable.

Update every internal reference with `apply_patch`:

- semantic script `CONFIG=` to the `_vN.yaml` path;
- semantic script `LOG=` to an aligned `_vN` log name;
- each run-time wrapper `exec bash` target to the `_vN` semantic script.

Keep wrapper slot headers and top-level config `gpu_ids` unchanged. Do not call `scripts/_run_times/new_run.sh` and do not allocate another slot.

## Validate Without Launching

1. Run `bash -n` on every renamed semantic script and run-time wrapper.
2. Verify each wrapper target exists, its `CONFIG=` target exists, and the registry/model family still matches.
3. Derive the new output directory with the same rule as `scripts/check_output_dir.py`, treating any top-level `output_dir` as the root before appending `model_name` and the new config basename. Normalize it and require it to differ from the old output directory, then run `python scripts/check_output_dir.py --config <new-config>`. Require `RESULT: OK`; if occupied, advance the config suffix and repeat the coordinated rename.
4. Derive the expected sample directory as `{output_root}/{model_name}/{new_config_basename}/sample` and verify every renamed semantic wrapper uses it. With a top-level `output_dir`, require inline parsing equivalent to `os.path.join(output_root, model_name, custom_cfg_name, "sample")`, resolving relative results under `REPO_ROOT` while preserving absolute results; repair stale literal `outputs/` logic with `apply_patch` and rerun `bash -n`.
5. Search all live repository text for exact old basenames and paths, excluding Git history and the generated `commands.csv` / legacy `commands.md` files handled below. Require no stale reference in executable files, configs, or maintained documentation. Update maintained docs that enumerate current launchable commands or paths; preserve historical records that intentionally describe the old run, but report those matches. Avoid naive substring checks that also match `_vN` names.
6. Confirm the old output directory was neither modified nor deleted.
7. Re-run `git status --short` and confirm this workflow did not alter the index. Preserve and report any staged state that existed before the workflow.

Do not run training, sampling, evaluation, preprocessing, or any GPU command.

## Regenerate the Description

After each renamed, resumed, or idempotently resolved target wrapper validates, read [`../describe-experiment/SKILL.md`](../describe-experiment/SKILL.md) completely and follow it for that wrapper. This is the Codex composition step corresponding to `$describe-experiment`; create a missing description or overwrite a moved/stale one with text grounded in current code without changing the suffix or slot.

If an affected date directory has `commands.csv` or a legacy `commands.md`, report each one as stale because wrapper paths changed. Do not rewrite or delete either file unless the user also requests `$command-table`. When requested, read [`../command-table/SKILL.md`](../command-table/SKILL.md) completely and follow it for each affected date directory; that workflow regenerates `commands.csv` and only reports legacy `commands.md`.

## Report

For each experiment, report:

- every old-to-new path;
- in-file references changed;
- the reason for re-bucketing;
- unchanged slot and `gpu_ids`;
- new output directory;
- validation results;
- the new launch command, without executing it.

For a resumed or idempotently completed chain, report its current artifacts, validation state, and launch command explicitly, and state that no additional rename or slot allocation occurred.

Use the project's tmux form for the handoff:

```bash
tmux new-window -t "$(tmux display-message -p '#S')" -n <name> \
  'bash scripts/_run_times/<date>/<slot>-<desc>_vN.sh'
```

## Boundaries

- Rename, edit, validate, and describe only. Never launch a run.
- Never allocate a new GPU slot or change `gpu_ids`.
- Never edit or delete old runtime artifacts under `outputs/`, `pretrained_ckpt/`, `training_logs/`, TensorBoard/smoke directories, or uppercase `REPA/`.
- Never stage, commit, push, force-push, amend, reset, or stash.
- Never keep copied old and new run-time wrappers on the same slot. For a rerun after model-code changes where the user explicitly requires both definitions to remain launchable, read [`../new-experiment/SKILL.md`](../new-experiment/SKILL.md) completely and hand off through its preserve-old rerun mode. That workflow must keep the old files untouched and use `scripts/_run_times/new_run.sh` to assign the additional launcher a distinct slot whose `gpu_ids` match the allocator assignment; parallel seeds and equivalent independent copies instead use the independent-parallel-run handoff defined above.
- Every handoff to `$new-experiment` must pass the resolved config, date, supported GPU count, and the semantic script when present; when it is absent, pass the intended family and training entrypoint so the missing-runtime mode can create it. The allocator accepts only 4 or 8 GPUs; for a historical 2-GPU wrapper, require a new compatible date/slot choice rather than reusing its manual packing.
