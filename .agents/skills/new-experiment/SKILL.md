---
name: new-experiment
description: Scaffold and schedule a complete ProMoE experiment without launching it. Create a model implementation and registry entry when needed, or a config-only ablation when existing code supports it; create the numeric-prefixed YAML and template-based train-sample-eval wrapper; validate the chain; preview and allocate a dated 4- or 8-GPU run-time slot with scripts/_run_times/new_run.sh; then generate the companion experiment description. Use for explicit $new-experiment requests or when the user asks to add a ProMoE model variant, ablation, experiment config, or all-in-one experiment wrapper. Never run training, sampling, evaluation, or Git commits.
---

# Create a ProMoE Experiment

Build the full experiment definition and allocate its launch wrapper while preserving repository conventions and existing work. Stop before launching any real run.

## Establish Context

1. Work from the repository root.
2. Read root `AGENTS.md` completely before acting. It defines current model families, config rules, template requirements, GPU allocation, output naming, validation, and protected paths.
3. Read relevant sections of `CLAUDE.md` only as supplemental model-family documentation. Prefer `AGENTS.md`, executable code, and current configs when they differ.
4. Read `scripts/template.sh`, `scripts/_run_times/new_run.sh`, the target training registry, and the closest existing model/config/wrapper before editing.
5. Inspect `git status --short` and relevant diffs. Preserve unrelated user changes and work with overlapping edits rather than replacing them.
6. Use `rg` for searches when available and fall back to `grep` or `find`.

## Resolve Inputs

Infer these values from the request and repository. Ask only when a reasonable choice would materially change the intended experiment:

- variant and family;
- training entrypoint: `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`;
- model size: `B`, `L`, or `XL`;
- exact `MoE_config` and `repa_config` overrides;
- GPU count: default to 4, except XL defaults to 8; accept only 4 or 8 unless the allocator changes in the same task;
- date in `YYYY_MM_DD`: default to today's date.

Classify the request before editing. Choose exactly one path; an explicit handoff takes precedence and must not fall through to another classification:

- **New model variant:** create the model, registry entry, config, semantic wrapper, and run-time wrapper.
- **Config-driven ablation:** keep `model_name` unchanged and create the new config plus wrappers.
- **Independent parallel run:** when `$rerun-experiment` explicitly hands off a parallel seed or equivalent independent copy, keep the old files unchanged. Create or resume a fresh config basename and semantic-script copy with a distinguishing user-provided or agreed suffix, apply only the requested runtime-consumed override, and ensure it has a distinct run-time wrapper and output bucket. If the requested seed or override has no established config consumer, ask instead of inventing a key.
- **Already defined experiment:** preserve existing model/config and create only missing wrappers. Use this only to complete a missing launch definition, never for a requested independent run or seed.
- **Preserve-old rerun handoff:** when `$rerun-experiment` explicitly hands off because the user requires both old and new launchers, keep the old files unchanged. Create or resume fresh `_vN` copies of the resolved config and semantic script, update only their versioned references, and ensure they have a new run-time wrapper with a distinct slot whose `gpu_ids` match the allocator assignment. Never copy the old run-time wrapper.
- **Missing-runtime rerun handoff:** when `$rerun-experiment` finds no dated run-time wrapper and the user accepts that no old slot can be preserved, keep the original definition unchanged, create or resume a fresh `_vN` config plus a semantic wrapper, and ensure they have their first run-time wrapper using the handed-off date and supported GPU count.
- **Previously launched experiment after model-code changes:** except for the explicit handoff classifications above, read [`../rerun-experiment/SKILL.md`](../rerun-experiment/SKILL.md) completely and follow `$rerun-experiment` so the output bucket is versioned without duplicating a slot.

## Resume Compatible Partial Work

Before choosing a suffix or writing any file, inventory the requested definition and any partial prior attempt:

1. Resolve the source config and semantic script when present, then enumerate candidate config, semantic-script, and run-time-wrapper paths independently. Record adjacent descriptions as generated sidecars, not as execution-chain identity. Include expected `_vN` and requested distinguishing suffixes even when only one execution artifact currently exists.
2. Validate each present execution artifact on its own: a config must match the requested model and settings apart from the planned suffix and explicit override; a semantic script must preserve the source or required template orchestration while naming the expected config and aligned log; a run-time wrapper must target the expected semantic script and carry a consistent slot header. A missing neighboring execution artifact makes the chain partial, not incompatible.
3. Classify each compatible execution chain by state. For an unlaunched partial chain, adopt it and create only its missing execution artifacts. For a complete chain, an idempotent repeat of the same generation must validate and report it without changing names or slots, even when its output exists. If the request explicitly starts another generation after a later code/config change, retain the completed chain and choose a new suffix. When launch or generation provenance is uncertain, ask rather than risk reusing or superseding its output. An already-defined maintenance request may fill a missing launcher for an existing output only when the user explicitly intends that behavior. A missing or stale description never changes this state: regenerate it in place after the target wrapper validates, regardless of launch/output state.
4. If multiple compatible execution chains exist, report every artifact and require an explicit target. If an execution-artifact path exists with incompatible contents, never overwrite it; choose a different suffix or ask when intent is ambiguous. A conflicting description is generated state and is overwritten only through `$describe-experiment` after its execution chain is selected.
5. Only when no compatible chain exists, or the request explicitly requires a new generation, may a fresh handoff choose a new name and create copies. Apply the same state classification to a repeated direct invocation whose requested config or semantic script already exists.

## Implement the Model Side

For a new model variant, create missing artifacts and preserve any compatible partial implementation already identified:

1. Create `models/models_ProMoE_TC_<variant>.py`, or `models/models_ProMoE_EC_<variant>.py` for an Expert-Choice family, using `apply_patch` and inheriting from the closest existing implementation.
2. Preserve the family's `forward` signature, return contract, auxiliary-loss flow, dtype/device behavior, and initialization conventions. For cross-alignment models, preserve cosine-similarity clamping after `torch.bmm` and detached inputs to block-wise weight predictors.
3. Register `(ModelClass, config_key)` in exactly the appropriate training entrypoint's `model_dict`. Confirm `sample.py` obtains it through merged registries.
4. Add any shared default to `config.py` only when runtime code needs it. Defaults for new flags must preserve existing behavior.

For each experiment config, create it only when missing; when a compatible config already exists, validate and preserve it instead of recreating it:

1. Create the missing `configs/004_ProMoE_<size>_<variant>.yaml` with `apply_patch`, based on the closest config.
2. Keep `model_name`, filename, config key, and wrapper name aligned.
3. Put model-level `MoE_config` and `repa_config` overrides under `DiT_<size>_config`; keep training-level REPA settings at top level where required.
4. Keep human-readable block ranges 1-indexed in filenames while `align_blocks` remains 0-indexed in YAML.
5. For a config-only ablation, change only the intended overrides and retain the existing `model_name`.
6. Preserve an intentional top-level `output_dir` unless the user requests a different storage root. It is a root prefix, not the complete experiment directory; runtime appends `{model_name}/{config_basename}`, so the fresh config basename supplies the distinct output leaf.

For a preserve-old or missing-runtime rerun handoff with no compatible partial candidate, use `python scripts/check_output_dir.py --suggest-version <old-config>` to choose the fresh `_vN` name and copy the resolved old config with `apply_patch`. A preserve-old handoff also copies its semantic script. A missing-runtime handoff copies the semantic script when one exists; otherwise the semantic-wrapper step below creates it from the current template. Preserve all experiment settings and update copied scripts' `CONFIG=` and `LOG=` references. Do not edit the model, registry, old config, old semantic script, or old run-time wrapper. For preserve-old handoffs, warn that the retained launcher still targets the old output and requires code compatible with its checkpoints.

For an independent-parallel-run handoff with no compatible partial candidate, copy the resolved config and semantic script with `apply_patch`, preserve the originals, and give both copies the same distinguishing suffix. Apply the requested seed or other supported override only after locating its runtime consumer. Update the copied script's `CONFIG=` and `LOG=` references, then let the normal output guard and allocator create a fresh output bucket and distinct slot. Never copy or reuse the old run-time wrapper.

Do not modify root `model.py`; active implementations live under `models/`. Do not edit uppercase `REPA/`.

## Create the Semantic Wrapper

When the resolved definition lacks a semantic wrapper, create `scripts/<family>/run_<size>_<variant>_train_sample_eval.sh` from the current structure of `scripts/template.sh`, using `apply_patch`. Preserve and validate a compatible existing wrapper instead of recreating it.

For a preserve-old handoff, use the versioned semantic-script copy created or resumed above and verify it still conforms to the current template. For a missing-runtime handoff, use its compatible copy when present; otherwise create a versioned semantic wrapper from the current template with the resolved family entrypoint. Do not rebuild an existing script in a way that drops experiment-specific logic.

For an independent-parallel-run handoff, likewise use the suffixed semantic-script copy created above. Verify it against the current template, but do not rebuild or overwrite it in a way that drops experiment-specific logic.

Unless the experiment genuinely requires extra orchestration, change only:

- `CONFIG=`;
- `LOG=`;
- the training entrypoint.

Preserve `set -euo pipefail`, `SCRIPT_DIR` / `REPO_ROOT`, inline Python YAML parsing, the sequential train-stop-sample/eval-resume loop, `find ... -name images | sort -V`, and the fixed interpreters:

- `/mnt/workspace/yujie/.conda/envs/promoe/bin/python` for training and sampling;
- `/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python` for evaluation.

When the config has a top-level `output_dir`, the template's literal `outputs/` sample path is insufficient. Extend its inline YAML parser to emit `os.path.join(output_root, model_name, custom_cfg_name, "sample")`, resolve a relative result under `REPO_ROOT` while preserving an absolute result, and use that value for `SAMPLE_BASE`. Validate it against the same effective directory reported by `scripts/check_output_dir.py`.

Do not create split train and sample/eval scripts and do not use `conda activate`.

## Validate Without Running the Experiment

Run checks proportional to the files changed:

1. Create an external temporary directory for `PYTHONPYCACHEPREFIX`. Use it for every compile and import check, and remove it afterward even when a check fails; do not create or clean repository `__pycache__` directories.
2. Run `python -m py_compile` for each edited Python file.
3. Import each new model module under the same external `PYTHONPYCACHEPREFIX` when the active environment provides its dependencies. If `torch` or another expected training dependency is unavailable, report the skipped import check without treating it as an implementation failure.
4. Run `bash -n` on the semantic wrapper.
5. Verify four-way consistency: registry key, `config.py` config key, YAML `model_name`, and wrapper `CONFIG=` plus training entrypoint.
6. Verify `total_train_batch_size` is divisible by the selected world size and any `sample_gpu_ids` choice is intentional.
7. Starting with the partial-work inventory, normalize the resolved semantic-script and config paths and derive the effective output directory, including any top-level `output_dir`. Include compatible configs and semantic scripts even when no run-time wrapper exists. Then trace every existing run-time wrapper through its `exec bash` target, semantic script `CONFIG=`, and effective output directory. Record definition candidates separately from run-time matches. A config-only run-time match is reusable only after its semantic pipeline is verified equivalent; otherwise it is a conflict. An output-only match is a collision, not a launcher to reuse.
8. Run `python scripts/check_output_dir.py --config <config>` and interpret it according to the selected classification. A new model, config-driven ablation, or explicit fresh handoff with no existing exact semantic/config match requires `RESULT: OK` before allocation. The already-defined-experiment path may intentionally retain its own existing output directory; report that state and do not auto-version it. A different config mapping to the same effective output remains an unresolved collision and blocks allocation.

For a path that requires a fresh output, resolve every conflicting config match, output-only match, or guard collision by choosing the suggested `_vN` config name and updating the semantic wrapper consistently. Recompute all normalized paths, wrapper matches, and the guard result after each rename. An exact semantic match, or a config match whose semantic pipeline is verified equivalent, instead means the workflow is already partially or fully complete: validate and reuse it rather than allocating again. The local guard cannot see the experiment server; version an experiment known to have run remotely even when the local check is clean.

Do not start a training, sampling, evaluation, preprocessing, or GPU smoke run.

## Resolve the Run-Time Slot

Use the resolved definition and run-time match sets collected during validation. Reuse and report exact semantic matches and verified-equivalent config matches; validate their syntax, slot header, `gpu_ids`, config, output, and target instead of allocating a duplicate. A non-equivalent config match or output-only match blocks allocation until the conflict is resolved. If multiple wrappers match, report the duplicate launch definitions and do not select, alter, or describe one without an explicit target.

Allocation is required only when the resolved compatible definition has no run-time wrapper and no unresolved config or output collision remains. This rule applies to every classification: if a fresh handoff's new semantic script or config already has a matching wrapper, treat it as partially completed work and reuse it.

When allocation is required, always preview the allocator before letting it write:

```bash
scripts/_run_times/new_run.sh \
  --script scripts/<family>/run_<...>_train_sample_eval.sh \
  --date <YYYY_MM_DD> \
  --gpus <4|8> \
  --dry-run
```

Inspect and report the preview's date directory, slot, `gpu_ids`, semantic script, config, and wrapper path. If they match the request, rerun the exact command without `--dry-run`.

Let `new_run.sh` patch the top-level YAML `gpu_ids` and create the thin run-time wrapper. Do not reproduce slot math or hand-edit `gpu_ids`. After allocation, run `bash -n` on the generated wrapper and confirm its `exec` target and config exist.

Never allocate a second run-time wrapper for the same semantic script and output bucket merely because the skill was invoked again. A second launcher requires an explicit fresh handoff that first creates a distinct config basename, semantic script, and output bucket.

## Compose the Experiment Description

After each newly created or explicitly resolved/reused target wrapper validates, read [`../describe-experiment/SKILL.md`](../describe-experiment/SKILL.md) completely and follow it for that wrapper. This is the Codex composition step corresponding to `$describe-experiment`; it must create the adjacent `*-describe.txt` without launching anything.

If this workflow created a run-time wrapper and the target date directory already has `commands.csv` or a legacy `commands.md`, report each one as stale because the wrapper set changed. Do not rewrite or delete either file unless the user also requests `$command-table`. When requested, read [`../command-table/SKILL.md`](../command-table/SKILL.md) completely and follow it for the target date directory; that workflow regenerates `commands.csv` and only reports legacy `commands.md`.

## Report

List every created, modified, or explicitly reused model, registry, config, semantic wrapper, run-time wrapper, and description. Include the assigned or reused slot, final `gpu_ids`, derived output directory, validation results, and the launch command, but do not execute it:

```bash
tmux new-window -t "$(tmux display-message -p '#S')" -n <name> \
  'bash scripts/_run_times/<date>/<slot>-<desc>.sh'
```

Remind the user that a long-lived launch requires an attached tmux session.

## Boundaries

- Scaffold, validate, allocate, and describe only. Never launch training, sampling, evaluation, preprocessing, downloads, or GPU jobs.
- Never edit runtime artifacts under `outputs/`, `pretrained_ckpt/`, `training_logs/`, TensorBoard/smoke directories, or uppercase `REPA/`.
- Never stage, commit, push, force-push, or amend.
- Do not use `git add -A` or `git add .` in any later handoff.
- Remove temporary validation artifacts immediately, including after failures.
