---
name: check
description: Review all current staged, unstaged, and untracked changes in PromeMoE++ (ProMoE-Plus) with three independent Codex subagents using one project-specific criterion. Use for pre-commit hygiene, code/config/script consistency checks, explicit $check requests, or requests to check uncommitted changes. Report findings by default; repair only when the user explicitly asks for fixes. Never stage or commit.
---

# Check Uncommitted ProMoE-Plus Changes

Inspect only the current uncommitted changes in `/mnt/cubefs/caoboyuan/ProMoE-Plus`. Include staged, unstaged, and untracked files of every type. Run proportional static checks and honor the authorization mode below. Never perform a Git write operation.

## Select the Authorization Mode

- **Report-only mode is the default.** A request to check, review, audit, or perform pre-commit hygiene does not authorize file edits. Run one complete three-inspector round plus parent checks, verify the findings, report them, and stop without changing files.
- **Repair mode requires an explicit request to fix or repair the findings.** Make only verified corrections, rerun the checks, and iterate until every inspector reports no blocker or error.
- A request to commit or push after the review does not by itself authorize repairs. Keep Git staging and commits outside this skill in every mode.

## Operating Boundaries

- Read root `AGENTS.md` and `CLAUDE.md` before reviewing. Root instructions override this workflow when they conflict.
- In repair mode, run at most 10 rounds and fix at most 30 verified findings per round. Report-only mode runs one round.
- Stop for user guidance when more than 3 findings in one round are false positives.
- Preserve unrelated working-tree changes. Do not refactor, rename, or add features opportunistically.
- Exclude and never edit uppercase `REPA/`; it is a separate vendored subproject.
- Do not inspect runtime artifacts under `outputs/`, `_previous_results/`, `pretrained_ckpt/`, `training_logs/`, TensorBoard, smoke output, logs, caches, generated images, or generated datasets.
- Do not run training, sampling, evaluation, preprocessing, downloads, or GPU jobs.
- Treat warnings as advisory. Only blockers and errors prevent a clean round.

## Collect the Scope

Run this read-only query from the repository root before every round:

```bash
{
  git -c core.quotepath=false diff --name-only
  git -c core.quotepath=false diff --cached --name-only
  git -c core.quotepath=false ls-files --others --exclude-standard
} | sort -u
```

Use the result as `files`.

- Exclude deleted files from direct inspection, but report references from in-scope files to deleted symbols or paths.
- Report uppercase `REPA/` changes as out of scope without modifying them.
- If `files` is empty, report `nothing to check` and stop.
- Recollect `files` before every round because fixes can add or modify files.
- If more than 50 files are in scope, tell inspectors to prioritize high-risk paths while still checking every changed contract.

Before round 1, show the user the file count and selected authorization mode, and state that this workflow will not stage or commit anything.

## Dispatch Three Independent Inspectors

In round 1, launch three concurrent `spawn_agent` tasks. Reuse those agents in later rounds with concurrent `followup_task` calls. Give all three the exact same read-only task.

The task must include the complete current `files` list and the full C1-C5 criterion below. Require reading root `AGENTS.md`, `CLAUDE.md`, and relevant documentation. Limit review to changed files and defects caused by those changes, while allowing reads of unchanged callers, producers, consumers, registries, configs, and docs needed to validate contracts.

Require no more than 500 Chinese characters in this format:

```text
[severity] path:line - issue - suggested fix
```

Allow `blocker`, `error`, and `warning`. Require the final line `ALL_CLEAN` when no blocker or error remains; warnings may precede it. Wait for all agents with long `wait_agent` calls.

## Diff-Scoped C1-C5 Criterion

### C1: Syntax, Imports, and Entrypoints

- Run `python -m py_compile` for every in-scope Python file, using an external `PYTHONPYCACHEPREFIX`, and run `bash -n` for every in-scope shell script.
- Report syntax failures and missing project symbols as blockers. Verify new third-party imports are declared in the owning dependency list: root `requirements.txt` for main code and `evaluation/requirements.txt` for evaluation code.
- If a changed module is consumed by an unchanged entrypoint, inspect and compile the relevant consumer as needed.
- Do not treat root `model.py` as a production entrypoint; active model implementations live under `models/`.

### C2: Model Logic and Training Invariants

- Read every changed Python file and its staged and unstaged diff. Check shapes, boundaries, dtype/device handling, empty selections, division by zero, distributed reductions, initialization, cleanup, and rank-specific behavior.
- When a model or registry changes, verify `model_dict` consistency across `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, and `train_with_mae.py`, plus the merged registries used by `sample.py`.
- Preserve forward contracts for plain ProMoE, standard REPA, MoS-REPA, multi-align, fused MoS, and cross-alignment models, including routing-contrastive loss flow through `AddAuxiliaryLoss` where designed.
- Preserve cond/uncond separation, prototype routing, TC/EC layout semantics, top-k choice, and expert specialization. Validate every unchanged caller against changed signatures and returns.
- For loss-free routing changes, keep the balancing bias non-trainable and selection-only unless the config explicitly defines another behavior.
- When touched, preserve current extension contracts: LS-Reg remains parameter-free and step-0-identical with its label/diagonal modes and rank-0 logging; shared-expert DAG-Fuse remains zero-initialized with source-specific config semantics; Expert-Contra parameter ablations retain their mode, block, bias/shared/uncond, and temperature controls.
- For changed standard or MoS cross-alignment files, require cosine-similarity clamp after `torch.bmm`; global-block and expert-local predictors must consume detached block outputs while projection uses the original tensor.
- If monitoring changes, preserve `TrainingMonitor` placement after backward/clipping and before `zero_grad()`, TensorBoard wiring, and monitored class-name validity.

### C3: Configuration, Data, and Script Contracts

- For each changed YAML, verify `model_name` registration, model/config/wrapper naming, defaults and `deep_update` behavior, and that every new key has a runtime consumer.
- Keep top-level and nested `repa_config` responsibilities separate. Verify teacher depth/type, zero-indexed `align_blocks`, and human-readable wrapper block ranges.
- For data/preprocessing changes, verify JPEG versus encoded-latent paths, `LatentFolder`, 8-channel latent parameters, numeric label directories, cache behavior, and the documented train-safe path replacement constraint.
- Treat `total_train_batch_size` as global; changing GPU count must preserve valid world-size divisibility.
- For changed experiment wrappers, verify `CONFIG`, `LOG`, model family entrypoint, and exact compliance with `scripts/template.sh`, including fixed promoe/fid_eval Python paths and sequential train-stop-sample/eval-resume behavior.
- For changed run-time schedules, verify allocation through `scripts/_run_times/new_run.sh`, a prior dry run, and YAML-owned `gpu_ids`. Treat `2026_08_05` as the documented historical 2-GPU packing exception; current allocator inputs remain 4 or 8.
- Verify output paths, `custom_cfg_name`, checkpoint/sample lists, and fresh `_vN` naming for reruns after model-code changes.
- A changed/new analysis entrypoint must have a same-basename Markdown guide and keep reusable logic in an analysis subpackage.

### C4: Documentation and Script Consistency

- Compare changed Markdown, YAML, CSV, and shell scripts with actual entrypoints, CLI arguments, defaults, output names, data flow, GPU assumptions, and existing files.
- Check both directions: documentation must describe reality, and changed behavior must not leave related unchanged documentation stale.
- Do not flag a labeled reference-only research plan merely because it is not implemented.

### C5: Cross-Cutting and Do-Not-Flag Rules

- Report leaked credentials, unsafe shell expansion, accidental machine-specific paths, newly introduced unexplained TODO/FIXME markers, requirements drift, and broken shared idioms.
- Treat documented absolute interpreter/shared-data paths and optional local model paths as intentional.
- Existing legacy split REPA scripts are allowed, but new experiments must use one template-based train+sample+eval wrapper.
- Treat `resume_checkpoint: True` with no checkpoint starting at step 0 as existing behavior.
- Treat uppercase `REPA/`, root reference `model.py`, and the `2026_08_05` schedule exception as documented, not defects.
- Report dead code and unused imports only as warnings unless they break a changed registered path.

## Aggregate, Verify, and Repair

Merge reports by `path:line`, preserve severity, and record source agreement as `x/3`. Inspect the relevant code yourself before accepting any finding, especially a `1/3` report. Record why rejected findings are false positives.

Run the parent proportional static checks after aggregation in every round; reviewer checks do not replace them.

In report-only mode, report all verified findings and static-check failures after the first round, then stop without editing. An `ALL_CLEAN` result means the review found no blocker or error; findings do not authorize a repair round.

In repair mode, finish only when all three reports end in `ALL_CLEAN` and the parent checks pass. Otherwise:

1. Fix only verified blockers/errors with `apply_patch`.
2. Keep fixes within the dirty set when possible. If a changed contract requires a formerly clean file, call that out to the user.
3. Ask the user only when intended behavior cannot be established from code, config, documentation, or existing conventions.
4. Run the static smoke test, recollect `files`, and begin the next round.

## Proportional Static Checks

- Compile every Python file in the current dirty set and any directly affected entrypoint, using a temporary external `PYTHONPYCACHEPREFIX` and removing it afterward.
- Run `bash -n` on every shell script in the dirty set.
- For a new or rerun YAML/wrapper, run `python scripts/check_output_dir.py --config <config>` or `--suggest-version` as appropriate.
- Do not import GPU-heavy modules or launch train/sample/eval entrypoints.
- Remove temporary configs, scripts, outputs, caches, and other smoke-only artifacts immediately, including after failure.
- In repair mode, fix and rerun failed static checks up to five times. In report-only mode, report failures and stop without editing.

## Git Prohibition

Do not run `git add`, `git commit`, `git stash`, `git reset`, `git checkout`, or any other Git write operation. Use only read-only status, diff, log, show, and `ls-files` queries.

On a clean repair-mode completion, report:

```text
$check converged at round R. Files inspected: F. Issues fixed: K.
Working tree remains uncommitted and ready for user review.
```

Include the final `git status --short` snapshot and list only the files actually modified by this workflow.

For report-only mode, state that one review round completed, list the verified findings and check results, and explicitly confirm that no file or Git state was changed.
