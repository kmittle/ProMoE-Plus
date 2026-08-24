---
name: inspect
description: Audit the entire PromeMoE++ (ProMoE-Plus) repository with three independent Codex subagents using one project-specific criterion. Use for whole-project audits, pre-milestone sweeps, or explicit $inspect requests. Report findings by default; repair only when explicitly requested, and commit repairs only with explicit commit authorization.
---

# Inspect the Whole ProMoE-Plus Project

Audit `/mnt/cubefs/caoboyuan/ProMoE-Plus` with three independent, read-only Codex subagents using the same C1-C5 brief. Verify their reports yourself, run the static smoke test, and honor the authorization mode below.

## Select the Authorization Mode

- **Report-only mode is the default.** A whole-project audit, inspection, or pre-milestone sweep does not authorize edits. Run one full review round plus the static smoke test, report verified findings, and stop without modifying files or Git state.
- **Repair-only mode requires an explicit request to fix or repair findings.** Apply verified fixes and repeat until clean, but do not stage or commit.
- **Repair-and-commit mode additionally requires explicit commit authorization.** Commit only the fixes from each completed repair round. A request to inspect or repair alone never grants commit permission.

## Preconditions and Boundaries

- Start with `git status --short`. If the tree is dirty, stop without modifying it and recommend `$check`; continue only if the user explicitly authorizes including the existing changes.
- Read root `AGENTS.md` and `CLAUDE.md` before inspecting. Root instructions override this workflow when they conflict.
- Do not inspect or modify the uppercase `REPA/` directory; it is a separate vendored subproject with its own instructions.
- Exclude `.git/`, `outputs/`, `_previous_results/`, `pretrained_ckpt/`, `training_logs/`, TensorBoard data, smoke-test output, checkpoints, logs, caches, generated images, and generated datasets.
- In either repair mode, run at most 10 rounds and fix at most 30 verified findings per round. Report-only mode runs one round.
- Stop for user guidance when more than 3 findings in one round are false positives.
- Do not run training, sampling, evaluation, preprocessing, downloads, or GPU jobs.
- Treat warnings as advisory. Only blockers and errors prevent convergence.

Before round 1, tell the user that three inspectors will run concurrently and name the selected authorization mode. In a repair mode, also state the 10-round cap; mention commits only when repair-and-commit mode is authorized.

## Dispatch Three Independent Inspectors

In round 1, launch three concurrent `spawn_agent` tasks. Reuse those agents in later rounds with concurrent `followup_task` calls. Give all three the exact same read-only task and inline the complete C1-C5 criterion below; do not merely tell them to read this skill.

Require each inspector to read root `AGENTS.md`, `CLAUDE.md`, and relevant implementation documentation. Require no more than 500 Chinese characters in this format:

```text
[severity] path:line - issue - suggested fix
```

Allow `blocker`, `error`, and `warning`. Require the final line `ALL_CLEAN` when there is no blocker or error; warnings may precede it. Wait for all agents with long `wait_agent` calls.

## Universal C1-C5 Criterion

### C1: Syntax, Imports, and Entrypoints

- Compile every tracked or newly added Python source outside uppercase `REPA/`, including root entrypoints and code under `models/`, `repa/`, `preprocess/`, `evaluation/`, `analyses/`, and `scripts/`. Use an external temporary `PYTHONPYCACHEPREFIX`.
- Run `bash -n` on tracked or newly added shell scripts outside uppercase `REPA/`.
- Report syntax failures and missing project symbols as blockers. Compare explicit third-party imports with the owning dependency list: root `requirements.txt` for main code and `evaluation/requirements.txt` for evaluation code.
- Do not treat root `model.py` as a production entrypoint; the active model implementations live under `models/`.

### C2: Model Logic and Training Invariants

- Check boundaries, shapes, device/dtype handling, empty selections, division by zero, distributed reductions, initialization, resource cleanup, and rank-specific behavior.
- Verify every `model_dict` entry in `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, and `train_with_mae.py` references an imported model class and an existing config key. Verify `sample.py` still merges all four registries.
- Verify every active YAML `model_name` is registered by the correct training entrypoint and that model/config/wrapper names remain aligned.
- Preserve forward contracts: plain DiT and ordinary ProMoE variants return predictions; standard REPA training returns `(pred, zs_proj)`; MoS-REPA, multi-align, fused MoS, and cross-alignment variants return `(pred, alignment_loss)`; routing-contrastive auxiliary loss continues through `AddAuxiliaryLoss` where designed.
- Preserve ProMoE conditional cond/uncond separation, learnable prototype routing, top-k/token/expert-choice semantics, and expert specialization losses. Check TC and EC paths separately rather than assuming they share layout semantics.
- For loss-free routing, keep the per-prototype balancing bias non-trainable and apply it only to selection as configured; do not silently change router gradients or base logits.
- Preserve current extension contracts: LS-Reg remains parameter-free and step-0-identical with its label/diagonal modes and rank-0 logging; shared-expert DAG-Fuse remains zero-initialized with source-specific config semantics; Expert-Contra parameter ablations retain their mode, block, bias/shared/uncond, and temperature controls.
- In all standard and MoS cross-alignment variants, clamp normalized cosine-similarity matrices to `[-1, 1]` after `torch.bmm`. In global-block and expert-local weight predictors, use detached block outputs for the predictor while retaining the original tensor on the projection path.
- If `TrainingMonitor` wiring changes, keep it after backward and gradient clipping but before `zero_grad()`, preserve existing TensorBoard integration, and verify every monitored class name exists.

### C3: Configuration, Data, and Script Contracts

- Compare every changed or referenced YAML field with defaults, `deep_update`, CLI parsing, and all runtime consumers. New defaults must preserve existing behavior unless the experiment explicitly changes it.
- Keep top-level training-loop `repa_config` separate from nested model-level `DiT_*_config.repa_config`; keep teacher type, block depth, zero-indexed `align_blocks`, and human-readable wrapper ranges consistent.
- Verify JPEG/VAE-latent selection, `LatentFolder`, latent parameter shape, label-directory parsing, image/latent caches, and producer/consumer paths. The legacy `str.replace('train', ...)` path requires exactly the documented train-safe layout; encoded-latent mode must not depend on it.
- Treat `total_train_batch_size` as global and verify divisibility by world size when GPU assignments change.
- New train+sample+eval wrappers must follow `scripts/template.sh`: sequential train-stop-sample/eval-resume behavior, repository-root discovery, fixed promoe/fid_eval Python paths, inline YAML parsing, and version-sorted image traversal.
- New scheduling wrappers must come from `scripts/_run_times/new_run.sh` with a dry run first. The allocator accepts 4 or 8 GPUs; treat `scripts/_run_times/2026_08_05/` as the documented historical 2-GPU packing exception.
- Verify `custom_cfg_name`, output directory construction, checkpoint selection, sampling lists, and rerun `_vN` naming remain consistent. Do not reuse an occupied output path after model-code changes.
- A new analysis entrypoint under `analyses/` must have a same-basename Markdown guide and place reusable logic in an analysis subpackage.

### C4: Documentation Synchronization

- Compare `README.md`, `AGENTS.md`, `CLAUDE.md`, `ProMoE-REPA.md`, design notes, analysis guides, configs, commands, paths, defaults, model names, output layouts, and GPU assumptions with implementation.
- Report stale, renamed, removed, contradictory, or undocumented user-facing behavior.
- Validate shell examples and referenced paths. Do not flag research plans merely because they are not implemented when the document labels them as reference-only or future work.

### C5: Cross-Cutting and Do-Not-Flag Rules

- Report leaked credentials, unsafe shell expansion, accidental machine-specific paths, new unexplained TODO/FIXME markers, requirements drift, and broken shared script idioms.
- Treat the documented absolute interpreter paths, shared dataset paths, and optional local checkpoint/VAE paths as intentional.
- Treat existing legacy split REPA scripts as allowed; flag only newly introduced split-purpose experiment wrappers.
- Treat `resume_checkpoint: True` with no checkpoint falling back to step 0 as existing behavior.
- Treat the uppercase `REPA/` tree as out of scope, root `model.py` as a non-imported reference, and the `2026_08_05` 2-GPU schedule as a historical exception.
- Report dead code and unused imports only as warnings unless they break a registered path.

## Aggregate and Repair

Merge findings by `path:line`, retain source hit count `x/3`, and sort by severity. Verify every finding against the code and project rules. Give extra scrutiny to `1/3` findings and record why rejected reports are false positives.

In report-only mode, run the static smoke test, report every verified finding and check failure after the first round, and stop without editing. Findings do not authorize a repair round.

When a verified blocker or error exists in either repair mode:

1. Make the smallest correction that resolves it; do not perform adjacent cleanup or feature work.
2. Use `apply_patch` for manual edits and preserve unrelated user changes.
3. Ask the user only when intended behavior is genuinely ambiguous after reading code, config, and documentation.
4. Run the smoke test before staging anything.

Run the parent static smoke test in every round, including when all inspectors report `ALL_CLEAN`. A round converges only when all three reports are clean and the parent smoke test passes.

## Static Smoke Test

Run from the repository root:

```bash
set -euo pipefail
promoe_compile_cache=$(mktemp -d)
cleanup_promoe_compile_cache() { rm -rf -- "$promoe_compile_cache"; }
trap cleanup_promoe_compile_cache EXIT
git ls-files --cached --others --exclude-standard -z -- '*.py' ':(exclude)REPA/**' \
  | while IFS= read -r -d '' path; do
      if [[ -f "$path" ]]; then
        printf '%s\0' "$path"
      fi
    done \
  | xargs -0 -r env PYTHONPYCACHEPREFIX="$promoe_compile_cache" python -m py_compile
git ls-files --cached --others --exclude-standard -z -- '*.sh' ':(exclude)REPA/**' \
  | while IFS= read -r -d '' path; do
      if [[ -f "$path" ]]; then
        printf '%s\0' "$path"
      fi
    done \
  | xargs -0 -r -n1 bash -n
cleanup_promoe_compile_cache
trap - EXIT
```

In either repair mode, fix and rerun failures up to five times. In report-only mode, report failures and stop without editing. If a new or rerun config was changed, also run `python scripts/check_output_dir.py --config <config>` or `--suggest-version` as appropriate. Do not import GPU-heavy modules or launch runtime entrypoints. Remove all temporary artifacts even when a check fails.

## Commit Only When Authorized

After all static checks pass in repair-and-commit mode:

- Confirm that the tree was initially clean or that the user explicitly authorized the pre-existing changes.
- Stage only files changed in this round, by explicit path. Never use `git add -A` or `git add .`.
- Review `git diff --cached` before committing.
- Use a concise imperative subject such as `fix(inspect): resolve round N findings` and a short body naming the corrected areas.
- Do not amend, bypass hooks, push, force-push, or add a fabricated co-author trailer.

In repair-only mode, do not stage or commit; begin the next round directly after checks pass. In repair-and-commit mode, begin it after committing. Either repair mode converges only when all three reports end in `ALL_CLEAN` in the same round and the parent smoke test passes.

On repair-and-commit success, report:

```text
$inspect converged at round R. Total commits: M. Issues fixed: K.
```

List the commits created by this invocation. In repair-only mode, instead report the converged round, issue count, modified files, and confirmation that they remain uncommitted. In report-only mode, report the one-round findings and smoke-test result and confirm that no file or Git state changed. If a round, retry, finding, or false-positive limit is reached, stop and report the unresolved state without creating a partial commit.
