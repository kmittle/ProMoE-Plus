---
name: inspect-cc
description: "Audit the entire PromeMoE++ (ProMoE-Plus) repository with two concurrent tracks: one report-only Claude Code reviewer launched through cc-yolo-api and three read-only Codex subagents. Use for $inspect-cc, whole-project dual-model review, or a Claude Code second opinion. Report findings by default; repair only when explicitly requested, and commit repairs only with explicit commit authorization."
---

# Inspect ProMoE-Plus with Claude Code

Run a whole-project dual-model inspection loop over `/mnt/cubefs/caoboyuan/ProMoE-Plus`:

- Track A: one external Claude Code review through `cc-yolo-api`.
- Track B: three concurrent Codex collaboration agents using the same C1-C5 criterion.

Cross-validate all findings, run static smoke checks, and honor the authorization mode below.

## Select the Authorization Mode

- **Report-only mode is the default.** A whole-project audit, inspection, or second-opinion request does not authorize edits. Run one complete dual-track round plus the parent smoke test, report verified findings, and stop without modifying files or Git state.
- **Repair-only mode requires an explicit request to fix or repair findings.** Apply verified fixes and repeat both tracks until clean, but do not stage or commit.
- **Repair-and-commit mode additionally requires explicit commit authorization.** Commit only the fixes from each completed repair round. A request to inspect or repair alone never grants commit permission.

## Preconditions and Limits

- Require a clean `git status --short`. If dirty, stop without modification and recommend `$check-cc`; continue only with explicit user authorization to include the existing changes.
- Read root `AGENTS.md` and `CLAUDE.md`. Root instructions override this workflow when they conflict.
- Require an attached tmux session: `test -n "${TMUX:-}"`. If absent, abort and ask the user to attach first.
- Exclude uppercase `REPA/`, `.git/`, `outputs/`, `_previous_results/`, `pretrained_ckpt/`, `training_logs/`, TensorBoard, smoke output, checkpoints, logs, caches, generated images, and generated datasets.
- In either repair mode, run at most 10 rounds and fix at most 30 verified findings per round. Report-only mode runs one round.
- Stop when more than 5 combined findings in one round are false positives.
- Do not run training, sampling, evaluation, preprocessing, downloads, or GPU jobs.
- Treat warnings as advisory. Blockers/errors alone prevent convergence.
- Remove every temporary Claude Code run directory through the bundled launcher after consuming it.

Announce both tracks, the three-agent Codex concurrency, and the selected authorization mode before starting. In a repair mode, also state the 10-round cap; mention per-round commits only in repair-and-commit mode.

## Verify `cc-yolo-api`

`cc-yolo-api` is an interactive-Bash shell function, not a PATH executable. Verify only its availability:

```bash
bash -ic 'type cc-yolo-api >/dev/null'
```

Do not print, copy, or persist its definition or credentials. Do not replace it with `claude` or another wrapper. Do not override its model, effort, provider, or permission settings.

The wrapper has broad permissions. Apply all safeguards:

- Put a strong report-only prohibition in every brief.
- Use the bundled launcher, which restricts declared tools to the read-only set `Read,Grep,Glob`.
- Before Track A, create a dedicated baseline directory with `mktemp -d /tmp/promoe_plus_cc_baseline.XXXXXX` and immediately register an exit trap that removes exactly that returned path. Store a read-only working-tree baseline there and compare it afterward. If Claude Code changed the repository, stop and report affected paths; never revert reviewer or user changes automatically. Remove the baseline directory after comparison and on every success, failure, timeout, or user-stop path, then clear its trap only after removal succeeds.

## Launch Track A in tmux

From the ProMoE-Plus repository root, invoke `.agents/skills/inspect-cc/scripts/launch_cc_review.sh` with a round label such as `inspect-r1`, the repository root, and the complete review brief on stdin. Capture its one-line output as `promoe_plus_cc_run_dir`.

The brief must contain:

1. The repository root and whole-project source scope.
2. Instructions to read root `AGENTS.md`, `CLAUDE.md`, and relevant implementation documentation first.
3. The complete do-not-flag list and prior-round false-positive carry-forward.
4. The complete C1-C5 criterion below.
5. A strict prohibition against modifying repository or Git state; temporary check files may exist only outside the repository and must be removed.
6. The exact finding format and `ALL_CLEAN` rule.

Immediately launch Track B after the helper returns so both tracks overlap. Poll `<run-directory>/status` at intervals no longer than 30 seconds for at most 20 minutes while keeping the user updated. Require exit status 0, then read `findings.txt`. On failure or timeout, inspect `cc.stderr.log` and `bash-startup.log`, report the failure, and clean the run directory.

Always clean with:

```bash
.agents/skills/inspect-cc/scripts/launch_cc_review.sh --cleanup "$promoe_plus_cc_run_dir"
```

Never use a bare background process, `&`, `nohup`, or a detached tmux session.

## Launch Track B

Launch three concurrent `spawn_agent` tasks in round 1. Reuse them with concurrent `followup_task` calls in later rounds. Give all three the exact same read-only task, with the complete C1-C5 criterion inlined. Use long `wait_agent` calls until every report is available.

Require every reviewer in both tracks to output at most 500 Chinese characters using:

```text
[severity] path:line - issue - suggested fix
```

Allow `blocker`, `error`, and `warning`. Require final line `ALL_CLEAN` when no blocker/error exists; warnings may precede it.

## Do-Not-Flag Rules

Include these rules in Track A's brief and carry confirmed false positives into subsequent briefs:

- Uppercase `REPA/` is a separate vendored subproject and out of scope.
- Root `model.py` is a non-imported reference; active models live under `models/`.
- Absolute promoe/fid_eval interpreter paths, shared ImageNet paths, and optional local teacher/VAE paths are documented deployment choices.
- Existing legacy split REPA scripts may remain, although new experiment wrappers must follow `scripts/template.sh` as one train+sample+eval script.
- `scripts/_run_times/2026_08_05/` is a historical manually packed 2-GPU exception; the current allocator accepts only 4 or 8 GPUs.
- `resume_checkpoint: True` with no matching checkpoint logging an error and starting at step 0 is existing behavior.
- Labeled design plans and `implementation-plan.md` are reference-only or future work unless the document states otherwise.
- `custom_cfg_name` is intentionally injected from the YAML basename, and output paths follow `outputs/<model_name>/<custom_cfg_name>/`.

## Shared C1-C5 Criterion

### C1: Syntax, Imports, and Entrypoints

- Track B and the parent compile tracked or newly added Python outside uppercase `REPA/` with an external temporary `PYTHONPYCACHEPREFIX`; include root entrypoints and `models/`, `repa/`, `preprocess/`, `evaluation/`, `analyses/`, and `scripts/`. Track A inspects source and import contracts without shell execution.
- Track B and the parent run `bash -n` on tracked or newly added shell scripts outside uppercase `REPA/`. Track A reviews shell text with its read-only tools.
- Report syntax failures and missing project symbols as blockers. Compare explicit third-party imports with the owning dependency list: root `requirements.txt` for main code and `evaluation/requirements.txt` for evaluation code.

### C2: Model Logic and Training Invariants

- Check shapes, boundary conditions, dtype/device handling, empty selections, distributed reductions, initialization, cleanup, and rank-specific behavior.
- Verify the four training-entrypoint `model_dict` registries, imported classes/config keys, YAML `model_name` values, and the union merged by `sample.py`.
- Preserve forward contracts for plain ProMoE, standard REPA, MoS-REPA, multi-align, fused MoS, and cross-alignment paths, including `AddAuxiliaryLoss` routing-contrastive flow.
- Preserve cond/uncond separation, prototype routing, TC/EC layout semantics, top-k selection, expert specialization, and loss-free balancing-bias behavior.
- Preserve current extension contracts: LS-Reg remains parameter-free and step-0-identical with its label/diagonal modes and rank-0 logging; shared-expert DAG-Fuse remains zero-initialized with source-specific config semantics; Expert-Contra parameter ablations retain their mode, block, bias/shared/uncond, and temperature controls.
- Require cosine clamp after normalized `torch.bmm` in all standard/MoS cross-alignment models. Global-block and expert-local predictors consume detached block output while projection consumes the original tensor.
- Preserve `TrainingMonitor` placement after backward/clipping and before `zero_grad()`, TensorBoard wiring, and monitored-class validity.

### C3: Configuration, Data, and Script Contracts

- Compare YAML fields/defaults with `deep_update`, CLI parsing, runtime consumers, model registration, config keys, and wrapper names.
- Keep top-level training-loop and nested model-level `repa_config` responsibilities separate; verify teacher type/depth and block indexing.
- Verify JPEG/encoded-latent selection, `LatentFolder`, latent shape, numeric labels, cache behavior, and the documented train-safe path derivation.
- Treat `total_train_batch_size` as global and check world-size divisibility after GPU changes.
- Require new all-in-one wrappers to follow `scripts/template.sh`, including fixed interpreters, sequential checkpoint loop, repo-root discovery, YAML parsing, and version-sorted evaluation traversal.
- Require runtime allocation via `scripts/_run_times/new_run.sh --dry-run`, with GPU IDs stored in YAML. Verify output collisions and fresh `_vN` rerun names.
- Require same-basename Markdown for new analysis entrypoints and reusable analysis logic in subpackages.

### C4: Documentation Synchronization

- Compare `README.md`, `AGENTS.md`, `CLAUDE.md`, `ProMoE-REPA.md`, design notes, analysis guides, configs, commands, paths, defaults, model names, output layouts, and GPU assumptions with implementation.
- Report stale, renamed, removed, contradictory, broken, or undocumented user-facing behavior.

### C5: Cross-Cutting Checks

- Report leaked credentials, unsafe shell expansion, accidental undocumented absolute paths, unexplained TODO/FIXME markers, requirements drift, and broken shared script idioms.
- Treat documented machine paths and the do-not-flag list as intentional.
- Report dead code and unused imports only as warnings unless they break an active registered path.

## Merge, Verify, and Repair

Merge findings by `path:line`; label sources `cc`, `codex x/3`, or `both`; sort by severity and agreement. Treat `both` and `codex >=2/3` as high signal, but verify every finding yourself. Scrutinize `cc-only` and `codex 1/3` reports against project context. Record rejected findings and append them to later briefs.

Run the parent static smoke test in every round; reviewer checks do not replace it.

In report-only mode, report all verified findings and smoke-test failures after the first dual-track round, then stop without editing. Findings do not authorize a repair round.

In either repair mode, converge only when all four reports end in `ALL_CLEAN` and the parent smoke test passes. Otherwise:

1. Fix verified blockers/errors only, with `apply_patch` and no adjacent refactor.
2. Ask the user only when intended behavior remains genuinely ambiguous after inspecting code, config, and documentation.
3. Run the static smoke test from `$inspect`: compile all tracked Python outside uppercase `REPA/` with an external cache, run `bash -n` on tracked shell scripts, and clean temporary artifacts.
4. For changed new/rerun configs, run `scripts/check_output_dir.py` in the appropriate normal or `--suggest-version` mode.
5. Confirm Track A did not modify the tree and clean its run directory.

In either repair mode, retry static-check fixes at most five times. In report-only mode, report failures and stop without editing. Never run real training, sampling, evaluation, preprocessing, or GPU work.

## Commit Only When Authorized

After static checks pass in repair-and-commit mode:

- Stage only files changed in this round, by explicit path. Never use `git add -A` or `git add .`.
- Review `git diff --cached` before committing.
- Use a concise imperative subject such as `fix(inspect-cc): resolve round N findings`; summarize whether each fix came from `both`, `cc`, or `codex` in the body.
- Never amend, bypass hooks, push, force-push, or add a fabricated co-author trailer.

In repair-only mode, do not stage or commit; start the next dual-track round directly after checks pass. In repair-and-commit mode, start it after committing. On success, report rounds, issues fixed, and false positives by source; include commit summaries only in repair-and-commit mode. In report-only mode, report the single round, verified findings, smoke-test result, and false positives, and explicitly confirm that no file or Git state changed. If a limit or external-review failure is reached, stop with unresolved findings and do not create a partial commit.
