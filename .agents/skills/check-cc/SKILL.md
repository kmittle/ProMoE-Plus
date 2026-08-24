---
name: check-cc
description: "Cross-check all current staged, unstaged, and untracked PromeMoE++ (ProMoE-Plus) changes with two concurrent tracks: one report-only Claude Code reviewer through cc-yolo-api and three read-only Codex subagents. Use for $check-cc, second-opinion pre-commit review, or uncommitted-change audits. Report findings by default; repair only when the user explicitly asks for fixes. Never stage or commit."
---

# Check ProMoE-Plus Changes with Claude Code

Inspect only the current uncommitted changes in `/mnt/cubefs/caoboyuan/ProMoE-Plus` with two tracks:

- Track A: one external Claude Code review through `cc-yolo-api`.
- Track B: three concurrent Codex collaboration agents using the same C1-C5 criterion.

Merge and verify both tracks, run proportional static checks, and honor the authorization mode below. Never perform Git writes.

## Select the Authorization Mode

- **Report-only mode is the default.** A request to check, review, audit, or obtain a second opinion does not authorize edits. Run one complete dual-track round plus parent checks, verify and report the findings, and stop without changing files.
- **Repair mode requires an explicit request to fix or repair findings.** Fix only verified defects, rerun both tracks and parent checks, and iterate until all tracks are clean.
- A request to commit or push after review does not itself authorize repairs. This skill never stages or commits in either mode.

## Boundaries

- Read root `AGENTS.md` and `CLAUDE.md`; root instructions override this workflow when they conflict.
- Require an attached tmux session: `test -n "${TMUX:-}"`. If absent, abort and ask the user to attach first.
- In repair mode, run at most 10 rounds and fix at most 30 verified findings per round. Report-only mode runs one round.
- Stop when more than 5 combined findings in one round are false positives.
- Preserve unrelated changes and avoid opportunistic refactors, renames, or feature work.
- Exclude and never edit uppercase `REPA/` or runtime/generated content under `outputs/`, `_previous_results/`, `pretrained_ckpt/`, `training_logs/`, TensorBoard, smoke output, checkpoints, logs, caches, generated images, and generated datasets.
- Do not run training, sampling, evaluation, preprocessing, downloads, or GPU jobs.
- Treat warnings as advisory. Blockers/errors alone prevent convergence.
- Remove every temporary Claude Code run directory through the bundled launcher.

## Collect the Scope

Build `files` before every round:

```bash
{
  git -c core.quotepath=false diff --name-only
  git -c core.quotepath=false diff --cached --name-only
  git -c core.quotepath=false ls-files --others --exclude-standard
} | sort -u
```

If empty, report `nothing to check`. Exclude deleted files from direct inspection, but validate references to deleted paths or symbols. Report uppercase `REPA/` changes as out of scope without touching them. Announce the file count and explicitly state that no Git write will occur.

## Verify the Claude Code Entry Point

`cc-yolo-api` is an interactive-Bash shell function, not a PATH executable. Check only its availability:

```bash
bash -ic 'type cc-yolo-api >/dev/null'
```

Do not inspect, print, copy, or persist its body or credentials. Do not substitute another command or override its model, effort, provider, or permission settings.

Because the wrapper has broad permissions:

- Put an explicit report-only prohibition in the brief.
- Use the bundled launcher, which restricts declared tools to the read-only set `Read,Grep,Glob`.
- Before launch, set `umask 077`, create a dedicated baseline directory with `mktemp -d /tmp/promoe_plus_cc_baseline.XXXXXX`, and immediately register an exit trap that removes exactly that returned path. Store `git status`, separate unstaged and staged patch files from `git diff` and `git diff --cached`, and content hashes for in-scope untracked files there. Put the exact absolute patch-file paths in Track A's brief so its read-only `Read` tool can inspect both diffs. Compare the current tree with that baseline after Track A. If anything changed, stop and report affected paths; do not revert user or reviewer changes automatically. Remove the baseline directory after comparison and on every success, failure, timeout, or user-stop path, then clear its trap only after removal succeeds.

## Launch Track A in tmux

From the ProMoE-Plus repository root, invoke `.agents/skills/check-cc/scripts/launch_cc_review.sh` with a unique label such as `check-r1`, the repository root, and the self-contained brief on stdin. Capture its one-line output as `promoe_plus_cc_run_dir`.

The brief must include:

1. The repository root and complete current `files` list.
2. The exact absolute paths of the external unstaged and staged patch files, with an instruction to read both before reviewing current file contents.
3. Instructions to read root `AGENTS.md`, `CLAUDE.md`, and relevant documentation first.
4. The do-not-flag list and prior-round false positives.
5. The complete C1-C5 criterion below.
6. A strict prohibition against modifying repository or Git state; temporary check files may exist only outside the repository and must be removed.
7. The finding format and `ALL_CLEAN` rule.

Immediately launch Track B so both tracks overlap. Poll `<run-directory>/status` at intervals no longer than 30 seconds for at most 20 minutes while keeping the user updated. Require status 0, then read `findings.txt`. On failure or timeout, inspect `cc.stderr.log` and `bash-startup.log`, report the failure, clean the run directory, and stop.

Always clean with:

```bash
.agents/skills/check-cc/scripts/launch_cc_review.sh --cleanup "$promoe_plus_cc_run_dir"
```

Never use a bare background process, `&`, `nohup`, or a detached tmux session.

## Launch Track B

Launch three concurrent `spawn_agent` tasks in round 1. Reuse them with concurrent `followup_task` calls in later rounds. Give all three the exact same read-only task, including `files` and the complete C1-C5 criterion. Limit findings to changed files and defects caused by them, while permitting reads of unchanged callers and contracts. Use long `wait_agent` calls until all reports arrive.

Require every reviewer in both tracks to output at most 500 Chinese characters using:

```text
[severity] path:line - issue - suggested fix
```

Allow `blocker`, `error`, and `warning`. Require final line `ALL_CLEAN` when no blocker/error exists; warnings may precede it.

## Do-Not-Flag Rules

Include these rules in Track A's brief and carry confirmed false positives into subsequent briefs:

- Uppercase `REPA/` is a separate vendored subproject and out of scope.
- Root `model.py` is a non-imported reference; active models live under `models/`.
- Documented absolute promoe/fid_eval interpreters, shared ImageNet paths, and optional local teacher/VAE paths are intentional.
- Existing legacy split REPA scripts may remain; new experiments use template-based all-in-one wrappers.
- `scripts/_run_times/2026_08_05/` is a historical 2-GPU packing exception; the current allocator accepts only 4 or 8 GPUs.
- `resume_checkpoint: True` with no checkpoint starting from step 0 is existing behavior.
- Labeled design plans are reference-only or future work unless stated otherwise.
- `custom_cfg_name` and `outputs/<model_name>/<custom_cfg_name>/` are intentional naming contracts.

## Shared Diff-Scoped C1-C5 Criterion

### C1: Syntax, Imports, and Entrypoints

- Track B and the parent compile each in-scope Python file with an external temporary `PYTHONPYCACHEPREFIX`; compile directly affected entrypoints when a changed module is imported by them. Track A inspects source and import contracts without shell execution.
- Track B and the parent run `bash -n` on each in-scope shell script. Track A reviews shell text with its read-only tools.
- Report syntax failures and missing project symbols as blockers. Verify new third-party imports are in the owning dependency list: root `requirements.txt` for main code and `evaluation/requirements.txt` for evaluation code.

### C2: Model Logic and Training Invariants

- Read every changed Python file and its staged/unstaged diff. Check shapes, boundaries, dtype/device behavior, empty selections, distributed reductions, initialization, cleanup, and rank behavior.
- When models or registries change, verify all four training `model_dict` registries, imported model classes/config keys, YAML `model_name`, and `sample.py`'s merged registry.
- Preserve forward contracts for plain ProMoE, standard REPA, MoS-REPA, multi-align, fused MoS, and cross-alignment variants, including `AddAuxiliaryLoss` flow.
- Preserve cond/uncond separation, prototype routing, TC/EC semantics, top-k choice, expert specialization, and loss-free selection-only balancing bias.
- When touched, preserve current extension contracts: LS-Reg remains parameter-free and step-0-identical with its label/diagonal modes and rank-0 logging; shared-expert DAG-Fuse remains zero-initialized with source-specific config semantics; Expert-Contra parameter ablations retain their mode, block, bias/shared/uncond, and temperature controls.
- For changed cross-alignment files, require normalized cosine clamp after `torch.bmm`; global-block and expert-local predictors consume detached block outputs while projection consumes the original tensor.
- For monitoring changes, preserve `TrainingMonitor` placement, TensorBoard wiring, and monitored-class validity.

### C3: Configuration, Data, and Script Contracts

- For changed YAML, verify registration, naming, defaults, `deep_update`, and runtime consumers for every new key.
- Keep top-level and nested `repa_config` responsibilities separate; verify teacher type/depth and block indexing.
- For data changes, verify JPEG/encoded-latent mode, `LatentFolder`, latent shape, numeric labels, caches, and train-safe path derivation.
- Treat `total_train_batch_size` as global and check world-size divisibility when GPU count changes.
- For changed wrappers, verify `CONFIG`, `LOG`, owning training entrypoint, fixed interpreters, and `scripts/template.sh` behavior.
- For changed schedules, verify `new_run.sh --dry-run`, YAML-owned GPU IDs, current 4/8-GPU allocator semantics, and the documented historical exception.
- Verify output collision handling and `_vN` reruns. Require a same-basename Markdown guide for each new analysis entrypoint and reusable analysis logic in an analysis subpackage.

### C4: Documentation Consistency

- Compare changed Markdown, YAML, CSV, and shell content with actual entrypoints, CLI options, defaults, paths, outputs, data flow, GPU assumptions, and related unchanged documentation.
- Report stale, contradictory, broken, or undocumented user-facing behavior. Do not require reference-only plans to be implemented.

### C5: Cross-Cutting Checks

- Report leaked credentials, unsafe shell expansion, accidental undocumented absolute paths, unexplained TODO/FIXME markers, requirements drift, and broken shared idioms.
- Treat documented paths and the do-not-flag list as intentional.
- Report dead code and unused imports only as warnings unless they break a changed active path.

## Merge, Verify, and Repair

Merge by `path:line` and label sources `cc`, `codex x/3`, or `both`. Treat `both` and `codex >=2/3` as high signal, but independently verify every report. Scrutinize `cc-only` and `codex 1/3` findings and record false-positive reasons.

Run the parent proportional static checks in every round; reviewer checks do not replace them.

In report-only mode, report all verified findings and static-check failures after the first dual-track round, then stop without editing. An `ALL_CLEAN` result means no blocker or error was found; findings do not authorize a repair round.

In repair mode, converge only when Claude Code and all three Codex reports end in `ALL_CLEAN` in the same round and the parent checks pass. Otherwise:

1. Fix verified blockers/errors only with `apply_patch`.
2. Keep fixes in the dirty set when possible. If a changed contract requires touching a formerly clean file, tell the user.
3. Run `python -m py_compile` on dirty Python and affected entrypoints with an external cache; run `bash -n` on dirty shell scripts.
4. Run `scripts/check_output_dir.py` for changed new/rerun configs in the appropriate normal or `--suggest-version` mode.
5. Remove temporary output, verify Track A did not alter the repository, recollect `files`, and repeat.

In repair mode, retry static-check fixes at most five times. In report-only mode, report failures and stop without editing. Never run real training, sampling, evaluation, preprocessing, or GPU work.

## Git Prohibition

Do not run `git add`, `git commit`, `git stash`, `git reset`, `git checkout`, or another Git write operation. Use only read-only status, diff, log, show, and `ls-files` queries.

On repair-mode success, report rounds, files inspected, issues fixed, Claude Code false positives, Codex false positives, files modified by this workflow, and final `git status --short`.

For report-only mode, report the single round, verified findings, false positives, parent-check results, and final `git status --short`, and explicitly confirm that no file or Git state was changed.
