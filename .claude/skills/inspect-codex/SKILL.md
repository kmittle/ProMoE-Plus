---
name: inspect-codex
description: Codex-augmented project-wide carpet check loop on ProMoE-Plus. Each iteration briefs an independent Codex reviewer (xhigh reasoning, launched headless via codex exec --dangerously-bypass-approvals-and-sandbox in a new tmux window that closes itself when the review finishes, kept review-only by instruction + a checksum-revert guard) that runs in parallel with Claude's own scan; Claude then aggregates both finding sets, adjudicates the real problems, fixes them, smoke-tests, and commits — repeating until 5 consecutive iterations find zero real problems. Runs FULLY HANDS-OFF: the Codex window auto-opens and auto-closes, findings are auto-collected and auto-fixed, and the required tmux/codex/git/mktemp commands are pre-allowlisted, so the loop never stops mid-run to ask the user to choose — the only stops are a missing tmux session (must abort) or a genuinely destructive/irreversible fix. Use when the user invokes /inspect-codex or wants a second-opinion review sweep before a milestone.
---

# /inspect-codex — Codex-augmented project-wide carpet check loop

Same goal as `/inspect` (sweep the whole ProMoE-Plus codebase, fix issues, smoke-test, commit, repeat), but every iteration cross-checks the codebase with an **independent Codex reviewer** running **in parallel** with Claude's own scan. Codex only reviews; **Claude is the sole adjudicator and fixer**. Repeat until **5 consecutive iterations** find zero real problems AND a passing smoke test. Hard cap: **20 iterations**.

**Run this fully hands-off.** Once invoked, drive every iteration to completion on your own — auto-launch the self-closing Codex window, auto-collect its findings, auto-adjudicate, auto-fix, auto-smoke-test, auto-commit — and do **not** pause to ask the user to confirm or choose. The required `codex` / `tmux new-window` / `git stash` / `git checkout` / `mktemp` / `cp` / `rm` / `sha256sum` commands are pre-allowlisted in `.claude/settings.local.json`, so no per-command permission prompt should appear. There are exactly **two** legitimate stops: (a) `$TMUX` is unset at the start (must abort — you cannot open a tmux window; see Prerequisites), and (b) a specific fix would be genuinely destructive or irreversible (see step 6). Everything else proceeds automatically.

Iteration shape (note the order — commit comes *after* a passing smoke test):
**brief Codex → (Codex reviews ‖ Claude scans) → aggregate → adjudicate real problems → fix → smoke test → commit**

## Prerequisites (check before iteration 1)
- **Must be inside tmux.** Codex runs as a long job in a new tmux window (project rule). If `$TMUX` is unset, **abort and ask the user to attach to a tmux session first** — never fall back to `&` / `nohup` / `run_in_background`. This is the *only* setup question the skill may ask; once inside tmux it never asks again.
- `codex` CLI on PATH (`which codex`), `~/.codex/auth.json` present, this repo path marked `trust_level = "trusted"` in `~/.codex/config.toml` (it is). `model_reasoning_effort = "xhigh"` is the configured default; we also pass it explicitly.
- **Zero-prompt automation:** the commands this skill runs directly (`codex`, `tmux new-window`, `tmux display-message`, `git stash`, `git checkout`, `mktemp`, `cat`, `cp`, `rm`, `sha256sum`, plus the already-present `git status` / `python3`) are pre-allowlisted in `.claude/settings.local.json`. Do not stop to confirm any of them — just run them. If a genuinely new command surfaces, run it directly rather than pausing to ask.
- **Cost note:** each iteration spends Codex (xhigh) quota — this loop is billed. That is why the 20-iteration hard cap exists.

## State to maintain across iterations
- `iter`: 1-indexed counter
- `consecutive_clean`: resets to 0 on any surviving real finding, any fix, or a smoke-test failure
- `commits[]`: SHAs produced this run
- `findings_history[]`: per-iteration notes (claude-found / codex-found / merged / adjudicated TP/FP), for the final report

## The shared checklist
Both the Codex briefing (step 1) and Claude's own scan (step 3) cover the SAME items — that overlap is exactly what makes them an independent cross-check:
1. **Syntax + imports** — `py_compile` every `.py` under `models/`, `repa/`, `analyses/`, plus top-level `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, `train_with_mae.py`, `sample.py`, `utils.py`, `config.py`, `preprocess/preprocess_vae.py`; and `importlib.import_module('models.<basename>')` for each `models/models_*.py` to catch import-time errors py_compile misses. (Claude runs these; Codex is read-only and reasons statically — see step 1.)
2. **Cross-reference drift** — `model_dict` ↔ `models/` ↔ `config.py` (every referenced `ModelClass` exists in its module; every `config_key` is defined in `config.py`; flag orphans in either direction); each `configs/*.yaml` `model_name` matches a registered key (union of the four training scripts); each `scripts/**/*_train_sample_eval.sh` `CONFIG=` resolves to an existing YAML and its training entrypoint matches the model family; file paths referenced in `CLAUDE.md` / `AGENTS.md` / `ProMoE-REPA.md` / `analyses/*.md` exist.
3. **Cross-alignment stability invariants** (CLAUDE.md "Cross-Alignment Stability Constraints") — in the 8 cross-alignment model files: every `torch.bmm(z_proj_norm, teacher_norm...)` is followed by `.clamp(-1.0, 1.0)`; the `cross_global_block` & `cross_expert_local` variants (standard + MoS, 4 files) invoke the attention module with `x.detach()` while the projection path uses unwrapped `x`. Violations are blockers.
4. **TrainingMonitor hook integrity** — every class name referenced by `TrainingMonitor` in `utils.py` exists in at least one `models/*.py`.
5. **Code hygiene** (lower priority) — unused imports, stray `print(` outside `if rank == 0:`, dangling `TODO/FIXME`. Never flag style/formatting (no formatter is configured).

## One iteration

### 1. Brief Codex — what to check + hard prohibitions
Create a per-run temp dir **outside the repo** so nothing pollutes the working tree: `CODEX_TMP="$(mktemp -d)"`. Write the briefing to `$CODEX_TMP/prompt_${iter}.txt`. It must carry the shared checklist, the prohibitions, and a parseable output format. Template (substitute `<REPO_ROOT>` with the absolute repo path):

```
You are an expert code reviewer for the ProMoE-Plus repo (PyTorch, MoE Diffusion Transformers).
You are running WITHOUT a sandbox (yolo mode), so nothing mechanically stops you from writing — but your ONLY job is to REVIEW and REPORT, and you MUST behave strictly read-only and change NOTHING. Any file you create, modify, rename, or delete is detected by a before/after checksum and reverted, and invalidates your review.

REVIEW SCOPE: the entire repository at <REPO_ROOT>.

CHECK FOR (priority order):
1. Syntax / import-time errors in .py under models/, repa/, analyses/ and the top-level entrypoints
   (train*.py, sample.py, utils.py, config.py, preprocess/preprocess_vae.py). Reason statically and use
   read-only commands (grep, cat, ls, git diff). Do NOT run py_compile or import the modules — that is
   Claude's job, and running it would create .pyc/__pycache__ artifacts; you are review-only.
2. Cross-reference drift:
   - model_dict in train.py / train_with_repa.py / train_with_MoS_repa.py / train_with_mae.py: every
     ModelClass referenced must exist in its module; every config_key must be defined in config.py. Flag
     orphans in either direction.
   - each configs/*.yaml: model_name must match a registered key (union across the four training scripts).
   - each scripts/**/*_train_sample_eval.sh: CONFIG= must resolve to an existing YAML and the training
     entrypoint (train*.py) must match that YAML's model family.
   - file paths mentioned in CLAUDE.md / AGENTS.md / ProMoE-REPA.md / analyses/*.md must exist.
3. Cross-alignment stability invariants (see CLAUDE.md "Cross-Alignment Stability Constraints"): in the 8
   cross-alignment model files, every torch.bmm(z_proj_norm, teacher_norm...) must be followed by
   .clamp(-1.0, 1.0); the cross_global_block and cross_expert_local variants (standard + MoS) must call the
   attention module with x.detach() while the projection path uses unwrapped x. Violations are BLOCKERS.
4. TrainingMonitor hook integrity: every class name referenced by TrainingMonitor in utils.py must exist in
   at least one models/*.py.
5. Code hygiene: unused imports, stray print() outside `if rank == 0:`, dangling TODO/FIXME. Do NOT report
   style/formatting — no formatter is configured.

HARD PROHIBITIONS — you MUST NOT:
- modify, create, rename, or delete ANY file
- run git add/commit/push/stash/reset/checkout or any mutating git command
- run training, sampling, evaluation, or any GPU / long-running job
- touch outputs/, pretrained_ckpt/, training_logs/, tb_smoke_*/, collapse_smoking_test*/, or REPA/ (uppercase)

OUTPUT — your FINAL message must be EXACTLY one of:
(a) the single token:  NO FINDINGS
(b) one or more findings, each formatted as:
### FINDING <n>
- file: <repo-relative path>
- line: <number / range / N/A>
- severity: blocker | warning | nit
- category: syntax | consistency | cross-align | monitor | doc | hygiene
- issue: <one sentence>
- evidence: <why it is a problem; cite the code>
- suggested_fix: <the minimal change>
Output nothing else outside these blocks.
```

### 2. Snapshot the tree, then launch Codex headless (yolo), in parallel (tmux)
Run Codex **non-interactively** with `codex exec` in a new tmux window. Headless is deliberate: `codex exec` **exits the moment the review finishes**, so its tmux window **closes itself** — nothing lingers and the user is never asked to do anything. (`codex exec` has no `--yolo`; its yolo launch is `--dangerously-bypass-approvals-and-sandbox` — skip approvals + no sandbox, required because `bwrap` cannot init the read-only sandbox here. This is the headless equivalent of `cx-yolo` = `codex -c model_reasoning_effort=xhigh --yolo`.)

Because Codex runs **unsandboxed**, the sandbox no longer enforces "review-only" — so first take an **integrity snapshot** (the checksum-revert guard), verified + reverted in step 5. Whole-repo scope, so snapshot via git:

```
git -C "$REPO_ROOT" status --porcelain > "$CODEX_TMP/tree_before_${iter}.txt"
SNAP=$(git -C "$REPO_ROOT" stash create)   # snapshot commit of current WIP; tree untouched (do NOT apply)
echo "${SNAP:-CLEAN}" > "$CODEX_TMP/snap_${iter}.txt"
```

Feed the prompt via stdin (avoids quoting), capture the final report with `-o`, and write a sentinel as the runner's **last** act (so the window closes right after). Keep the tmux command quote-clean with a tiny runner:

```
cat > "$CODEX_TMP/run_${iter}.sh" <<EOF
codex exec -C "$REPO_ROOT" --dangerously-bypass-approvals-and-sandbox -c model_reasoning_effort=xhigh \\
  -o "$CODEX_TMP/findings_${iter}.md" < "$CODEX_TMP/prompt_${iter}.txt" \\
  > "$CODEX_TMP/codex_${iter}.log" 2>&1
echo "EXIT:\$?" > "$CODEX_TMP/codex_${iter}.done"
EOF
test -n "${TMUX:-}" || { echo "abort: not inside tmux — ask the user to attach"; exit 1; }
tmux new-window -t "$(tmux display-message -p '#S')" -n codex-inspect "bash '$CODEX_TMP/run_${iter}.sh'"
```
The window runs only the runner; when `codex exec` returns and the sentinel is written, the runner exits and **tmux closes the window automatically** — no `kill-window`, no decision from the user. Do NOT block on this — go straight to step 3.

### 3. Claude's own scan (in parallel, while Codex runs)
Run the full shared checklist yourself, exactly as `/inspect` does — including actually running `py_compile` and the per-model `import_module` checks (Codex cannot, being read-only). Collect Claude's findings list.

### 4. Wait for Codex (sentinel), then collect its findings — the window has self-closed
Do **not** foreground-`sleep`-poll. After finishing Claude's scan, use the **Monitor** tool's until-condition to block until the runner's sentinel exists: `test -f "$CODEX_TMP/codex_${iter}.done"`. The codex window closes itself when the runner exits, so by the time the sentinel appears the window is already gone — there is **nothing to kill and nothing for the user to do**.

Read the captured report from `$CODEX_TMP/findings_${iter}.md` (the `-o` output); if it is empty/missing, fall back to the newest rollout transcript's `last_agent_message`, then to `$CODEX_TMP/codex_${iter}.log`:

```
python3 - "$CODEX_TMP/findings_${iter}.md" <<'PY'
import sys, json, glob, os
out = open(sys.argv[1], encoding='utf-8', errors='replace').read().strip() if os.path.exists(sys.argv[1]) else ''
if not out:  # fallback: newest rollout transcript's last_agent_message
    rolls = sorted(glob.glob(os.path.expanduser('~/.codex/sessions/**/*.jsonl'), recursive=True), key=os.path.getmtime)
    if rolls:
        lines = [json.loads(l) for l in open(rolls[-1], encoding='utf-8', errors='replace').read().splitlines() if l.strip()]
        tc = [d for d in lines if d.get('type')=='event_msg' and (d.get('payload') or {}).get('type')=='task_complete']
        out = (tc[-1]['payload'].get('last_agent_message') or '').strip() if tc else ''
print(out)
PY
```
Parse the `### FINDING` blocks. If the sentinel shows a non-zero `EXIT:` (auth / quota / crash) or the findings are unreadable, log it in `findings_history`, proceed with Claude-only findings this iteration, and do **not** count the iteration as clean unless Claude's scan was also clean.

### 5. Verify-revert Codex's writes, then aggregate + adjudicate
- **Checksum-revert guard (do this FIRST — Codex ran unsandboxed):** re-run `git status --porcelain` and compare to `$CODEX_TMP/tree_before_${iter}.txt`. If Codex created / modified / deleted anything, revert the tree to the pre-Codex snapshot — restore each tracked file Codex touched from the snapshot commit (`git checkout $SNAP -- <paths>`, with `$SNAP` from `snap_${iter}.txt`; this restores the **pre-Codex WIP content, not HEAD**, so legitimate WIP is preserved) and `rm` only the untracked files Codex created — so the tree exactly matches the snapshot before you proceed. Record any reverted Codex writes in `findings_history`. **Claude remains the sole writer.**
- Merge Claude's findings and Codex's findings; **dedupe** by (file, line, issue).
- For EACH finding (whatever the source), **verify it against the actual code** — do not trust Codex (or yourself) blindly. Read the cited lines. Mark each **true positive** or **false positive** with one line of reasoning.
- **Autonomous Codex follow-up (Claude's call, never the user's):** if a Codex finding is plausible but you cannot confirm or refute it from the code, you MAY *yourself* send ONE focused follow-up to the same Codex session before deciding — resume it headless with `codex exec resume --last --dangerously-bypass-approvals-and-sandbox` and a short question via stdin, capturing the reply with `-o` (re-snapshot + re-verify the tree afterward). This is an internal cross-check you decide on and run automatically — **do not surface it to the user or ask permission.** Keep follow-ups rare and targeted; they cost quota. If still unresolved after one follow-up, treat the finding as a false positive (record it) rather than pausing.
- Only **true positives** proceed to step 6. Record dismissed false positives in `findings_history`.

### 6. Fix (Claude only — Codex never fixes)
For each true-positive finding, make the smallest correction that resolves it. **Do not refactor unrelated code** (CLAUDE.md rule). **Default to fixing autonomously — do not pause to ask the user about ordinary findings.** If a finding is merely *ambiguous* (dead code vs staged-for-a-future-variant, a low-confidence hygiene nit, unclear intent), resolve it **conservatively without asking**: prefer the minimal, reversible change; if even the minimal change is uncertain, skip that finding this iteration and record it in `findings_history` rather than pausing. Ambiguity alone is never a reason to stop.

**Pause only for a genuinely destructive or irreversible fix** — one that deletes non-trivial code that plausibly is a staged future variant, rewrites semantics, or spans many files. Only in that narrow case, surface the specific change to the user and halt (do not guess, do not auto-commit). This is the sole in-loop stop; everything short of it is auto-fixed and the loop continues.

### 7. Smoke test (before any commit)
Run, in order. Any failure resets `consecutive_clean` to 0 and becomes a finding for the next iteration:
a. **Full `py_compile` sweep** of all source listed in checklist item 1.
b. **Import check for touched modules** — for each `models/models_<X>.py` touched this iteration, `python -c "from models.models_<X> import *"`.
Never start real training / sampling / evaluation or anything on a GPU — out of scope for this skill.

### 8. Commit (only if smoke passed AND files were changed)
One commit per iteration, staging **only** the files touched this iteration (`git add <path> ...` — never `git add -A` / `.`):

```
chore(inspect-codex): iter N — <one-line summary>

<bulleted list of fixes if more than one>
codex-found: <c>, claude-found: <m>, dismissed false positives: <f>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
Never amend. Never `--no-verify`. Never push. If nothing was fixed, do not create an empty commit. Record the new SHA in `commits[]`.

### 9. Bookkeeping + cleanup
- `consecutive_clean += 1` iff zero true-positive findings survived adjudication AND smoke passed; otherwise `consecutive_clean = 0`.
- Delete this iteration's Codex temp files (`$CODEX_TMP/*_${iter}.*`). They live outside the repo so they never touch git — clean them anyway, per the project's smoke-artifact rule.
- One-line summary: `iter N: codex=<c>/claude=<m> raw, <tp> real, <fixed> fixed, smoke=<ok|fail>, consecutive_clean=X/5`.

## Termination
- **Success:** `consecutive_clean == 5`. Print: total iterations, total real problems fixed, commit SHAs created, and "5/5 consecutive clean (Codex + Claude agree) — clean for this scope." Remove `$CODEX_TMP`.
- **Cap hit:** `iter == 20` without 5/5. Print the summary + the outstanding findings from the last iteration. Remove `$CODEX_TMP`.
- **Destructive-fix pause (rare):** only if step 6 hit a genuinely destructive/irreversible fix — never for ordinary or merely-ambiguous findings, which are auto-resolved. Halt with the specific change + current state (`iter`, `consecutive_clean`, pending finding). Resume on user input (do NOT remove `$CODEX_TMP` while paused — the run may resume).

## Workflow rules (project-wide, see CLAUDE.md)
- **Run hands-off — no mid-run confirmation prompts.** Drive the whole loop autonomously; never stop to ask the user to confirm launching Codex, approve a command, continue to the next iteration, or adjudicate an ordinary finding. The pre-allowlisted commands run without prompts. The only permitted stops are the missing-tmux abort and the destructive-fix pause.
- **Codex runs only in a new tmux window of the current session** — headless `codex exec`, which exits when its review finishes so the window **closes itself** (no `kill-window`, no user action; step 4). Never `&` / `nohup` / `run_in_background`. If `$TMUX` is unset, abort and ask the user to attach.
- **Clean up smoke-test / Codex artifacts immediately** — the `$CODEX_TMP` dir and all sentinel / log / findings / runner files, as soon as each iteration (and the whole run) finishes.

## What this skill must NOT do
- **Do not pause to ask the user for routine confirmations** — launching Codex, running an allowlisted command, advancing to the next iteration, or adjudicating an ordinary/ambiguous finding all proceed automatically. The only two permitted stops are the missing-tmux abort (Prerequisites) and the destructive-fix pause (step 6).
- Do not let Codex's writes survive — it runs **yolo / unsandboxed** (headless `codex exec`), so it is kept review-only by instruction **plus** the step-2 snapshot / step-5 checksum-revert guard that reverts anything it touches. **Claude is the only writer.**
- Do not push, force-push, or amend. Do not `--no-verify` or bypass pre-commit hooks.
- Do not run real training / sampling / evaluation — smoke test is `py_compile` + import only.
- Do not edit `outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`, `collapse_smoking_test*/`, or the vendored `REPA/` (uppercase).
- Do not write Codex temp files inside the repo — keep them in `$CODEX_TMP` outside the tree.
- Do not edit docs unless the scan finds factual drift (broken path, missing variant, wrong reference) in the same iteration.
