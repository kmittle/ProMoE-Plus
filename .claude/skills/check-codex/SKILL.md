---
name: check-codex
description: Codex-augmented carpet check scoped to the uncommitted diff (modified + staged + untracked vs HEAD). Each iteration briefs an independent Codex reviewer (xhigh reasoning, launched as the interactive cx-yolo TUI (codex --yolo) in a new tmux window that is closed once the review finishes, kept review-only by instruction + a checksum-revert guard) that runs in parallel with Claude's own diff-scoped scan; Claude aggregates both finding sets, adjudicates the real problems, and fixes them, then smoke-tests — but never commits (leaves validated WIP dirty for the user). Repeats until 5 consecutive iterations find zero real problems. Use when the user invokes /check-codex or wants a Codex second opinion on WIP before committing.
---

# /check-codex — Codex-augmented check on uncommitted changes

Same loop as `/inspect-codex`, but **scoped to the uncommitted diff** and — like `/check` — it **never commits**. The deliverable is a clean, Codex-and-Claude-vetted working tree that the user commits themselves. Codex only reviews; **Claude is the sole adjudicator and fixer**. Repeat until **5 consecutive iterations** find zero real problems AND a passing smoke test. Hard cap: **20 iterations**.

Iteration shape: **brief Codex (diff scope) → (Codex reviews ‖ Claude scans) → aggregate → adjudicate → fix → smoke test** — no commit, ever.

## Prerequisites
Same as `/inspect-codex`: **must be inside tmux** (else abort and ask the user to attach — never background with `&`/`nohup`/`run_in_background`); `codex` CLI authed; repo path `trusted`; `model_reasoning_effort=xhigh`. **Each iteration is billed** (Codex xhigh) — hence the 20-iteration cap.

## Defining "the dirty set" each iteration
Re-compute at the start of every iteration (fixes change the diff):
```
git status --porcelain
```
Dirty set = anything reported `M` / `A` / `??` / `R` / `C`. Track both the **paths** and, for `M`/`A`/staged, the **hunks** via `git diff HEAD -- <path>`. **If the dirty set is empty at the start of an iteration: stop immediately with "nothing to check — working tree is clean. Did you mean /inspect-codex?"**

## State to maintain
`iter`, `consecutive_clean` (0..5), `findings_history[]`. **No `commits[]`** — this skill does not commit.

## One iteration

### 1. Brief Codex — diff scope + hard prohibitions
`CODEX_TMP="$(mktemp -d)"` — **outside the repo**: a findings file inside the tree would itself show up in `git status` and contaminate the very dirty set this skill is scoped to. Write `$CODEX_TMP/prompt_${iter}.txt` using the `/inspect-codex` step-1 template with these changes:
- **REVIEW SCOPE** becomes: *"ONLY the current uncommitted changes. First run `git status --porcelain` and `git diff HEAD` (read-only, allowed) to determine the dirty set (modified + staged + untracked). Report findings only for those files — or for still-clean files that the dirty changes break (e.g. an importer of a changed module). Do NOT report pre-existing issues in code the diff did not touch."*
- The cross-reference / cross-alignment / TrainingMonitor checks apply **only when a relevant file is in the dirty set** (check cross-alignment invariants only if a cross-alignment model file is dirty; `model_dict` consistency only if a `model_dict` or a `models_*.py` is dirty; etc.).
- Keep the **same HARD PROHIBITIONS** and the **same `### FINDING` OUTPUT FORMAT** as `/inspect-codex`.

### 2. Snapshot the dirty set, then launch the interactive cx-yolo TUI (yolo), in parallel (tmux)
Same mechanics as `/inspect-codex` step 2 — launch the **interactive `cx-yolo` TUI** (`codex -c model_reasoning_effort=xhigh --yolo "<prompt>"`; top-level `codex [PROMPT]` forwards to the interactive CLI) in a new tmux window named `codex-check`, seeded with the diff-scoped briefing. The prompt itself constrains Codex to the uncommitted diff (interactive mode has no `--uncommitted` flag). Because Codex runs **unsandboxed**, first back up the **dirty set** so any write it makes can be reverted in step 5; also record existing rollout transcripts (step 4 picks out the new one) and capture the window id (step 4 closes it). Diff scope makes the backup precise and cheap — copy each dirty file's content:
```
REPO_ROOT="$(git rev-parse --show-toplevel)"
git status --porcelain > "$CODEX_TMP/tree_before_${iter}.txt"
mkdir -p "$CODEX_TMP/snap_${iter}"
git status --porcelain | sed -E 's/^...//' | while IFS= read -r f; do
  [ -e "$f" ] && { mkdir -p "$CODEX_TMP/snap_${iter}/$(dirname "$f")"; cp -a "$f" "$CODEX_TMP/snap_${iter}/$f"; }
done
find ~/.codex/sessions -type f -name '*.jsonl' | sort > "$CODEX_TMP/rollouts_before_${iter}.txt"
cat > "$CODEX_TMP/run_${iter}.sh" <<EOF
cd "$REPO_ROOT" && codex -c model_reasoning_effort=xhigh --yolo "\$(cat "$CODEX_TMP/prompt_${iter}.txt")"
EOF
test -n "${TMUX:-}" || { echo "abort: not inside tmux — ask the user to attach"; exit 1; }
WIN=$(tmux new-window -P -F '#{window_id}' -t "$(tmux display-message -p '#S')" -n codex-check "bash '$CODEX_TMP/run_${iter}.sh'")
echo "$WIN" > "$CODEX_TMP/winid_${iter}.txt"
```
Do NOT block — go to step 3. `--yolo` skips approval prompts, so the review runs unattended.

### 3. Claude's own scan (diff-scoped, in parallel)
While Codex runs, run the diff-scoped scan exactly as `/check` does: `py_compile` each dirty `.py`; `import_module` each dirty `models/models_*.py`; py_compile the still-clean importers of any dirty module; the cross-reference / cross-alignment / monitor / hygiene checks **only as triggered by the dirty set**. Findings count only if they involve dirty files or are caused by them.

### 4. Wait for the review to finish (rollout `task_complete`), read findings, close the window
Same as `/inspect-codex` step 4: Monitor (no `sleep`-poll) until the new rollout `*.jsonl` (not in `rollouts_before_${iter}.txt`) contains an `event_msg` with `payload.type == "task_complete"`, extract that event's `last_agent_message` into `$CODEX_TMP/findings_${iter}.md` (fall back to the last non-empty `agent_message`), then close the window: `tmux kill-window -t "$(cat "$CODEX_TMP/winid_${iter}.txt")"`. On Codex error (no new rollout / window died / no `task_complete`), kill the window anyway, proceed Claude-only, and don't count clean unless Claude was clean too.

### 5. Verify-revert Codex's writes, then aggregate + adjudicate
First the **checksum-revert guard** (Codex ran unsandboxed): re-run `git status --porcelain` and `sha256sum` the dirty set; if Codex created / modified / deleted anything, restore each changed file from its pre-Codex copy in `$CODEX_TMP/snap_${iter}/` and `rm` any untracked file Codex created — so the tree exactly matches the pre-Codex snapshot (this restores **pre-Codex WIP content, never HEAD**). Record any reverted Codex writes in `findings_history`. Then as in `/inspect-codex` step 5 — merge, dedupe by (file, line, issue), **verify each finding against the actual code** (no blind trust), mark TP/FP, optional single focused `codex exec resume --last --dangerously-bypass-approvals-and-sandbox` follow-up (re-snapshot + re-verify the tree after). Only true positives proceed.

### 6. Fix (Claude only — Codex never fixes)
Smallest correction per true positive. **Confine fixes to the dirty set when possible** — if a fix must touch a previously clean file (e.g. an importer), that's allowed, but **call it out in the iteration summary** so the user knows their commit-to-be will grow. Ambiguity → surface to the user and pause.

### 7. Smoke test (NO commit after)
Run, in order. Any failure resets `consecutive_clean` to 0 and becomes a finding for the next iteration:
a. `py_compile` every `.py` currently in the dirty set.
b. For each dirty `models/models_<X>.py`: `python -c "from models.models_<X> import *"`.
c. If `train*.py` / `sample.py` / `utils.py` / `config.py` / `repa/*.py` is dirty, also py_compile that file + its direct importers among the entrypoints.
Never run real training / sampling / evaluation.

### 8. Bookkeeping (NO commit) + cleanup
- `consecutive_clean += 1` iff zero true-positive findings survived adjudication AND smoke passed; otherwise `consecutive_clean = 0`.
- **Do not run `git add` / `git commit` / `git stash`.** The working tree stays dirty by design — the goal is to hand a vetted WIP back to the user.
- Delete this iteration's `$CODEX_TMP/*_${iter}.*` files.
- One-line summary: `iter N: dirty=<d>, codex=<c>/claude=<m> raw, <tp> real, <fixed> fixed, smoke=<ok|fail>, consecutive_clean=X/5`.

## Termination
- **Success:** `consecutive_clean == 5`. Print: total iterations, total real problems fixed, current dirty set (`git status --short`), and a line suggesting the user commit when ready. **Do not commit on the user's behalf.** Remove `$CODEX_TMP`.
- **Cap hit:** `iter == 20` without 5/5. Print the summary + outstanding findings + the current dirty set. Remove `$CODEX_TMP`.
- **Clean tree at start of an iteration:** stop with "nothing to check — working tree is clean. Did you mean /inspect-codex?"
- **Ambiguity pause:** halt with the question + state (`iter`, `consecutive_clean`, dirty set, pending finding). Resume on user input (do not remove `$CODEX_TMP` while paused).

## Workflow rules / What this skill must NOT do
- **No git commits** — not per-iteration, not at the end, not "just one for the fixes." Committing is the user's call. No `git stash` / `reset` / `checkout --` that drops or hides WIP. No push / force-push / amend.
- **Codex runs as the interactive `cx-yolo` TUI (yolo / unsandboxed), review-only, in a new tmux window** that is killed once its review finishes (abort if `$TMUX` unset) — kept review-only by instruction **plus** the step-2 dirty-set backup / step-5 checksum-revert guard that reverts any file it touches (restoring pre-Codex WIP, never HEAD). **Claude is the only writer.** Never `&` / `nohup` / `run_in_background`.
- No real training / sampling / evaluation — smoke test is `py_compile` + import only.
- No edits to `outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`, `collapse_smoking_test*/`, or the vendored `REPA/` (uppercase).
- Keep all Codex temp files in `$CODEX_TMP` **outside the repo** — a findings file inside the tree would contaminate the dirty set this skill is scoped to.
- No edits to clean (non-dirty) files unless required to fix a finding caused by the dirty set — and then say so in the iteration summary.
