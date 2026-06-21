---
name: command-table
description: Organize the run-time launch wrappers in a scripts/_run_times/<date>/ directory into a Markdown command table. Reads every <slot>-<desc>.sh wrapper in the date dir, traces each one (wrapper → semantic run script → config) to fill the columns, renders using command-tables/command-table-template.md, and writes the result to commands.md in that same date dir. Use when the user asks to turn/organize/summarize the run-time commands (the scripts under scripts/_run_times/<date>/) into a command table — including phrasings like "把 scripts/_run_times/<date> 中的指令整理为命令表格", "整理/生成命令表格", or "make a command table for the run-time scripts". Does NOT launch any run, and does NOT commit, push, or amend.
---

# /command-table — Build a run-time command table from a date directory

Turns the auto-generated launch wrappers in one `scripts/_run_times/<date>/` directory into a
single reference table (`commands.md`) so the experiments launched that day can be seen at a glance:
description, git branch, launch command, and output location.

This skill **reads and writes one Markdown file only** (`commands.md`). It never launches training,
sampling, or evaluation, and never commits.

## Step 0 — Resolve the target date directory
- If the request names a date (e.g. `2026_06_21`) or a path, use `scripts/_run_times/<date>/`.
- Otherwise default to **today's** date dir (`scripts/_run_times/$(date +%Y_%m_%d)/`); if that
  doesn't exist, use the most recent existing date dir under `scripts/_run_times/` and say which.
- List the wrapper scripts in it: every `*.sh` file **except** `new_run.sh` and any helper. The
  wrappers follow the `<slot>-<desc>.sh` name pattern (e.g. `1.1-B_ec_bc_proto_t_direct.sh`).
- Sort wrappers by slot in natural order (`1.1`, `1.2`, `2.1`, `2.2`, … then full-server `3`).

## Step 1 — Trace each wrapper to fill the four columns
For each wrapper file, gather:
1. **Slot + GPUs** — from the wrapper's header comment line
   `# Date group: <date>   Slot: <slot>   GPUs: <list>`.
2. **Semantic script** — from the `exec bash "${REPO_ROOT}/<path>"` line (the script it delegates to).
3. **Config** — read that semantic script's `^CONFIG=` line → `configs/<name>.yaml`.
4. **model_name** — read `model_name:` from that config. `custom_cfg_name` = the config's basename
   without `.yaml`.
5. **Output dir** — `outputs/{model_name}/{custom_cfg_name}/` (per CLAUDE.md "Configuration System").

Then map to the template's columns:
- **实验描述** — a human-readable label combining slot, GPU range, and the variant. Derive the
  variant from the config/script name. Example: `Slot 1.1 · GPU 0-3 · ProMoE-B EC-BC proto_t (direct)`.
- **git分支** — the current branch (`git rev-parse --abbrev-ref HEAD`) by default. If the user states
  a per-experiment branch, use that. Either way, note in the report that the branch column reflects
  the **current** checkout and should be verified if experiments target different branches.
- **启动命令** — `bash scripts/_run_times/<date>/<wrapper>` (the wrapper is the entry point). Per
  the project rule, runs launch in a tmux window — include a single note line above the table with
  the tmux form rather than repeating it in every cell.
- **输出位置** — the output dir from step 5, e.g. `outputs/ProMoE_EC_BC_B_proto_t/004_ProMoE_B_EC_BC_proto_t_direct/`.

## Step 2 — Render from the template
- Use `command-tables/command-table-template.md` as the table skeleton (header row + separator:
  `实验描述 | git分支 | 启动命令 | 输出位置`). Keep exactly those four columns.
- One row per wrapper, in sorted slot order.
- Prepend a short note line: launch each command in a new tmux window of the current session,
  e.g. `tmux new-window -t "$(tmux display-message -p '#S')" -n <name> '<启动命令>'`.

## Step 3 — Write `commands.md`
- Write the rendered table to `scripts/_run_times/<date>/commands.md` (overwrite if it exists —
  this is a regenerated summary, not hand-maintained state).
- Report: the date dir, how many wrappers were summarized, the output path, and the branch caveat.

## Edge cases
- A wrapper that is **not** the auto-generated form (no `Slot:`/`exec bash` lines): fall back to
  whatever launch command it does contain; leave columns you cannot resolve blank and flag them.
- A semantic script with multiple config references: use the top-level `^CONFIG=` assignment.
- A config missing `model_name`: leave 输出位置 blank for that row and flag it.

## What this skill must NOT do
- **No real training / sampling / evaluation runs.** It only reads files and writes `commands.md`.
- **No git commits, push, force-push, or amend.** Leave `commands.md` dirty for the user to commit.
- No `git add -A` / `git add .` — if the user later asks to commit, stage explicit paths only.
- No edits to runtime artifact dirs (`outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`,
  `collapse_smoking_test*/`) or the vendored `REPA/` (uppercase) subproject.
- Do not edit the wrapper scripts, the semantic run scripts, the configs, or the template — read-only
  except for the single `commands.md` output.
