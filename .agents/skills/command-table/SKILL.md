---
name: command-table
description: Organize ProMoE run-time launch wrappers from a dated scripts/_run_times directory into a Notion/Excel-importable commands.csv. Trace each wrapper through its semantic run script and YAML config to record the experiment description, current Git branch, launch command, and output directory. Use for explicit $command-table requests or when the user asks to organize, summarize, or generate a command table for a dated run-time batch. Never launch experiments, stage, commit, push, or amend.
---

# Build a Run-Time Command Table

Create one `scripts/_run_times/<date>/commands.csv` from the launch wrappers in that date directory. Read experiment definitions only; write or replace only that CSV.

## Establish Context

1. Work from the repository root.
2. Read root `AGENTS.md` completely and obey it. Read relevant sections of `CLAUDE.md` only as supplemental project documentation; `AGENTS.md` and the user's request take precedence.
3. Inspect `git status --short` before writing. Preserve all unrelated changes.
4. Use `rg` for searches when available and fall back to `grep` or `find` without changing behavior.

## Resolve the Date Directory

- Use a date or path named by the user.
- Otherwise use today's `scripts/_run_times/$(date +%Y_%m_%d)/` directory.
- If today's directory does not exist, use the most recent existing date directory and report that choice.
- Select every `*.sh` launch wrapper except `new_run.sh` and helpers.
- Sort wrapper names by slot in natural order: `1.1`, `1.2`, `2.1`, `2.2`, then full-server slots such as `3`. Preserve historical slot forms such as `X.3` and `X.4` when present.

## Trace Each Wrapper

For every wrapper, resolve these values from repository files rather than inferring them from names alone:

1. Read the header `# Date group: <date>   Slot: <slot>   GPUs: <list>` for the slot and GPU list.
2. Read the `exec bash "${REPO_ROOT}/<path>"` line for the semantic run script.
3. Read the semantic script's top-level `CONFIG=` assignment for the YAML path.
4. Read the config's top-level `model_name:`. Set `custom_cfg_name` to the config basename without `.yaml`.
5. Treat an intentional top-level `output_dir` as the output root, defaulting to `outputs`. Derive the full directory as `{output_root}/{model_name}/{custom_cfg_name}/`; the override never replaces the `model_name` and config-basename leaf components. When uncertain, confirm the rule in `scripts/check_output_dir.py`.

Map those values to exactly four columns:

- `实验描述`: a concise human-readable label containing slot, GPU range, and variant, for example `Slot 1.1 · GPU 0-3 · ProMoE-B EC-BC proto_t (direct)`.
- `git分支`: the current branch from `git rev-parse --abbrev-ref HEAD`, unless the user specifies a per-experiment branch.
- `启动命令`: `bash scripts/_run_times/<date>/<wrapper>`.
- `输出位置`: the derived output directory.

## Render the CSV

- Read `command-tables/command-table-template.csv` and preserve its exact header and column order: `实验描述,git分支,启动命令,输出位置`.
- Write one record per wrapper in sorted slot order.
- Use plain-text cells. Do not add Markdown backticks or pipe separators.
- Apply RFC 4180 quoting mechanically: quote cells containing commas, double quotes, or newlines, and double every internal double quote.
- Make the header the first line. Add no title, prose, notes, or trailing explanatory block to the file.
- Use `apply_patch` for the file edit. Overwrite an existing `commands.csv` because it is generated state.

## Handle Irregular Inputs

- For a non-generated wrapper without `Slot:` or `exec bash`, trace its actual launch command. Leave unresolvable columns empty and report them.
- If a semantic script references multiple configs, use its top-level `CONFIG=` assignment.
- If the config lacks `model_name`, leave `输出位置` empty and report the row.
- If a legacy `commands.md` exists, report that it may be stale. Do not delete it.

## Verify and Report

Re-read the completed CSV and verify:

- the header is exact;
- the data-row count equals the wrapper count;
- every launch command names an existing wrapper;
- every resolvable output path matches its config.

Report the date directory, wrapper count, output path, unresolved rows, the current-branch caveat, and the project tmux launch convention. Keep those notes in chat, never in the CSV.

## Boundaries

- Do not run training, sampling, evaluation, preprocessing, or GPU jobs.
- Do not edit wrappers, semantic scripts, configs, templates, runtime artifact directories, or uppercase `REPA/`.
- Do not stage, commit, push, force-push, or amend.
- Write only the target `commands.csv`.
