# AGENTS Outline

Use sections only when the repository contains enough evidence to support them. `AGENTS.md` should be a compact execution guide, not a full design document.

## Default Sections

### Project Structure & Module Organization

- Name the real entrypoints and the directories another agent must understand first.
- Describe output or artifact locations if the workflow depends on them.
- Call out subprojects explicitly when the repo has more than one.

### Build, Test, and Development Commands

- Prefer exact commands already documented in `README*`, `CLAUDE.md`, `package.json`, `pyproject.toml`, `Makefile`, shell wrappers, or CI files.
- Include only the commands that matter for day-to-day work: setup, build, train, sample, eval, test, lint, format, smoke checks.
- If the repo uses multiple environments, separate them clearly.

### Coding Style & Naming Conventions

- Infer style from touched files and formatter or linter config.
- Record only conventions that are concrete and repo-specific.
- Mention naming schemes that affect discoverability, such as numbered config prefixes or required file patterns.

### Testing Guidelines

- State the real test or smoke-check surface for the repo.
- Match recommendations to the code touched: training path, sampling path, API route, frontend page, evaluation job, and so on.
- If there is no formal test suite, say so and list the practical checks that are expected instead.

### Configuration Notes

- Capture environment variables, config merge rules, cache paths, data layout assumptions, GPU or runtime requirements, and output path conventions.
- Mention any easy-to-miss rules that would break a run if omitted.

### Commit & Pull Request Guidelines

- Include this section only if the repo already documents commit style, PR expectations, or required evidence.
- Keep it factual: subject style, scope expectations, required logs, screenshots, or metrics.

## Merge Rules

- Preserve valid repository-specific instructions already present in `AGENTS.md`.
- Use `CLAUDE.md` as an input source, not as a file to mirror.
- Compress duplicated content instead of keeping parallel versions of the same guidance.
- Drop stale instructions that no longer match the tree, commands, or manifests.

## Extraction Checklist

- Verify every path you mention exists.
- Verify every command references a real file, script, or target.
- Prefer stable instructions over ephemeral details unless the ephemeral detail is operationally critical.
- Keep the file short enough that another agent can scan it quickly before starting work.
