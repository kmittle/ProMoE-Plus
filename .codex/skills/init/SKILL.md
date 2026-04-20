---
name: init
description: Build or refresh a repository-local AGENTS.md by inspecting the current project instead of copying a stale template. Use when Codex is asked to initialize project guidance, migrate CLAUDE.md-style instructions into AGENTS.md, sync AGENTS.md with the repo's actual structure and commands, or update AGENTS.md after workflows, scripts, configs, or entrypoints change.
---

# Init

## Overview

Inspect the repository, extract the developer workflow that actually exists, and write or update `AGENTS.md` in a concise operational style. Prefer repository facts over assumptions, and preserve valid manual instructions that already exist.

## Workflow

1. Run `scripts/project_snapshot.py <repo-root>` to collect a compact inventory before reading files manually.
2. Read the highest-signal sources only: existing `AGENTS.md`, `CLAUDE.md`, `README*`, build manifests, root entrypoints, representative files under major source directories, and wrapper scripts that define the real workflow.
3. Decide whether the task is:
   - create a new `AGENTS.md`
   - refresh an existing `AGENTS.md` in place
   - merge facts from `CLAUDE.md` and current code into `AGENTS.md`
4. Write or update `AGENTS.md` with exact paths, exact commands, and repo-specific guardrails.
5. Verify every referenced file path exists and every command maps to real files, scripts, or package targets.

## Source Priority

- Treat the repository state as the source of truth when docs disagree.
- Preserve still-valid policy or process notes already present in `AGENTS.md`.
- Use `CLAUDE.md` as a rich secondary source, but do not copy it verbatim.
- Use `README*`, manifests, and wrapper scripts to recover commands and runtime assumptions.
- Ignore generated or cached directories unless they are part of the required workflow.

## Content Rules

- Keep `AGENTS.md` concise and execution-oriented. Drop narrative project history unless it changes how work must be done.
- Prefer commands that the repo already advertises in scripts, manifests, or docs. Do not invent commands.
- Capture only repo-specific conventions: layout, entrypoints, config rules, testing commands, generated-output paths, cache paths, environment split, deployment quirks, and review expectations.
- If the repo contains multiple subprojects, separate their commands and assumptions explicitly.
- When updating, merge missing facts into the existing file instead of replacing user-authored guidance wholesale.
- If a section cannot be supported by repo evidence, omit it.

## Recommended Outline

Use `references/agents-outline.md` to choose sections. The common default is:

- `Project Structure & Module Organization`
- `Build, Test, and Development Commands`
- `Coding Style & Naming Conventions`
- `Testing Guidelines`
- `Configuration Notes`
- `Commit & Pull Request Guidelines`

Omit sections that are unsupported or irrelevant.

## Update Strategy

When `AGENTS.md` already exists:

- Keep valid repository rules already present.
- Remove or rewrite stale statements that conflict with the current tree, commands, or tooling.
- Pull over missing operational detail from `CLAUDE.md` or other docs, then compress it into the terser `AGENTS.md` style.
- Avoid duplicating long background sections that help humans read but do not help an agent execute.

## Validation

- Re-run `scripts/project_snapshot.py` if the repo is large or changed during the turn.
- Check that each path named in `AGENTS.md` exists.
- Check that each command references a present file, script, binary, or package target.
- If the repo has environment splits, caches, generated artifacts, or dataset assumptions, make sure they are called out explicitly.

## Resources

- `scripts/project_snapshot.py`: produce a compact repo inventory before drafting or refreshing `AGENTS.md`.
- `references/agents-outline.md`: default section outline and extraction checklist.

## Example Trigger

Use this skill when the request sounds like:

- "Initialize Codex for this repo."
- "Create an AGENTS.md like CLAUDE.md, but based on the current project."
- "Refresh AGENTS.md after the training scripts changed."
- "Merge repo facts from README and CLAUDE.md into AGENTS.md."
