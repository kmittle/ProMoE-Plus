---
name: clear
description: Reset the working context for the current conversation without pretending to erase the actual transcript. Use when Codex is asked to clear context, start fresh, drop prior assumptions or plans, stop carrying forward earlier task state, or treat the next request as a new task.
---

# Clear

## Overview

Use this skill to simulate a fresh working state. Discard transient task context, pending plans, and unstated assumptions from earlier messages, then wait for the next user request.

## Hard Boundary

You cannot actually delete the existing system prompt, chat transcript, or tool outputs. Do not claim the history is gone. Instead, stop relying on prior task context except for durable facts that can be re-read from the repository or are explicitly repeated by the user.

## Behavior

1. Abandon any active mental plan from the previous task.
2. Drop task-local assumptions, tentative conclusions, remembered diffs, and ad hoc constraints unless the user repeats them or they are re-discovered from files.
3. Do not summarize old work unless the user asks for that summary explicitly.
4. Reply with a short reset confirmation, then wait for the next task.
5. For the next task, rebuild context from the new prompt and the current repo state instead of relying on memory.

## Response Style

Use a minimal confirmation such as:

`已清除当前工作上下文。请给我新的任务。`

If the user asked to preserve a specific constraint, note that exception explicitly.

## When Not To Use

- Do not use this skill for deleting files, clearing caches, or cleaning a git worktree.
- Do not use it as a substitute for a truly isolated new session when strict separation is required.
- If the user wants persistent instructions removed from a file such as `AGENTS.md` or `CLAUDE.md`, edit the file itself instead of using this skill.

## After Clear

- Treat the next request as a new task.
- Re-open files instead of trusting stale memory.
- Recompute plans from scratch.
