---
name: describe-experiment
description: Write grounded bilingual plain-text descriptions for ProMoE run-time experiment wrappers. Trace a dated scripts/_run_times wrapper through the semantic run script, YAML config, registry, model implementation, and project documentation, then write a sibling describe.txt file with the core changes versus base ProMoE and the immediate parent variant. Use for explicit $describe-experiment requests, requests to describe one wrapper or a dated batch, and as the final composed step of $new-experiment or $rerun-experiment. Never launch runs or commit changes.
---

# Describe a ProMoE Experiment

Create a concise `*-describe.txt` beside each target run-time wrapper. Ground every statement in the config, registry, model code, or project documentation. Write Chinese-first prose while preserving English identifiers and technical terms.

## Establish Context

1. Work from the repository root.
2. Read root `AGENTS.md` completely and obey it.
3. Read the relevant `CLAUDE.md` model-registry, variant-family, auxiliary-loss, and MoE-parameter sections only as supplemental technical documentation. When documentation conflicts, prefer `AGENTS.md`, then executable code and config.
4. Inspect `git status --short` before writing and preserve unrelated changes.
5. Use `rg` for searches when available; fall back to `grep` or `find` if needed.

## Resolve Targets

- For one wrapper path, describe that wrapper.
- For a date directory or a request covering the whole batch, select every `<slot>-<desc>.sh` wrapper except `new_run.sh` and helpers.
- With no explicit target, use today's date directory. If it does not exist, use the most recent existing date directory and report the choice.
- When composed by `$new-experiment` or `$rerun-experiment`, describe only wrappers newly created, renamed, or explicitly resolved and reused as targets by that workflow.

## Trace the Experiment Definition

For each wrapper, resolve the full chain:

1. Read the wrapper's `exec bash "${REPO_ROOT}/<path>"` target and optional `Slot:` / `GPUs:` header.
2. Read the semantic script's top-level `CONFIG=` assignment.
3. Read the config's top-level `model_name` and complete model configuration, including `MoE_config`, both levels of `repa_config` where applicable, and explanatory inline comments.
4. Locate the exact registry entry in the appropriate training entrypoint and identify the registered class plus config key.
5. Open the backing `models/models_*.py` implementation. Inspect the relevant class, inheritance chain, initialization, routing/loss logic, and `forward` behavior.
6. Consult the supplemental variant-family and parameter documentation only to clarify what the code and config establish.

Do not infer behavior from a filename when the actual config or implementation is available. Do not invent formulas, parameter counts, initialization guarantees, or step-0 equivalence.

## Establish Both Baselines

Describe changes relative to both baselines when they differ:

- **Base ProMoE:** use `ProMoE-TC` for token-choice families and `ProMoE-EC` for Expert-Choice families. The base combines conditional cond/uncond routing, static learnable `cluster_centers`, cosine-similarity prototypical routing, a shared expert, and routing contrastive guidance. Confirm family-specific details in the backing implementation.
- **Immediate parent:** prefer the direct inheritance parent when it represents the meaningful previous design. For flag-driven ablations, use the default or sibling flag setting instead, such as `direct` versus `residual`, routing versus replacement, or one sweep ratio versus another.

If the immediate parent is base ProMoE, state that directly. If parentage is ambiguous, explain the evidence in chat and avoid an unsupported claim in the file.

## Write the Core Changes

- Order changes by importance. Put the mechanism defining the variant first.
- Aim for three numbered points. Use two when only two distinct changes exist and never exceed four.
- State exact config fields and values where they define behavior.
- Explain routing-family differences and ablation-specific settings only when they distinguish this experiment.
- Mention step-0 equivalence, non-strict checkpoint loading, train-fresh requirements, or parameter changes only when code or documentation proves them.
- Write primarily in Chinese and retain identifiers such as `cluster_centers`, `proto_t_update_mode`, EC-BC, and InfoNCE in English.
- Compare with base ProMoE and, where meaningful, explain the incremental difference from the immediate parent.

## Write the Description File

Strip `.sh` from the wrapper basename and append `-describe.txt`. Write the file beside the wrapper with `apply_patch`; overwrite an existing description because it is generated state.

Use this shape:

```text
实验 / Experiment: <wrapper-stem>
model_name: <model_name>  |  config: configs/<name>.yaml
基线 / Baseline: 基础 ProMoE-<TC|EC>（<一句话刻画>）；直接父变体 / Parent: <parent 一句话>

核心改动 / Core changes（相对基线，重点优先）:
1. <最重要改动，中文说明并保留 English identifiers>
2. <次要改动>
3. <其余独立改动>
```

Example of the expected specificity:

```text
1. 时间步条件原型 / timestep-conditioned prototypes（proto_t）：将静态 `cluster_centers` 变为由 `t_emb` 条件化的逐样本原型，使路由随噪声时间步变化。
2. `proto_t_update_mode: direct`：说明 exact update path、初始化，以及相对 residual sibling 的差异；只有实现能证明时才写 step-0 等价。
3. EC-BC 路由：说明 expert-choice 的选择域和容量计算，并明确它相对基础 token-choice 路由的变化。
```

## Handle Edge Cases

- If a wrapper lacks an `exec bash` target, trace any script or config it actually references. If no definition is resolvable, write nothing and report the blocker.
- For a shared-model config ablation, describe the flag behavior from its consumer and comments. Do not claim a new model file exists.
- If documentation does not cover the variant, use config plus implementation and note the documentation gap in chat.
- For a `_vN` rerun wrapper, regenerate the description from current code. Describe the variant behavior, not the filename change alone.

## Verify and Report

Re-read each output and verify its wrapper, config, and `model_name` exist and agree. Confirm every numbered claim has a source in the files inspected. Report each wrapper, output path, and a one-line gist; for a batch, also report the count.

## Boundaries

- Do not run training, sampling, evaluation, preprocessing, or GPU jobs.
- Do not edit wrappers, semantic scripts, configs, models, runtime artifacts, or uppercase `REPA/`.
- Do not stage, commit, push, force-push, or amend.
- Write only the target `*-describe.txt` files.
