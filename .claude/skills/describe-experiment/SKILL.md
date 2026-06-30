---
name: describe-experiment
description: Write a plain-text description of a ProMoE run-time experiment wrapper. Reads a scripts/_run_times/<date>/<slot>-<desc>.sh wrapper, traces it (wrapper → semantic run script → config → model file / CLAUDE.md variant table), and writes <wrapper-stem>-describe.txt next to the wrapper in the same date dir — a numbered 1/2/3 list of this experiment's core changes relative to the baseline, most-important change first, bilingual (中文为主 + English technical terms). "Baseline" is written against both the base ProMoE model and the immediate parent variant. Auto-fires right after a run-time wrapper is created or re-bucketed (it is invoked at the end of /new-experiment and /rerun-experiment), and on requests like "为 scripts/_run_times/<date>/<slot>-<desc>.sh 写实验描述" / "给这个实验写一段描述" / "describe this experiment". Does NOT launch any run, and does NOT commit, push, or amend.
---

# /describe-experiment — Write a `*-describe.txt` for a run-time wrapper

Produces a short, human-readable description of what one experiment changes **relative to the
baseline model**, and drops it next to the launch wrapper so the date dir self-documents alongside
`commands.md`. The description is a numbered list of **core changes, most-important first**.

This skill **reads source files and writes one `.txt` file** (`<wrapper-stem>-describe.txt`). It
never launches training, sampling, or evaluation, and never commits.

## When it runs
- **Auto-fire (primary):** right after a run-time wrapper is written. `/new-experiment` invokes it on
  the wrapper it just created; `/rerun-experiment` invokes it on the `_vN` wrapper it just renamed to.
- **Manual:** when the user points at a wrapper (or a whole date dir) and asks for a description —
  e.g. "为 `scripts/_run_times/2026_06_21/1.1-B_ec_bc_proto_t_direct_v2.sh` 写实验描述",
  "给这批实验各写一段描述", "describe this experiment".

## Step 0 — Resolve the target wrapper(s)
- A single wrapper path → that one wrapper.
- A date dir (e.g. `scripts/_run_times/2026_06_21/`) or "这批/all" → every `<slot>-<desc>.sh` wrapper
  in it (every `*.sh` **except** `new_run.sh` / helpers; skip `commands.md`).
- No target given → default to **today's** date dir; if it doesn't exist, use the most recent date
  dir under `scripts/_run_times/` and say which.

## Step 1 — Trace the wrapper to the experiment's real definition
For each wrapper, follow the same chain `/command-table` uses, then go one level deeper into the
model so the description is grounded in what the code actually does:
1. **Wrapper** → the `exec bash "${REPO_ROOT}/<path>"` line gives the **semantic run script**. (Also
   note `Slot:` / `GPUs:` from the header comment — context only, not part of the change list.)
2. **Semantic script** → its `^CONFIG=` line gives `configs/<name>.yaml`.
3. **Config** → read `model_name` and the full `MoE_config` / `repa_config` blocks, **including
   inline comments** (proto_t / anchor / proto_choice configs carry comments that state the exact
   mechanism and the step-0 behavior). The config basename (minus `.yaml`) is `custom_cfg_name`.
4. **Model + docs** → from `model_name`, find the variant in CLAUDE.md's **variant-family table**
   ("Family / Files / Key difference") and the **Key MoE Parameters** notes, and open the backing
   `models/models_*.py` (or read its class header / the relevant `forward`) to confirm the mechanism
   and the step-0 relationship to base ProMoE. Ground every claim — do **not** invent behavior the
   config/model doesn't show.

## Step 2 — Pin down the two baselines
The description is written against **both**:
- **Base ProMoE** — the original two-step-router model: `ProMoE-TC` for token-choice families,
  `ProMoE-EC` for Expert-Choice families (`models_ProMoE_TC.py` / `models_ProMoE_EC.py`).
  Characterized by: conditional routing (uncond expert for class==1000) → prototypical routing via
  **static learnable `cluster_centers`** (cosine-sim), top-1 token-choice, shared expert, and the
  routing **contrastive** loss. This is the "Key difference" anchor used by CLAUDE.md's variant rows.
- **Immediate parent variant** — the one step this experiment is an increment over. Find it two ways
  and use whichever is the meaningful "one step back":
  - **Inheritance parent** — the class the model file inherits from (`class ProMoE_..._<v>(<Parent>)`).
  - **Sweep sibling / default flag** — for a config-driven flag (`proto_t_update_mode` direct↔residual,
    `anchor_apply_mode` routing↔replace, `contrastive_proto_choice_ratio` 083↔125, EC-BC vs TC), the
    parent is the same model with the **default / sibling** flag value.

If the immediate parent *is* base ProMoE (e.g. a variant that only adds one mechanism on top of base,
or `proto_choice` where only the contrastive loss differs), say so explicitly instead of inventing a
second baseline.

## Step 3 — Write the change list (the actual content)
A numbered list of **core changes vs the baseline, most-important change first**:
- **Ordering:** the headline mechanism that defines the variant is #1. Then the next-most-defining
  change, then the rest. Routing-family difference (TC vs EC-BC) and flag-specific detail
  (direct/residual, 083/125, routing/replace) are usually #2/#3.
- **Count:** aim for **3 points**. Don't pad — if the experiment genuinely has only 1–2 distinct core
  changes, write 2; never exceed 4.
- **Both baselines per point where it matters:** state the change vs base ProMoE, and where the
  experiment is a sub-variant, note the increment vs the immediate parent (e.g. "vs proto_t-residual:
  uses `direct` update mode …").
- **Bilingual:** 中文为主, keep model/flag/field names and key terms in English inline
  (`cluster_centers`, `proto_t_update_mode`, EC-BC, InfoNCE, …). Don't translate identifiers.
- **Grounded & specific:** name the exact config flag + value and the mechanism (formula/shape when
  the config comment or model gives one). Note step-0 behavior when the variant documents it
  (e.g. "step-0-identical to base ProMoE" / "not step-0-identical — train fresh").

## Step 4 — Write `<wrapper-stem>-describe.txt`
- **Filename:** strip the trailing `.sh` from the wrapper's basename and append `-describe.txt`.
  `1.1-B_ec_bc_proto_t_direct_v2.sh` → `1.1-B_ec_bc_proto_t_direct_v2-describe.txt`.
- **Location:** the **same date dir** as the wrapper.
- **Overwrite** if it already exists (regenerated artifact, like `commands.md`).
- **Format** — a 2-line header identifying the experiment, then the numbered change list:

  ```
  实验 / Experiment: <wrapper-stem>
  model_name: <model_name>  |  config: configs/<name>.yaml
  基线 / Baseline: 基础 ProMoE-<TC|EC>（<一句话刻画>）；直接父变体 / Parent: <parent 一句话>

  核心改动 / Core changes（相对基线，重点优先）:
  1. <重点改动 — 中文说明（English terms / flags inline）>
  2. <次要改动>
  3. <其余改动>
  ```

Worked example — `1.1-B_ec_bc_proto_t_direct_v2.sh` (`model_name: ProMoE_EC_BC_B_proto_t`,
`proto_t_update_mode: "direct"`):

```
实验 / Experiment: 1.1-B_ec_bc_proto_t_direct_v2
model_name: ProMoE_EC_BC_B_proto_t  |  config: configs/004_ProMoE_B_EC_BC_proto_t_direct_v2.yaml
基线 / Baseline: 基础 ProMoE（静态 cluster_centers 原型 + 两步路由）；直接父变体 / Parent: ProMoE_EC_BC_B（EC-BC 路由，静态原型）

核心改动 / Core changes（相对基线，重点优先）:
1. 时间步条件原型 / timestep-conditioned prototypes（proto_t）：把固定可学习的 `cluster_centers` 换成由 `PrototypeMLP(concat(cluster_centers, t_emb))` 逐样本生成的原型，使余弦相似度路由在与噪声水平对齐的空间中进行；MoE block forward 因此多接收一个 `t_emb` 参数。
2. `direct` 原型更新模式 / update mode：`prototype_t = proto_proj(cluster_centers) + MLP(...)`（`proto_proj` 恒等初始化 + `fc2` 零初始化 → step-0 与基础 ProMoE 等价，但 cc→原型映射可训练，额外约 +3.5M 参数）。相对父变体的 residual 模式（`cc + MLP(...)`）这是它的消融轴。
3. EC-BC 批展平 Expert-Choice 路由：相对基础 ProMoE-TC 的逐 token top-1 路由，专家在批展平的 cond-token 池（`B_cond*S`）上各取 top-k（容量 `k = B_cond*S/E*top_k`），经 `torch.gather`/`index_add_` 派发；contrastive 辅助损失仍走 `AddAuxiliaryLoss`。
```

## Step 5 — Report
In chat: the wrapper(s) described, the `.txt` path(s) written, and a one-line gist of each. If invoked
standalone over a date dir, say how many were written.

## Edge cases
- **Wrapper not the auto-generated form** (no `exec bash` line): describe from whatever config/script
  it does reference; if none is resolvable, say so and write nothing rather than guessing.
- **Config-driven ablation where the model file is shared** (no new `models_*.py`): the parent is the
  same model with the default flag; trace the flag's meaning from the config comment + CLAUDE.md
  "Key MoE Parameters". Don't claim a model-code change that didn't happen.
- **Variant not yet in CLAUDE.md's table:** fall back to the config + model file; describe only what
  they show, and flag (in chat) that the docs don't cover it yet.
- **Re-bucketed `_vN` wrapper** (from `/rerun-experiment`): the change list is about the *variant*, so
  it's the same as the base name — but regenerate it freshly (the model code changed, which is why it
  was re-bucketed) and write it under the `_vN` stem.

## What this skill must NOT do
- **No real training / sampling / evaluation runs.** It only reads files and writes `*-describe.txt`.
- **No git commits, push, force-push, or amend.** Leave the `.txt` dirty for the user to commit.
- No `git add -A` / `git add .` — if the user later asks to commit, stage explicit paths only.
- No edits to runtime artifact dirs (`outputs/`, `pretrained_ckpt/`, `training_logs/`, `tb_smoke_*/`,
  `collapse_smoking_test*/`) or the vendored `REPA/` (uppercase) subproject.
- Do not edit the wrapper, the semantic run script, the config, or the model files — read-only except
  for the single `<wrapper-stem>-describe.txt` output.
