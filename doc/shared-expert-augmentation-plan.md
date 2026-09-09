# Shared-Expert Augmentation — 实现计划 v2(思路 1–4)

> 目标:在 **base ProMoE(TC)** 上,用**融合/注意力**从更丰富的来源去**补强每个 MoE block 的 shared 专家**,全部 **zero-init → step-0 与 base 逐比特一致**、可非严格加载 base checkpoint。
>
> **v2 变更**:已把 Codex 逻辑审查(gpt-5.5 / xhigh)的 8 条修正 + 用户三点决策(思路4 区域 index 规则、缓存用 **raw shared**、14 变体全上)全部折入。核心:**不再"复用" K=2 的 `DAGFuseModule`,改为新建通用 `augment_shared(target, sources, mask)` 模块**。

---

## 0. 动机 / 前置发现

上一批 DAG-fusion 在 conditional(routed)输出 C 与 shared 输出 S 之间做结构融合,三臂:`cond_from_shared` / `shared_from_cond` / `bidirectional`。**结果:`shared_from_cond`(用 conditional 补强 shared)最好** → 推测**"补强 shared 专家"是关键杠杆**。本计划的四个思路都是**换不同来源去补强 shared**。

---

## 1. 共同设计(四思路通用)

- 基座 = **base ProMoE_TC**:Dense/MoE **交错**(`use_moe_flag[i]=(i%2==1)`,即 **block 0 Dense、奇数 index 为 MoE**);shared 专家在每个 MoE block 内、服务所有 token。
- **补强对象 = MoE block 的 shared 专家输出**(例外:思路4-(b) 挂载点作用于残差流,见 §5)。

### 1.1 通用融合模块 `augment_shared`(新建,取代 K=2 的 DAGFuseModule)

**契约**:`augment_shared(target, sources: List[Tensor], mask) -> augmented_target`
- **query-only**:只更新 `target`(=当前 shared 输出),**不更新任何 source**;支持 **K 个 source**(K 可变、可为 0→直接返回 target)。
- **zero-init delta**:输出投影 zero-init ⇒ step-0 贡献恒为 0 ⇒ 与 base 逐比特一致。
- **两种机制变体**(同一 source 集合、self/current 是否入 key 保持一致,保证 dag-vs-softmax 消融干净):
  - **`dag`**:各 source 与 target 各过 `FusedRMSNorm → down_proj(d→d_g=64)`;**GELU 门控 value-sum over K sources**(gate·value 求和);`→ zero-init up_proj(d_g→d) → 残差加回 target`;`fusion_num_iter=1`。结构照搬 `models_ProMoE_TC_dagfuse.py::DAGFuseModule` 的**单节点更新**逻辑,但把 keys 从固定 2 扩到 **1(self)+K**。
  - **`softmax`**(仅思路4 用):`target` 做 query,`sources` 做 key/value,**逐 token(同 patch 位置)在 K 个 source 上 softmax**;`→ zero-init 输出投影 → 残差加回`。cos/attn 若走 normalize+bmm,**必须 `.clamp(-1,1)`**(见 CLAUDE.md 跨对齐稳定性约束)。
- **`mask`**:cond/all 消融——`cond` 时只对 `labels!=1000` 的 token 切片做融合(uncond 不动),`all` 时对全部 token。

### 1.2 其他通用点

- 每个 MoE block 各带**可学习调制 MLP/linear**,对"来源"做变换后再融合(思路2 是**每 block 一个共享** MLP,其余每 block 一个)。
- **缓存 = raw shared**:forward 中缓存每个 MoE block **自身 shared 专家的原始输出(补强之前)**,用 **forward-local list**(非 `self.*` 持久缓存),**不 detach**(保梯度)。梯度图更浅、消融更干净(用户决策)。
- **zero-init 二次清零**:所有新投影(`up_proj` / softmax 输出投影)在 `initialize_weights()` 的 `self.apply(_basic_init)` **之后**再手动清零(照抄 dagfuse 对 `fusion_up_projectors` 的处理),否则 xavier 会覆盖 zero-init、破坏 step-0。
- forward 返回约定不变(routing contrastive loss 仍走 `AddAuxiliaryLoss`,plain tensor);融合发生在 block 内部,**routing 逐比特不变**(`compute_router`/dispatch 与 base 一致,**除思路4-(b)**,见 §5/§7)。

---

## 2. 思路1 — 前一 Dense 块输出补强 shared(`_dagfuse_dense`)

- **来源** = **前一个 Dense block 的输出**(FFN 与残差相加后的**整块输出**)。这是当前 MoE block 运行前就已算好的固定张量;**与 attention 无关**,**不是** shared 专家的(经过本 block attention 的)输入。
- **管线(Codex #3 修正)**:该"上一 Dense 输出"= 当前 MoE block 的**入口张量 x(attention 之前)**。**必须在 `DiT.forward` 主循环里捕获它、经 `DiTBlock.forward` 显式传进** MoE block 的融合入口 —— **不能**用 `SparseMoeBlock` 收到的(已过本 block attention 的)`hidden_states`。
- **做法**:上一 Dense 输出 →(每 MoE block 一个)调制 MLP/linear → `augment_shared(dag)` 补强当前 shared。
- **变体**:cond / all = **2**。

## 3. 思路2 — DenseNet 式,融合之前所有 MoE shared(`_dagfuse_densenet`)

- **来源** = **之前所有 MoE block 的 raw shared 输出**(forward-local 缓存)。
- **调制**:每个 MoE block **一个共享的调制 MLP/linear**(所有"之前的 shared"都过同一个)。
- **聚合**:`augment_shared(dag)` **一次聚合**,K = 之前 MoE block 数(可变),GELU 门控 value-sum。
- 第一个 MoE block(index 1)无"之前的 shared" → K=0 → 退化为 base。
- **变体**:cond / all = **2**。

## 4. 思路3 — 路由选择,思路2 省资源版(`_dagfuse_sharedroute`)

- **路由器(Codex #5 修正)**:**改造** MoS 的 `BlockRouter`/`PerBlockRouter`(非 drop-in),**每个 MoE block 一个**:实例化 `depth=1`、`num_teacher_blocks=num_prev_moe`、squeeze 掉多余维;**top-k 在模块外做**;query = 当前 MoE block 的 raw shared 输出;**逐 token** 在"之前的 MoE block"里选。
- **取值**:取被选中 block 的**同一 patch 位置**的 **raw shared token** →(每 block 一个)调制 MLP/linear → **softmax 权重加权** `augment_shared(dag)` 进当前 shared。**top-1 也加权**(用被选中的 softmax 概率,**不归一化成 1.0**,否则路由器无梯度)。
- **early-block 退化**:`top-k = min(k, num_prev)`;num_prev=0 → base。
- **变体**:cond/all × top-1/top-2 = **4**。

## 5. 思路4 — 区域 / Block-AttnRes(`_dagfuse_region`)

依据论文 **"Attention Residuals"(Kimi Team, arXiv 2603.15031)** 的 **Block AttnRes**:把层分成**固定大小的连续 block(区域)**;**区域代表 = 该区域最后一层的输出**;当前对**之前各区域代表**做 **per-token 深度方向聚合**;**以残差形式加回**(zero-init)。

- **区域** = **连续 3 个 block**;**区域代表 = 区域最后一个 block 的输出**(残差流 x,forward-local 缓存);**区域内 base 不动**。
- **区域 index 规则(用户确认)**:区域 = 全序列每 3 个连续 block:`{0,1,2},{3,4,5},{6,7,8},{9,10,11}`。
  - **(a) shared-attach**:在**每个区域内第一个 MoE block** 上补强其 shared(该区域其余 block 保持 base),source = **之前所有区域的代表**;cond/all 适用。(区域0 无"之前区域"→ base。)
  - **(b) resid-attach**:在**区域边界**(新区域第一个 block 之前)给**主干 x** 加注意力残差,作用于**全部 token**;cond/all 不适用。
- **两轴消融**:
  - 聚合机制:**{ dag / softmax }**(`augment_shared` 两变体,**同一 source 集合**);
  - 挂载点:**(a) 补强 shared 输出** / **(b) 区域边界残差流**。
- **(b) 诚实标注(Codex #4)**:(b) 改的是主干残差流 `x`,**zero-init 只保证 step-0 一致**;训练后 x 改变 → **改变下游 block 的路由输入** ⇒ **行为上影响路由**(全家唯一非路由不变的臂)。作为**明确的消融臂**保留(用户已同意"两个都做")。
- **变体**:(a) 2(cond/all)×2(mech)=4 + (b) 2(mech)=2 = **6**。

---

## 6. 完整变体矩阵(共 14,4 个模型文件)

| 思路 | 模型文件 | 补强来源 | 融合模块 | 消融轴 | 变体数 |
|---|---|---|---|---|---|
| 1 | `models_ProMoE_TC_dagfuse_dense.py` | 前一 Dense 块输出(经 DiTBlock 传入) | `augment_shared(dag)` | cond/all | 2 |
| 2 | `models_ProMoE_TC_dagfuse_densenet.py` | 之前所有 MoE raw shared | `augment_shared(dag)` K 变 | cond/all | 2 |
| 3 | `models_ProMoE_TC_dagfuse_sharedroute.py` | 路由选前 MoE block 同位置 raw shared | `augment_shared(dag)` + 改造 BlockRouter | cond/all × top1/top2 | 4 |
| 4 | `models_ProMoE_TC_dagfuse_region.py` | 之前各区域(3块)代表=末块输出 | `augment_shared(dag/softmax)` | mech{dag/softmax} × attach{(a)shared:cond/all,(b)resid} | 6 |

model_name = `ProMoE_TC_B_<variant>`(`_dagfuse_dense`/`_dagfuse_densenet`/`_dagfuse_sharedroute`/`_dagfuse_region`);host 训练脚本 = `train.py`;每变体一 config(带 `fuse_apply`(cond/all)、`fuse_top_k`、`fuse_mech`(dag/softmax)、`region_attach`(shared/resid)、`region_size` 等开关,默认值退化为 base)。

---

## 7. step-0 一致性 & 红线

- 所有融合 **zero-init(+ `_basic_init` 后二次清零)→ step-0 == base**、非严格可加载。
- **不碰路由**:思路1–3 与思路4-(a) 的 `compute_router`/dispatch/`forward` 与 base 逐比特一致(融合只加在 shared 输出上、且 zero-init)。
- **思路4-(b) 是唯一例外**:改主干残差流 `x`,**结构上 step-0 一致、行为上训练后影响下游路由**;作为明确消融臂保留、单独标注,不宣称"不碰路由"。

---

## 8. Codex 审查(gpt-5.5 / xhigh)已解决

| # | Codex 发现 | v2 处置 |
|---|---|---|
| 1 | `DAGFuseModule` 写死 K=2 | §1.1 新建通用 `augment_shared`,支持 K 源 |
| 2 | zero-init 被 `_basic_init` 覆盖 | §1.2 二次清零 |
| 3 | 思路1 料取错(post-attention) | §2 经 `DiTBlock.forward` 传 block 入口 x |
| 4 | 思路4-(b) 非"补强 shared"、行为动路由 | §5/§7 单列 + 诚实标注 |
| 5 | `BlockRouter` 非 drop-in | §4 改造(depth=1/num_prev/top-k 外置) |
| 6 | 缓存 raw vs augmented 未定 | §1.2 用户定 **raw**、forward-local、不 detach |
| 7 | 思路4 区域 index 歧义 | §5 精确规则(区域首个 MoE block / 区域边界) |
| 8 | dag-vs-softmax 源集合不一致 | §1.1 同一 source 集合 |

**实现第一块基石(Codex (d))**:先落地 `augment_shared(target, sources, mask)` 的确切张量契约(K 源融合、输出语义、mask、缓存输入、re-zero),其余风险随之消解。

---

## 9. 实现顺序

1. `augment_shared` 通用模块(dag + softmax 两变体)+ 单测(step-0 恒等、K=0 退化、K 变、cond mask、grad 流)。
2. 四个模型文件(依次 dense → densenet → sharedroute → region),各自 `py_compile` + import + step-0 恒等 smoke。
3. `train.py` model_dict 注册 4 个 key。
4. 14 个 config + 14 个 run 脚本(`/new-experiment`,train.py 入口)。
5. `check_output_dir.py` 逐 config 防撞 + 四向一致性。
6. `new_run.sh` 分配 GPU slot(4-GPU/个,`--dry-run` 预览)+ auto `/describe-experiment`。

_附:融合模块参照 `models/models_ProMoE_TC_dagfuse.py::DAGFuseModule`(K=2,勿改);路由器参照 MoS 的 `BlockRouter`/`PerBlockRouter`;交错/首块见 `models/models_ProMoE_TC.py` DiT.__init__(`use_moe_flag`)。Codex 协助统一 `-m gpt-5.5 -c model_reasoning_effort=xhigh`。_
