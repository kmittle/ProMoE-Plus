# ProMoE 路由负载均衡 · 显式控制(显式干预)设计

> 本文档整理 ProMoE 原型路由(cosine-prototype top-1 router)下,**负载均衡的「显式控制」这一支**的设计思路。
> 评判标准已明确:目标是**最终生成质量 / FID**,以及**范式 novelty**;**不**以「是否 loss-free / step-0-identical / 零干扰梯度」为加减分项。
> 相关代码:`models/models_ProMoE_TC.py`(base)、`models/models_ProMoE_TC_lossfree.py`(loss-free)、`models/models_ProMoE_TC_lbcontra.py`(load-balance-aware contrastive)。

---

## 0. 定义与范围

**显式干预(explicit intervention)** = 直接对 token 的路由**相似度(`cos_sim`)或分配(assignment / selection)**写入一个干预项。
其反面是 **隐式干预(implicit)** = 不直接改相似度/分配,而是重塑「产生这些相似度的东西」(原型几何、训练目标、探索动态),让均衡作为副产品涌现。

按这个定义,下面这些都属于**显式**这一支:

- DeepSeek loss-free:`sel = cos_sim + b_i`,直接给选择分数加 per-prototype 偏置。
- `lbcontra` 的 `logit_adjust` / `reweight` / `balance_term`:直接改 routing-contrastive 的相似度 logits / 逐行权重。
- OT / Sinkhorn 平衡分配:直接算出一个平衡 assignment 并摁给路由/对比目标。
- (被否)用户 idea 1:按负载 reweight InfoNCE 对角线——同样是直接改相似度矩阵。

本文档只覆盖**显式控制族**;隐式/几何方向(OT-as-contrastive-target 重塑原型、prototype uniformity、dead-code revival、σ-clocked routing 等)在 §6 只做出口衔接,细节另文。

---

## 1. 现状:仓库里已有的显式干预

**base 路由(`models_ProMoE_TC.py::SparseMoeBlock.compute_router`)**
- cond token 与 K≈12 个可学习原型 `cluster_centers` 算 `cos_sim`;`cond_weights = softmax(cos_sim)`(生产 config 用 `router_weight_mode: identity` ⇒ `cond_weights == cos_sim`)。
- top-1 selection(`torch.topk`),被选原型的权重乘到专家输出上;uncond token(label==1000)走专用 uncond expert;shared expert 恒加。
- routing contrastive loss(`compute_routing_contrastive_loss`):对每个原型,取分给它的 token 均值做正样本,InfoNCE(对角线为正)。

**loss-free(`models_ProMoE_TC_lossfree.py`)** — 显式作用在 **selection 分数**:
- `expert_bias`(buffer,no grad):`sel_score = cond_weights + expert_bias` 只用于 top-k **选择**;权重用 **unbiased** 的 `gather(cond_weights, idx)`。
- 更新:`expert_bias += u · sign(mean_c − counts)`,`counts` 经 `all_reduce` 求全局负载。

**lbcontra(`models_ProMoE_TC_lbcontra.py`)** — 显式作用在 **contrastive loss 的相似度**(routing/dispatch 不变):
- 软基:`q_t = softmax(cos / τ_route)`,软计数 `ñ`(per-GPU local)。
- `logit_adjust`:候选 logit 加 `τ·log(ñ_j)`(balanced-softmax);`reweight`:逐行 InfoNCE 乘 `(ñ_i)^(−β)`;`balance_term`:加 `λ·load`;`soft_only`:仅软基。

**被否的显式尝试(用户 idea 1)** — 按负载 reweight InfoNCE **对角线**:
- 方向在 dead-cluster 侧是**反的**(加强饿死原型对自身稀疏质心的拉力,只会把它更深地钉进稀疏区、招不来 token);且它是 loss-shaping、和 `lbcontra` 重复、改错了元素(应作用在负样本/分配上,而非正样本对角线)。**结论:放弃这一形态。**

---

## 2. 核心统一:显式干预在「选择」层面收敛到 per-prototype 加性偏置

这是本设计最关键的一条,决定了显式支的 novelty 边界。

对 `cost = −cos_sim`、熵正则 `ε`、列边际设为 uniform(每个原型拿 `N/K` 个 token)解 Sinkhorn,最优平衡 plan 在 log 域为:

```
log P_ij  ∝  ( cos_sim_ij + f_i + g_j ) / ε
```

- `f_i`:per-token(行)对偶势,负责「每个 token 被分出去一次」。
- `g_j`:**per-prototype(列)对偶势**,负责「每个专家拿到均衡份额」。

对**选择**(`argmax_j`)而言,`f_i` 对给定 token 跨所有 `j` 是常数、不改变 argmax ⇒ **只有 `g_j` 影响选哪个专家**。而 `g_j` 就是「给每个原型的相似度加一个偏置以实现均衡」。

**⇒ loss-free 的 `b_i` 与 OT 的列对偶势 `g_j` 是同一个对象、同一个角色。** 区别只在**怎么得到它**:

| 方法 | 如何得到 per-prototype 偏置 | 性质 |
|---|---|---|
| loss-free (DeepSeek) | `b += u·sign(mean−count)`,从**实测计数**慢速 sign-feedback | 滞后积分控制器,均衡后仍 ±u 抖动 |
| Sinkhorn 对偶偏置 | `g_j` 由当前 `cos_sim` 几何**闭式**解出 | 一步到位,magnitude+geometry aware,无 ±u 振荡 |
| `lbcontra` logit_adjust | `τ·log(ñ_j)`(balanced-softmax),作用在 **contrastive** logits | 经梯度间接影响,不直接改 selection |

**推论(重要):把 OT 用到 selection / dispatch 上 ≈「闭式版的 loss-free」。** 它是更好的负载控制器,但**没有范式增量**——落在「改进 loss-free 控制器」这个已被明确否掉的低-novelty 桶里。

---

## 3. 显式干预的谱系(两条轴)

**轴 A — 干预对象(被改的相似度在哪被消费):**
- (A1) **forward selection 分数**:loss-free、Sinkhorn 对偶偏置。→ 直接改 token 实际去哪。
- (A2) **forward dispatch 分配**:OT-as-dispatch、capacity/BPR。→ 直接改分配矩阵本身。
- (A3) **contrastive loss 的相似度 / 分配**:lbcontra、(被否的)对角线 reweight。→ 经学习间接改路由。

**轴 B — 偏置如何得到:**
- (B1) load-feedback sign(loss-free)
- (B2) 闭式 OT 对偶(Sinkhorn `g_j`)
- (B3) balanced-softmax logit adjust(lbcontra)
- (B4) 硬约束分配(capacity / 匈牙利 / EC)

**关键洞察:A1 × {B1,B2,B3} 基本是同一族(见 §2),表达力都封顶在「per-prototype 标量偏置」。** 真正跳出这一族的,只有 **A2 + B4**(改变分配的**性质**,而非给分数加标量),见 §5。

---

## 4. 显式控制的天花板(the ceiling argument)

这是可以写进论文 framing 的一条论证:**所有「A1 selection 偏置」类显式干预共享同一个上限。**

1. **表达力上限**:一个 per-prototype 标量偏置 `b_j`,只能沿相似度轴**平移**每个原型的 argmax 胜出区(Voronoi 边界),**不能改变原型方向**。
2. **对几何病理无能为力**:若一个死原型与一个过载原型**近似同向**,给死原型加 `b` 也翻不动 argmax(过载原型的 `cos_sim` 仍更高),它继续饿死。计数偏置改不了「原型学成什么」。
3. **balance ≠ specialization ≠ FID**:per arXiv 2505.22323,纯负载均衡会 **under-specialize** 专家。选择偏置只重排「给定原型时 token 去哪」的计数,不触碰专门化;因此其 FID 收益上限 = 「计数均衡对 FID 的贡献」,而这可能很小(尤其有 always-on shared expert 兜底时,少数死原型的边际代价不大)。

**结论:显式-on-selection 这条线已被 loss-free / lbcontra 基本覆盖,继续在里面做「更好的 `g_j`」novelty 与上限都受限。**

---

## 5. 显式里「真正不同于 loss-free」的设计(若坚持走显式)

这些不是「另一种算 `g_j` 的方法」,而是**改变了分配的性质**——所以能越过 §4 的表达力上限。

### D1. Capacity-Clamped Top-1 + Next-Best Semantic Reroute(BPR)· 推荐的显式主打
- **机制**:batch-flatten cond pool,设容量 `C = ceil(cf · T/K)`(`cf ≥ 1`)。按 top-1 置信度排序,沿每个 token 自己的 `topk(cos_sim, 3)` 候选列表,分给**第一个未满**的原型(cumulative-count mask 向量化)。权重 = `gather(cos_sim, final_idx)`。
- **为什么不同于 loss-free**:溢出 token 被 reroute 到**它自己次相似的专家**(语义 reroute,尊重 cosine 几何),是**硬约束下的分配**,而非给分数加标量;**每个 batch 精确均衡**,无控制器滞后,也不会出现「加 `b` 翻不动 argmax」的死原型。副产品:没有空簇,contrastive loss 从不 skip。
- **train/inference**:推理设 `cf = ∞`(去掉容量)⇒ 分配塌回 `argmax(cos_sim)` = base top-1,确定性、bit-identical。
- **核心风险**:train/inference **dispatch gap**(模型可能依赖推理时不会发生的 reroute)。缓解:`cf` 贴近 1(低 reroute 率);`cf` 退火,末期趋近纯 top-1(此时若已均衡,overflow→0,gap 消失);**永不丢弃 overflow token**(丢弃=杀掉该专家梯度),只 reroute。
- **注意**:`structured_batch` case-2 的单类 batch 会让 uniform 目标失真,需排除该路径或松弛。

### D2. 闭式 OT 对偶偏置(承认 = loss-free 升级,作**对照 baseline**,不主打 novelty)
- 在已有 no-grad 偏置更新块里,用当前 `cos_sim` 跑几步 log-domain Sinkhorn 对偶得到 `g_j`,EMA 更新 `expert_bias ← (1−m)·expert_bias + m·g`。选择用 biased、权重用 unbiased(与 loss-free 卫生一致)。
- 价值:magnitude+geometry aware、一步到位、无 ±u 抖动。**但按 §2 它就是闭式 loss-free**,应作为「显式-on-selection 的最强形态」对照,用来标定 §4 天花板的实际高度,而不是当作贡献。

### D3. OT-as-dispatch + swapped-assignment 蒸馏(强显式)
- OT plan 直接做 train-time dispatch,再加蒸馏头教 batch-free argmax 复现它。是 §3 A2+B4,但要正面解决 train/infer gap + swap-CE vs MSE 早期梯度冲突。风险高于 D1,收益不明显更高,列为备选。

---

## 6. 显式的出口:通往几何(bridge,越出显式定义)

§4 的天花板要破,必须停止「给 selection 加标量」,转而**改变原型学成什么**——这已属隐式:

- **surgical OT-as-contrastive-target**:forward dispatch 仍 `argmax`,OT 平衡分配只用来定义 InfoNCE 的**正样本目标** ⇒ 重塑 prototype 几何 ⇒ 均衡**从源头**涌现。它锚在一个**代码可验证的缺陷**:`compute_routing_contrastive_loss` 里 `mask.sum()>0` 才进 `valid_clusters`、`len(valid_clusters)<2` 直接 `return 0.0` ⇒ **零 token 的原型拿不到任何对比梯度、又永不被 argmax 选中 ⇒ 永久死亡螺旋**。OT 列边际强制每个原型都拿到活的、非空、按 cos-sim 排序的正样本,把它救活。这是 **specialization 修复而非计数均衡**,是显式天花板的破法。
- 说明:严格按 §0 定义,这一版已**偏隐式**(经几何间接生效),因此不在本文档主体展开,仅作衔接。

---

## 7. 评测纪律(所有显式方案共用)

- **主指标**:FID-50k / IS,**matched training steps**(200k/300k/400k/500k)+ **matched NFE**;CFG 1.0 与 1.5 都报(路由变化与 uncond expert 交互)。
- **必须打败 `lbcontra`,不只 base ProMoE** —— 否则只是「又一个均衡旋钮」。
- **load entropy / 均衡改善 ≠ 赢**:直方图变平但 FID 不动 = **NULL,如实报为失败**;反之增加不均衡但 FID 降(真专门化)= 赢。
- **matched-compute 对照**:capacity(D1,`cf>1` 多算)/ 任何多路由变体必须等 FLOPs 或做对照,否则收益会被归给「多花算力」。
- **诊断量(仅验证机制,非成功判据)**:live prototype 数随训练(确认死亡螺旋是否打破)、per-expert 计数直方图、per-timestep-bucket 去噪残差、routing entropy。

---

## 8. 小结与建议

1. **显式-on-selection 族(loss-free / lbcontra / 闭式 OT 偏置)本质同源**(§2),表达力封顶在 per-prototype 标量偏置(§4),继续在此做增量 novelty 有限。
2. **显式里唯一有独立价值的是「改变分配性质」的 D1(capacity + 语义 reroute)** —— 它不是加偏置,是硬约束分配,能越过标量偏置的上限;主要代价是 train/inference dispatch gap,靠 `cf` 退火解决。
3. **最有 novelty 的出口在 §6(OT-as-target / 几何)**,但那已越出「显式」定义,单独讨论。
4. 可写进论文的 framing:**「显式干预的天花板」** —— 论证 loss-free / OT-dispatch / logit-adjust 共享同一 per-prototype-偏置上限与 balance≠FID 的局限,从而把贡献落到「改变分配性质(D1)」或「重塑原型几何(§6)」。

---

_附:文中代码锚点均相对 `models/models_ProMoE_TC*.py`;数学统一(§2)基于 entropic OT 的 Sinkhorn 对偶,`argmax` 对 per-token 行势 `f_i` 不变、只受 per-prototype 列势 `g_j` 影响。_
