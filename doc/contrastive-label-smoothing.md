# Contrastive Label Smoothing — 模型设计与实验计划

> **一句话 thesis**:在 ProMoE 的 routing contrastive loss 上做**按可靠性自适应的 label smoothing**,以**防止 prototype 对比学习过拟合**(尤其是低样本簇用极少 token 估的噪声均值)。
> **两条定调(务必贯穿)**:
> 1. 这**不是 load balance**。我们怀疑 loss-free / balance 正则之所以间接有用,是因为它抬高了每个簇的样本数 → `cluster_mean` 估得更稳 → prototype 更不易过拟合。我们**直接**治过拟合。
> 2. **绝不干预路由**(区别于 loss-free 的 selection bias):`compute_router` 与 base 逐比特一致,ε **detach**,**不与 loss-free 叠加**。读计数只是**观测**路由结果,不是干预。
>
> 相关文档:`load-balance-design.md`(显式控制那一支的分析,本工作属于"不碰路由"的对照面)。
> 完备性:已用 codex 基于源码做过代码级完备性核查(见 §3.1 / §6 的修正)。

---

## 1. 背景与动机

**base 路由 + 对比损失**(`models/models_ProMoE_TC.py`):
- cond token 与 K≈12 个可学习原型 `cluster_centers` 算 `cos_sim`,top-1 选专家(生产 B config `top_k: 1`,故 `torch.topk` ≡ argmax;uncond token label==1000 走专用 expert;`use_shared_expert: True` 时 shared expert 恒加)。生产 config `router_weight_mode: identity`(⇒ `cond_weights == cos_sim`)。
- `compute_routing_contrastive_loss`:对每个原型 i,取 argmax 分到它的 cond token **均值** `cluster_mean_i` 作正样本;`sim_matrix = normalize(centers) @ normalize(means).T`(K×K,对角线为正样本对);`logits = sim_matrix / temperature`;`F.cross_entropy(logits, arange)`。**0-token 原型被 `mask.sum()>0` 跳过(不进 `valid_clusters`);`len(valid_clusters)<2` 直接 `return 0.0`。**

**动机**:对比损失的正样本 `cluster_mean_i` 是**用样本估计的**;**低样本簇的均值方差大、噪声重**,对它硬压 one-hot CE 会**过拟合噪声质心**。标准解法就是 **label smoothing**——软化监督目标,按估计噪声大小自适应地软化。

---

## 2. 方法

**逐行(per-prototype)label smoothing,软目标 CE**:
```
Q[i,i] = 1 - ε_i ;  Q[i,j≠i] = ε_i / (V-1)        # 行和=1
loss   = -(Q * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
```
`ε_i` 随该原型的**样本可靠性**变化(不动 `logits` / `sim_matrix` 本身)。

**平滑幅度基准(务必按类数调整)**:label smoothing 的 `ε` 默认 0.1 是 **ImageNet 1000 类**的值;这里对比 CE 只有 V≤K≈12 个"类"。按**目标熵匹配**:V=1000、ε=0.1 → H(target)≈1.01 nats;V=12 要达到同等正则需 ε≈0.2(H≈0.98 nats)。故**所有幅度旋钮的基准从 0.1 上调到 0.2**(fixed 扫描仍向上探到 0.4 看拐点)。

**第一批映射(linear,pivot+偏移;直接对齐 fixed 扫描单位):**
```
ε_i = clamp( ε_base − slope·rel_i , 0 , cap ) ,  rel_i = clamp((n_i−n̄)/n̄, −1, 1)   # rel>0 过载
```
- 基准 `ε_base=0.2`、`slope=0.2`、`cap=0.4`:过载(rel=+1)→ε=0(hard)、欠载(rel=−1)→ε=0.4、平均≈0.2。
- `n_i` = 代码里的 `mask.sum()` = 构建 `cluster_mean_i` 的样本量。

**第二批映射(invsqrt,reliability/SEM,绝对参考;更 principled 的对照):** 标准误 ∝ σ/√n,用绝对参考计数 `n_ref`(=均匀期望计数 `S·B_cond/K`):
```
ε_i = ε_max · clamp( sqrt(n_ref / max(n_i, 1)) - 1 , 0 , 1 )     # ε_max≈0.4
```
`n_i=n_ref`→0;`n_i=n_ref/4`→ε_max;`n_i>n_ref`→0(hard)。第一批跑通后再上,和 linear 对比哪种映射更好。

**负载/样本信号**:
- `n_i` 用 **per-block、本地 `mask.sum()`**(`cluster_mean_i` 是逐卡本地算的,其噪声匹配**本地**样本数);
- **EMA 平滑**(β≈0.9–0.99,per-block `[K]` buffer)以降瞬时方差(单 block 少 token + 混 timestep + CFG split → 高方差);
- **不额外加 cross-GPU all_reduce**;`ls_load_ema` 是 `register_buffer`,在默认 DDP(`broadcast_buffers=True`,见 `train.py:637`)下每次 forward 会被 **rank 0 广播** → ε 实为 **rank-0 来源**(非 rank-local)的负载估计,并与各卡当前 batch 计数混合。这没问题:各卡数据 IID(rank-0 计数 ≈ 全局)、ε 已 **detach**(无梯度、无 DDP 归约影响)、且让 ε 跨卡一致(见 §6);
- `n_i` 来自 argmax 计数(不可导),ε **detach** → 无梯度回流路由。

---

## 3. 变体与配置

- 变体 key **`ProMoE_TC_B_lsreg`**,文件 `models/models_ProMoE_TC_lsreg.py`(自包含,继承/复制 `ProMoE_TC`)。**改动点(codex 核查:仅覆盖一个方法不够)**:
  1. `SparseMoeBlock.__init__`:新增并存储 `ls_*` 参数(`**MoE_config` 传入),注册 buffer `load_ema[K]` 与(可选)`_ls_step`——**均 `persistent=False`**(见 §6)。
  2. 覆盖 `compute_routing_contrastive_loss`(软目标 CE + 可靠性 ε)。
  3. `compute_router` / `forward` / `DiTBlock` / `DiT.forward` **保持与 base 逐比特一致**(红线)。
- config flags(`MoE_config` 下,嵌套在 `DiT_B_config.MoE_config`):
  - `ls_mode`: `off`(见下)/ `fixed` / `dyn_both` / `dyn_under` / `dyn_over`
  - `ls_eps_base`(fixed 的定额 / dynamic 的 pivot;**动态基准默认 0.2**,12 类熵等效值,非 0.1)、`ls_slope`(默认 0.2)、`ls_eps_cap`(默认 0.4)
  - `ls_load_map`: `linear`(第一批,主)/ `invsqrt`(第二批对照,用 `n_ref` + `ls_eps_max`)
  - `ls_ema_beta`(默认 0.9;0 = 用当前 batch 计数)
  - `ls_warmup`(默认 **0** = 无 warmup,保持简单;>0 时用 §6 的 `_ls_step` 计数器 ramp slope)
- **`ls_mode: off` 必须 dispatch 到 base 原始 hard-mean CE**(先于任何 clamp/fp32/EMA/counter 逻辑),照 lbcontra `none → _hard_mean_infonce()` 的写法 → forward 与 base ProMoE **逐比特一致**(backward-compat + 干净对照点)。
- forward 仍返回 plain tensor(经 `AddAuxiliaryLoss`),与 base/lbcontra 同类;host 训练脚本 = **`train.py`**。

**dynamic 三档语义(按「样本数」而非「负载」重述)**:
- `dyn_under`:低样本原型多平滑,高样本保持 pivot。
- `dyn_over`:高样本原型多平滑,低样本保持 pivot。**← 证伪臂**(高样本=可靠,本不该多平滑;若它反而赢,过拟合故事错了)。
- `dyn_both`:低样本↑、高样本↓(≈`invsqrt` 的离散版)。

### 3.1 实现清单(新变体必走的项目步骤,照 lbcontra 对齐)

1. `models/models_ProMoE_TC_lsreg.py`(§3 的三处改动;`compute_router` diff 须为空)。
2. `train.py`:`import` + 在 `model_dict` 注册 `"ProMoE_TC_B_lsreg": (DiT, "DiT_B_config")`(`sample.py` 自动合并)。
3. `configs/004_ProMoE_B_lsreg_*.yaml`:`model_name: ProMoE_TC_B_lsreg`;`ls_*` flags 嵌套在 `DiT_B_config.MoE_config` 下(其余 `top_k:1`/`router_weight_mode:identity`/`use_shared_expert:True` 继承)。
4. `python -m py_compile models/models_ProMoE_TC_lsreg.py`。
5. `python scripts/check_output_dir.py --config configs/004_ProMoE_B_lsreg_*.yaml`(输出目录碰撞守卫)。
6. `scripts/lsreg/run_..._train_sample_eval.sh`,从 `scripts/template.sh` 派生,训练入口 = `train.py`。
7. `scripts/_run_times/new_run.sh --script ... --dry-run` 预览 slot,再正式分配(patch YAML `gpu_ids` + 写 wrapper)。

---

## 4. 实验矩阵(全并行,一次全开;GPU 充足)

思路1(**定额强度扫描**,回答"平滑总量→FID")与思路2(**按样本自适应分配**,回答"分配是否有用")**并行**。动态基准取 **0.2**(12 类的目标熵等效值,见 §2),不再用 0.1;`ls_mode: off`(=base ProMoE,ε=0)**已跑过,不再重复**,直接用现有 base 结果当 ε=0 参照。

| # | ls_mode | ε_base | slope | cap | load_map | 说明 |
|---|---|---|---|---|---|---|
| A1 | fixed | 0.05 | – | – | – | 定额扫描 |
| A2 | fixed | 0.10 | – | – | – | 1000 类默认(此处偏低) |
| A3 | fixed | 0.20 | – | – | – | 12 类熵等效基准 |
| A4 | fixed | 0.30 | – | – | – | |
| A5 | fixed | 0.40 | – | – | – | 看上行拐点 |
| B1 | dyn_both | 0.20 | 0.20 | 0.40 | linear | 双侧再分配(过载→hard、欠载→0.4) |
| B2 | dyn_under | 0.20 | 0.20 | 0.40 | linear | 只多平滑低样本 |
| B3 | dyn_over | 0.20 | 0.20 | 0.40 | linear | **证伪臂** |

- 参照:**现有 base ProMoE(ε=0,已跑)** + **lbcontra**(必须打败 lbcontra,不只 base)。
- 全并行 = 5(fixed)+ 3(dynamic)= **8 个新 run**。
- fixed 网格 `{0(现有), 0.05, 0.1, 0.2, 0.3, 0.4}` 必须**覆盖 dynamic 臂的实际平均 ε 范围**(≈0.2–0.3),否则无法做 §5 的去混淆。
- 第二批(可选):`ls_load_map: invsqrt`(`ε_max≈0.4`,`n_ref=S·B_cond/K`)复跑 B1–B3,对比 `linear` vs `invsqrt` 映射。

---

## 5. 去混淆 & 评测纪律

**去混淆(核心)**:dynamic 单侧臂的平均 ε ≠ 0.2(pivot)。判定"**按样本自适应分配**是否真有用":
1. 记录每个 dynamic 臂**实际平均 ε**(`E_i[ε_i]`,训练期均值)——**需日志 hook,见 §6**;
2. 画成 `(平均ε, FID)` 点,叠到 fixed 臂的 **FID-vs-ε 曲线**上;
3. **只有落在 fixed 曲线下方**(同等平均预算 FID 更低)才证明"自适应分配"有用,而非"碰巧平滑得多/少"。clamp 造成的预算漂移由此自动吸收。

**主指标**:FID-50k / IS,**matched training steps**(200k/300k/400k/500k)+ **matched NFE**;CFG 1.0 与 1.5 都报。

**判据纪律**:
- **必须打败 lbcontra,不只 base**。
- **load entropy / 均衡改善 ≠ 赢**:直方图变平但 FID 不动 = **NULL**(arXiv 2505.22323:纯均衡会 under-specialize)。本工作不追均衡,load 只是诊断。

**诊断量(非判据)**:live prototype 数、per-prototype token 直方图、per-timestep-bucket 去噪残差、`cluster_centers` 两两 cos-sim、router entropy(`TrainingMonitor`)。

**假说预测**:`dyn_under`/`dyn_both`(≈`invsqrt`)应帮;`dyn_over` 应无益或有害(证伪)。

---

## 6. 实现要点(含 codex 代码级核查的修正)

- 循环里 `cluster_counts.append(mask.sum())`,与 `cluster_means`/`valid_clusters` **同序**收集(`sim_matrix` 按 valid 位置索引,不是原始 cluster_id)。
- **`ls_mode == "off"` 走 base 原始 hard-mean CE**(未 clamp 的 `sim_matrix @ /temperature` + `F.cross_entropy`),**先于**任何 clamp/fp32/EMA/counter → 保证 off ≡ base 逐比特一致。
- 软目标 CE,**autograd-safe**:`Q` 由 detached `ε` 构造(`ε` 来自不可导计数);梯度只走 `log_softmax(logits)` 的正常路径。
- **数值**:对 active 模式(非 off),对 `cos_sim` 做 `clamp(-1,1)`(照 lbcontra `models_ProMoE_TC_lbcontra.py` 的 clamp 习惯);**fp32 是 lsreg 的显式选择**(lbcontra 只 clamp、未显式 fp32,故此处按需 `.float()`,勿写成"沿用 lbcontra 的 fp32")。
- `len(valid_clusters)<2` 仍早退;**死原型(0 token)不在矩阵、够不到 → 已知局限**(要救死原型需 OT 列边际 / revival,见 §7)。
- **EMA / step buffer 的注册与 checkpoint 语义(codex #13)**:`load_ema[K]`(及可选 `_ls_step`)在 `SparseMoeBlock.__init__` 里 `register_buffer(..., persistent=False)`。理由:`train.py` 载入时 `strict=False` 但**断言无 missing key**,persistent 新 buffer 会挡住从旧/base checkpoint 载入;且它们是 running stats,resume 时重置可接受。(对比:lossfree 的 `expert_bias` 用 persistent=True,那是要保留的控制状态,本工作不同。)
- **`ls_warmup` 的 step 来源(codex #1)**:`compute_routing_contrastive_loss` **拿不到 global step**(`train.py` 只向 `DiT.forward` 传 `context`)。解法:照 `adepth` 的 per-block 自增计数器——`register_buffer("_ls_step", torch.zeros(1), persistent=False)`,在 `self.training` 时于本方法内 `+= 1`,由它算 warmup 系数。**默认 `ls_warmup=0` 即不启用**(保持简单)。
- **平均 ε / load 日志 hook(codex #15)**:`AddAuxiliaryLoss` 隐藏了 aux loss,`train.py` 只记 `loss_dict` 显式项、无 per-block 统计通路。解法:在 block 上存 `self.last_mean_eps` / `self.last_load_hist`(no-grad),由 `TrainingMonitor`(它已按类名挂 hook)或 rank-0 周期性读取写 TB。去混淆(§5)依赖此项。
- **红线自检**:`git diff` 中 `compute_router` 无改动;无 `expert_bias`;无 selection 分数改动;ε detach;不 import/启用 loss-free。

---

## 7. 已知局限 & 可选扩展

**局限**:
- 间接:改的是对比目标 / 原型几何,路由是独立 argmax,耦合是下游副作用。
- 够不到死原型(0 token 无对应行)。
- 方向(哪种映射/哪一侧)终究经验性 → 靠矩阵定。
- **最深风险(记录备查)**:balance 也许是靠"欠用专家拿到更多 FFN/router 梯度(coverage/specialization)"起作用,而非"降低 cluster-mean 噪声"。若如此,只削弱对比信号的 label smoothing 复制不了该收益,甚至可能让低样本专家更弱 → 结果为 NULL。诊断:看低样本原型的**更新范数是否下降但使用率不变**。

**可选扩展**(按需,勿挤第一批 / 勿加实验复杂度):
- **全局 `cluster_mean`**(all_reduce token 和+计数):直接降低估计噪声,是比平滑更根本、更直接检验"噪声假设"的做法;此 all_reduce 为**降噪、非 balance、不碰路由**。可作为更强的假设闸门(若它都不动 FID,平滑几乎必然也不动)。
- **row-downweight 替代平滑**:噪声在正样本(自己的均值),负样本(别的簇均值)仍有用;均匀 label smoothing 两头都软化。对噪声行用 `w_i·CE_row`(保方向、只降信任)可能优于平滑。
- **rank/分位映射**:对"超级过载原型夹爆 rel"更 robust。
- **死原型(不碰路由)**:EMA/memory 均值当 `n_i=0` 时的正样本(detached、低权重)、prototype-only 负样本、中心斥力正则、或 all_reduce 统计让"单卡死、全局活"拿到目标。
- **簇内方差可靠性**:用 intra-cluster variance(而非仅计数)度量可靠性——更直接但更复杂。

---

## 8. 与其它变体的关系

| | 干预点 | 框架 | 碰路由? |
|---|---|---|---|
| **loss-free** | selection 分数加 `expert_bias` | load balance(闭环控制器) | **是** |
| **lbcontra** | 对比 loss(reweight/logit_adjust/…) | load-balance-aware | 否(但仍是 balance 框架) |
| **本工作 `lsreg`** | 对比 loss 的**监督目标**(label smoothing) | **anti-overfitting(按样本可靠性)** | **否(红线)** |

本工作的贡献定位:**拿到 balance 间接带来的"防过拟合"收益,却不付出"干预路由"的代价**。若碰了路由,就退化为又一个 balancer。
