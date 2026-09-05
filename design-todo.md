# design-todo：ProMoE_TC 改进计划（MoE 主线）

> **执行顺序（2026-09-04）**：先补齐历史表中缺失的真实 300K/500K evaluator 结果，再做四点组合。补测必须 training from scratch、固定 `global_seed=0`，不得用中途 checkpoint 代替独立实验。训练过程不得以 50K 指标或主观判断提前停止；只有预定的 300K/500K 评估完成后，才根据生成质量、路由负载和专家分工决定去留。一次性的 200K 窗口交接不再复制到新实验。

> **目录 / Index**（历史独立改进组；当前优先做补测和组合消融）：
> 1. **改进组一 · DAG-fuse**（shared↔conditional 单向融合）—— ✅ 已实现 + 验证 + **已 push**（`ProMoE_TC_B_dagfuse`，3 臂）
> 2. **改进组二 · lbcontra**（路由对比损失负载均衡）—— ✅ 已实现 + 验证 + **已 push**（`ProMoE_TC_B_lbcontra`，13 run）
> 3. **改进组三 · adaptive-depth**（token 自适应跳过 / 加深 FFN，MoD 式）—— ✅ 已实现 + 验证 + **已 push**（`ProMoE_TC_B_adepth`，fixed_q，扫 depth_q ×4）
> 4. **改进组四 · lossfree**（无损路由负载均衡，DeepSeek arXiv 2408.15664）—— ✅ 已实现 + 验证（`ProMoE_TC_B_lossfree`，扫 u ×3）；未提交（本次新增）
> 5. **当前任务 · 结果补全与四点组合**——历史表缺失的参数正则 / adaptive-depth 结果先按同一 seed 和协议补齐；随后运行 `capacity_combo` 的受控组合消融（H、H+R、H+O、H+P、H+O+P、H+R+O、H+R+P、H+R+O+P）。
>
> 运行时 slot 按实验批次保存在对应的 `scripts/_run_times/<date>/` 目录；当前组合批次为 `2026_09_04`。
> **四组共用约定**：均在 base `ProMoE_TC`（`models/models_ProMoE_TC.py`：两步路由 + 静态 `cluster_centers` + top-1 token-choice + shared expert + 路由 InfoNCE 对比损失）上做**自包含变体**（`models_ProMoE_TC_<variant>.py` + config 开关）；**uncond token 一律不受影响**；尽量 **step-0 与 base 前向逐比特一致**；默认各自**独立消融**、不叠加。运行时 slot 按实验批次保存在对应的 `scripts/_run_times/<date>/` 目录。

---

# 改进组一：DAG-fuse — shared↔conditional 单向融合（DAG-MoE 风格）

> **状态：已实现 + 验证 + 已 push**（commit `d61c125`/`015a8c5`/`f733c35`）。架构采用方案 (b)：新建自包含 `models/models_ProMoE_TC_dagfuse.py` + `fusion_arm` 开关。
> 验证：`py_compile` ✅ / import ✅ / 四向一致性 ✅ / 输出目录碰撞守卫 ✅ / **step-0 与 base `ProMoE_TC` 前向逐比特一致（none + 三臂 max|Δ|=0.0）** ✅ / 三臂 query-set 语义 ✅ / uncond 不受影响 ✅ / **codex 独立审查 NO FINDINGS** ✅。真实训练 smoke 待在训练服务器上跑。
> 本版只做核心三臂、不加额外旋钮/改动（见本章末「本版明确不做」）。

## 0. 背景与来源

- 灵感论文：**DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts**（ICML 2026，arXiv 2606.01062）。核心：把 MoE 的"加权求和聚合"换成在被选专家间学一个 **DAG 做结构化聚合**；其轻量「DAG learning module」公式见论文 Eq 6–12，开源实现见 `github.com/JiaruiFeng/DAG-MoE` 的 `model/dag_moe/modeling_dagmoe.py::DAGLearningModule`。
- 关键经验（论文 Table 3）：**去门控的朴素 concat→MLP 混合反而比标准 MoE 更差**；起作用的是**门控/结构化**的形式。
- 基线：本仓库 `models/models_ProMoE_TC.py`（`SparseMoeBlock`），主线是 **top-1 路由**（`top_k=1`，`router_weight_mode=identity`），cond token 走原型路由、uncond token（label==1000）硬路由到 uncond 专家，shared 专家处理所有 token；当前块输出 = `routed加权和 (C) + shared (S)`。

## 1. 想法（与 DAG-MoE 的区别 = 创新点）

主线 top-1 时同一个 cond token 只有一个被选专家 → 在「被选专家之间」建 DAG 退化（K=1）。所以**不动 conditional 聚合**（动了也只是复刻 DAG-MoE、无新意），而是把 DAG-MoE 的「结构化聚合」迁移到**两个角色节点之间**：

- 节点 **C** = conditional/routed 专家聚合输出（**局部**信息）
- 节点 **S** = shared 专家输出（**全局**信息）

直觉：S 是全局/全集，C 是局部；局部应从全局拿参考来**补全自己**；而 S 本就是全集、未必需要回看 C。这只是假设，因此做**三臂方向性消融**。

## 2. 已锁定决策（decision log）

| 项 | 决定 |
|---|---|
| 主算子 | **忠实 K=2 DAG-MoE 边算子**（{C,S} 两节点）。FiLM（DiNeFu）**暂缓**。 |
| 跑哪些 | **3 个臂**：`cond_from_shared` / `shared_from_cond` / `bidirectional`。 |
| base 对照 | `ProMoE_TC_B`（`configs/004_ProMoE_B.yaml`）**已在另一台服务器跑过，不重跑**；三臂训练/采样超参**严格对齐它**以保证可比。 |
| timestep 条件化 | **本版不做**（`fusion_use_cond` 不实现）。理由：会把"方向"与"时间步门控"两个变量绑一起，破坏干净的方向对照；留作后续消融。 |
| 超参 | `d_g (fusion_dim)=64`、`L (fusion_num_iter)=1`、gate 激活 = **GELU(tanh)**（与本仓库 `MoeMLP` FFN 一致）。 |
| step-0 | up-proj 零初始化 ⇒ 块输出 = C+S，**与 base ProMoE_TC_B 严格等价**。 |
| 训练方式 | fresh（从零训练；不 warm-start base ckpt）。 |

## 3. 核心机制（精确规格）

对**仅 cond token**生效；uncond token 完全不受影响（保持原 `C+S`）。把每个 cond token 的两节点堆成 `X∈R^[N,2,d]`，`X[:,0]=C`、`X[:,1]=S`（N=batch 内 cond token 数，d=768）。

```
for l in range(L):                                  # L=1
    residual = X                                    # [N,2,d]
    Xn   = RMSNorm_l(X)                              # fp32 规约后 cast 回；[N,2,d]
    Xd   = down_l(Xn)                               # Linear d->d_g, no bias; [N,2,d_g]
    comb = combined_l(Xd)                           # Linear d_g->4*d_g, no bias; [N,2,4*d_g]
    g_src, g_tgt, v_src, v_tgt = comb.chunk(4, -1)  # 各 [N,2,d_g]
    # 因子化边：act(g_src[i]+g_tgt[j]) == σ(W_edge·concat(x_i,x_j)) 的 [W_s|W_t] 拆分（本实现 act=GELU-tanh）
    gate  = act(g_src.unsqueeze(-2) + g_tgt.unsqueeze(-3))   # [N,2(query i),2(key j),d_g]
    value =     v_src.unsqueeze(-2) + v_tgt.unsqueeze(-3)    # [N,2,2,d_g]
    msg   = gate * value                            # [N,2,2,d_g]
    agg   = msg.sum(dim=-2)                          # 对 key j 求和 -> [N,2,d_g]
    upd   = up_l(agg)                               # Linear d_g->d, no bias, 权重 ZERO-INIT; [N,2,d]
    X     = upd + residual                          # 模块内部残差 (Eq 11)
    # 臂限制（每次迭代都施加；key 集合恒为 {C,S}，含自环）：
    if   arm == "cond_from_shared":  X = stack([X[:,0],        residual[:,1]], 1)   # 只更新 C
    elif arm == "shared_from_cond":  X = stack([residual[:,0], X[:,1]       ], 1)   # 只更新 S
    # bidirectional: 保留 X（C、S 并行更新，各自读更新前的 {C,S}）
return X[:,0], X[:,1]                                # C_new, S_new
```

- **读出**（Eq 12, K=2）：cond token `out = C_new + S_new`；uncond token `out = C + S`（不进模块）。
- **三臂只差「更新哪个节点」(query 集合 Q)，key 集合恒为 {C,S}**，三臂参数完全相同 ⇒ 唯一实验变量是方向（iso-param 受控消融）。
- **act = GELU(tanh)**，与 `MoeMLP` 一致；**up-proj 权重 = 全 0**（仿 DAG-MoE `DAGLearningUpProjection`），down/combined 用默认初始化；RMSNorm.weight = 1。
- **dtype 流**：`_fuse_shared_cond` 把 cond 侧 C/S 先 `.float()`，故 **FusedRMSNorm 的规约与节点残差路径在 fp32**；但 **bf16 autocast 下 down/combined/up 的 matmul 与 `gate*value` 仍走 bf16**（autocast 覆盖输入 dtype），经残差 promote 回 fp32、末端 `.to(out_flat.dtype)` 写回。
- **step-0 证明**：up-proj=0 ⇒ 每次迭代 `upd=0` ⇒ `X=residual` 不变 ⇒ `C_new=C, S_new=S` ⇒ 块输出 = C+S，与 base 逐比特一致；base ckpt 可 `strict=False` 加载。
- **1/K 残差说明（重要纠正）**：DAG-MoE 的 `1/K·x`（Eq 6）是**块级**残差，作用是把层输入作为节点基底；在 ProMoE 里块级残差由 `DiTBlock`（`x = x + gate_mlp * x_mlp`）在 MoE 块**外部**处理，且这里节点是原始专家输出 C、S，**故 Eq 6 的 1/K 项结构上不适用、正确做法是省略**；只保留模块内部残差（Eq 11）。

**参数量**（d=768, d_g=64, L=1，每个 MoE 块）：RMSNorm 768 + down 49,152 + combined 16,384 + up 49,152 ≈ **0.115M/块**；DiT-B interleave → 6 个 MoE 块 → **≈0.69M**（约 B 模型的 0.5%）。三臂相同。
**计算开销**：每 cond token 两节点 ≈0.23M MAC，约为「routed MoeMLP + shared MoeMLP」的个位数 %，量级与论文 ~1.5%(L=1) 一致；只在 cond token（CFG 下约 90%）上付出。

## 4. 实现清单（file-by-file）

- [x] **模型**：新建 `models/models_ProMoE_TC_dagfuse.py`，作为 `models_ProMoE_TC.py` 的**自包含拷贝**（仿 `models_ProMoE_TC_anchor.py` / `_proto_choice.py` 模式）：
  - 新增 `FusedRMSNorm(d)`（fp32 规约，bf16-safe）与 `DAGFuseModule`（持 `L` 组 `down/combined/up` + `norms`，4 个 `nn.ModuleList`；**up-proj 列表命名 `fusion_up_projectors`** 以便 `utils.TrainingMonitor` 的 `*projectors` 规则自动挂梯度统计）。
  - `SparseMoeBlock.__init__` 读取 `fusion_arm`(默认 `"none"`)、`fusion_dim`(64)、`fusion_num_iter`(1)；`fusion_arm!="none"` 时建 `self.dag_fuse` 并 `assert use_shared_expert`。`cluster_centers/experts/shared_expert/_init_weights` 不变。
  - **插桩点**：`SparseMoeBlock.forward` 中，routed 循环算出 `final_output`(=C, [B,S,d]) 且 `shared_output=shared_expert(hidden_states)` 后，把 `final_output += shared_output`（约 195 行）替换为：展平 C、S 到 `[B*S,d]`；`out = C_flat + S_flat`；`cond_mask = (flat_labels != 1000)`；若 `fusion_arm!="none"` 且 cond 非空：gather `Cc,Sc` → `(Cc_new,Sc_new)=self.dag_fuse(Cc,Sc)` → `out_flat = out_flat.index_copy(0, cond_pos, fused)`（**非原地** `index_copy`）；reshape 回 `[B,S,d]`。**routing-contrastive 段与 `return final_output, loss` / `AddAuxiliaryLoss` 约定保持不变**；guard cond 数为 0 时跳过。**（已实现细节）** 该逻辑抽成 `_fuse_shared_cond` helper；cond 侧 `Cc,Sc` 先 `.float()`、`fused=(Cc_new+Sc_new).to(out_flat.dtype)` 回写；`none` 臂用**非原地** `final_output = final_output + shared_output`。
- [x] **注册**：`train.py` 加 `from models.models_ProMoE_TC_dagfuse import DiT as ProMoE_TC_dagfuse`，`model_dict["ProMoE_TC_B_dagfuse"] = (ProMoE_TC_dagfuse, "DiT_B_config")`（仿 anchor/proto_choice 模式；**实际落点** import `train.py:54`、model_dict 项 `train.py:89`）。`sample.py` 自动合并。
- [x] **配置**（从 `configs/004_ProMoE_B.yaml` 拷贝，超参严格对齐 base；只改 `model_name` 并在 `DiT_B_config.MoE_config` 加 `fusion_*`）：
  - `configs/004_ProMoE_B_dagfuse_condfromshared.yaml` → `fusion_arm: cond_from_shared`
  - `configs/004_ProMoE_B_dagfuse_sharedfromcond.yaml` → `fusion_arm: shared_from_cond`
  - `configs/004_ProMoE_B_dagfuse_bidirectional.yaml` → `fusion_arm: bidirectional`
  - 三者均 `model_name: "ProMoE_TC_B_dagfuse"`、`fusion_dim: 64`、`fusion_num_iter: 1`。
- [x] **脚本**（`scripts/template.sh` 模式，只改 `CONFIG`/`LOG`，训练入口 `train.py`）：`scripts/dagfuse/run_B_dagfuse_condfromshared_train_sample_eval.sh` 等 3 个。
- [x] **GPU slot**：对每个脚本 `scripts/_run_times/new_run.sh --script ... --gpus 4 --dry-run` 预览后写入（3 个 4-GPU slot，会自动 patch 各 YAML 的 `gpu_ids` 并生成 per-date wrapper）。

## 5. 验证

- [x] `python -m py_compile models/models_ProMoE_TC_dagfuse.py train.py`
- [x] `python scripts/check_output_dir.py --config <每个 yaml>`（强制输出目录碰撞守卫）
- [x] 四向一致性：`model_dict` ↔ `models/` ↔ `configs/` ↔ `scripts/`
- [x] **step-0 数值等价**：已验证 `fusion_arm="none"` **与三臂**在初始化时前向都与 base `ProMoE_TC_B` **逐比特一致（max|Δ|=0.0）**；另单测三臂 query-set 语义正确、uncond 不受影响。
- [ ] 短 smoke 训练（几十步不发散）；跑完清理 smoke 产物。**（唯一未完成项：本机无 GPU/promoe 环境，待训练服务器上跑。）**

## 5.5 交付物 / Delivered（改进组一实际产物）

- **模型**：`models/models_ProMoE_TC_dagfuse.py`（`FusedRMSNorm`、`DAGFuseModule`、`SparseMoeBlock._fuse_shared_cond`、`DiT.initialize_weights` 里 `_basic_init` 后重置 up-proj 零初始化）。
- **注册**：`train.py:54` import、`train.py:89` model_dict `"ProMoE_TC_B_dagfuse"`。
- **配置（3）**：`configs/004_ProMoE_B_dagfuse_{condfromshared,sharedfromcond,bidirectional}.yaml`。
- **脚本（3）**：`scripts/dagfuse/run_B_dagfuse_{condfromshared,sharedfromcond,bidirectional}_train_sample_eval.sh`。
- **Run-time slot（3，现于 `scripts/_run_times/2026_07_01/`；原 06_30 已并入今日 + 重编号以避开 lbcontra 撞号）+ 各自 `*-describe.txt`**：

  | slot | 臂 fusion_arm | gpu_ids | wrapper |
  |---|---|---|---|
  | 7.2 | cond_from_shared | [4,5,6,7] | `7.2-B_dagfuse_condfromshared.sh` |
  | 8.1 | shared_from_cond | [0,1,2,3] | `8.1-B_dagfuse_sharedfromcond.sh` |
  | 8.2 | bidirectional | [4,5,6,7] | `8.2-B_dagfuse_bidirectional.sh` |

- **启动（训练服务器上、tmux 内）**：
  ```bash
  tmux new-window -n dagfuse_cfs 'bash scripts/_run_times/2026_07_01/7.2-B_dagfuse_condfromshared.sh'
  tmux new-window -n dagfuse_sfc 'bash scripts/_run_times/2026_07_01/8.1-B_dagfuse_sharedfromcond.sh'
  tmux new-window -n dagfuse_bi  'bash scripts/_run_times/2026_07_01/8.2-B_dagfuse_bidirectional.sh'
  ```

## 6. 本版明确不做（推迟，仅记录，勿实现）

> 用户指示「先不加这些额外的改动」。以下都不写进代码/配置，留待三臂出信号后再单独评估：

- `fusion_operator=film`（AdaLN-Zero FiLM / DiNeFu 备选算子）。
- `fusion_use_cond`（把 `c=t_emb+y_emb` 注入融合门的 timestep 条件化，需一行 `DiTBlock` 改动）。
- `fusion_edge_tie`（`combined_proj` 4·d_g→2·d_g 的门控残差对照）。
- `fusion_gate_act=sigmoid` / `fusion_update_clamp`（bf16 逃生阀）。
- `fusion_detach_key`（方向纯度消融）。
- 任何额外的 base/对照配置（base 已在别处跑过）。

## 7. 风险提示（实现时留意）

- bf16：`gate*value` 是无界积，靠 RMSNorm 限输入 + up-proj 零初始化 + `max_grad_norm=0.5` 兜底；本算子**无 cos-sim/normalize+bmm**，不属于已记录的那类 loss 尖峰崩溃。头 ~10k 步看 `monitor/grad/fusion*`。（注：残差/RMSNorm 走 fp32，但投影与 `gate*value` 在 autocast 下仍 bf16，故此注意成立。）
- ckpt resume：新 `fusion_*` 参数是额外 key，加载 base 须 `strict=False`（本版 fresh 训练，主要影响后续若要 warm-start）。
- 公平性：三臂配置除 `fusion_arm` 外必须与 `configs/004_ProMoE_B.yaml` 完全一致（lr/batch/steps/data/sample 设置）。


---

# 改进组二：对比路由的负载均衡（Load-balance-aware routing contrastive loss）

> （改进组一 = 上文 DAG-fuse，已实现并验证。）本组**状态：已实现并验证（13-run 脚手架就绪 = 12 sweep + 1 `soft_only` 归因对照）；未提交、未训练**。实现 = 自包含 `models/models_ProMoE_TC_lbcontra.py`（**无新参数**，只改 `compute_routing_contrastive_loss`；`lb_contra_mode ∈ {none, soft_only, reweight, logit_adjust, balance_term}`）+ 注册 `train.py:55/91` + 13 config/script/slot（`scripts/_run_times/2026_07_01/` slot 1.1–7.1）+ 13 describe.txt。`soft_only`（软均值全 K InfoNCE、无调制，= `balance_term` 且 λ=0）作对照，把"硬→软均值"与"均衡调制"两个变量拆开归因。验证：前向与 base 逐比特一致（无新参、strict 兼容）、`none`==base 硬均值损失精确相等、三软模式可微且梯度到 `cluster_centers`、`balance_term` 均匀处最小、**codex NO FINDINGS**。锁定决策：count=per-GPU 局部/per-batch、纳入全部 K 簇（软分配均值）、**τ_route 首轮固定 0.07**（`lb_route_tau` 是 config 旋钮；观察：init 时软 InfoNCE 近乎 0、a/b 早期信号弱 → 若 a/b 明显弱于 c 再单独扫 τ_route）。目标：把 `models/models_ProMoE_TC.py` 的路由对比损失从"只管路由准确"升级为"同时考虑负载均衡"。

## 背景与问题

现状 `SparseMoeBlock.compute_routing_contrastive_loss`：对每个**非空** prototype i，取分到它的 cond token embedding 的**均值** `mean_i`；设 M = 非空簇数（≤ K=12），构 M×M 相似度矩阵 `sim[i,j]=cos(center_i, mean_j)`，`labels=arange`，逐行 `cross_entropy`（对角为正样本），对 M 行**等权平均**。这是个 InfoNCE，只驱动"每个 center 与自己簇均值最像" = **路由准确**。

- **问题1（均值丢 count）**：分到 1 个 vs 1000 个 token 的簇都只贡献一个 `mean` 向量、占一行、等权 —— loss 对"谁拿了多少 token"完全无感。
- **问题2（对负载零约束）**：该辅助损失不含任何均衡项，且配置 `load_balance_loss_coef=0`，没有其它均衡来源。
- 空簇当前被直接跳过、零惩罚 —— "欠载/没人选"这一头也没人管。

## 方向陷阱（实现时必须守住）

直接"给过载行的 CE **加大**权重"会**适得其反**：上调过载簇 CE → 其 `center_i` 被更强地拉向那一大堆 token 的均值、并与他簇推得更开 → 该簇更专一/更有吸引力 → 下一步抓更多 token → **更过载**。所以"惩罚过载"必须落在**降低过载簇吸引力 / 抬高欠载簇牵引**的方向上。
另：硬路由 `topk` 不可导，对比损失并不直接改"这一步的分配"，而是**塑造 prototype 与 token embedding**，跨步间接影响后续的 cos-sim 路由 —— 故方向必须对，否则朝错误方向塑造 prototype。

## 三个消融臂（用户认可，均作为 ablation 尝试；`n_i` = 分到 prototype i 的 cond token 数，`N=Σ n_i`；**默认 top_k=1（base 配置）**——top_k>1 时现有 `.any(dim=1)` 会把一个 token 计进多个簇、`Σn_i≠N_cond`，需改成 dispatch 分数）

- **臂 a — count 反比重加权 CE**：`loss = Σ_i w_i · CE_i`，`w_i` 随 `n_i` **递减**（欠载簇权重大、过载簇权重小，如 `w_i ∝ 1/n_i` 归一化），让欠载簇 center 获得更多牵引去"抢" token。注意方向：是**欠载加权**而非过载加权（见上"方向陷阱"）。
- **臂 b — balanced-softmax / logit 调整**：把 `n_i` 当"频率偏置"注入 CE —— 在候选列 j 上 **加 `log(n_j)`**（balanced-softmax 方向；**是加不是减**——减 `log(n)` 会落进上面的"方向陷阱"），减小对过载簇的对角特化压力。最贴合"把 count 融入 cross entropy"的表述。**（注意：`n_j` 并非严格类别频率，见"实现细节"caveat。）**
- **臂 c — 额外 count 驱动均衡项**：在 contrastive loss 上**加**一个显式均衡惩罚，如 Herfindahl `Σ (n_i/N)²`（均匀时最小）或 Switch 式 `Σ f_i·P_i`。**⚠ Switch 式基线里已实现**：`models_ProMoE_TC.py:135-155` 的 `load_balance_loss=(Pi·fi).sum()·α` 正是 `Σ f_i·P_i`，只是 `α=load_balance_loss_coef=0` 关着 → 臂 c 很大程度 = **重开/调这条现成损失**（唯 Herfindahl 才算真新增项）。

## 待定（设计阶段拍板）

- 三臂的精确公式：`w_i` 的具体形式与归一化；logit 偏置的确切位置（对角 vs 列）与系数；均衡项的形式与权重。
- 空簇/欠载处理 + count 数值稳定（`1/n_i`、`log n_i` 加 eps）+ count 统计口径 → 统一见下"实现细节·横切"一节（此处不重复）。
- 与改进组一（dagfuse）是否正交、可否叠加 —— 默认先各自独立消融。
- step-0 行为：本组只改辅助 loss 的形式（不改前向路径），通常 step-0 前向不变；需确认不引入早期不稳定。
- 实现方式：沿用改进组一的架构决策（自包含变体文件 + 开关）还是配置驱动 flag —— 留到设计阶段定。

## 实现细节（关键设计点，推荐默认 + 待确认）

**统一判断（可微性在三方案里性质不同）**：区别在于 count 是"被求导的对象"还是"detach 的调制量"。
- **a / b：count 只当 detached 的权重 / 偏置，不需要 STE**。CE 对 `sim_matrix`（→ centers、token means）本就可导，`n_i` 只是常数权重（a）/常数 logit 偏置（b），梯度照常从 cos-sim 流回 prototype 与 embedding。
- **c：count 本身进了 loss（`Σ(n_i/N)²`），才真正不可微**（`n_i` 来自 `topk` 硬 argmax，整数常数，梯度断）。

**方案 c —— 可微化（STE 往往不必要）**：
- **推荐默认 = Switch 式**（`aux = α·K·Σ_i f_i·P_i` 的思路）：定义每 token 软分配 `q_t = softmax(cos(token_t, centers)/τ_route)`（K 维），`P_i = mean_t q_t[i]`（可导），`f_i = n_i/N`（detach）；均衡项 `= K·Σ_i f_i·P_i`，梯度只走 `P_i`，STE-free。
- **备选 = 软 Herfindahl**：软 count `ñ_i = Σ_t q_t[i]`，均衡项 `= Σ_i (ñ_i/N)²`，同样可导、贴合原始 `Σ(n_i/N)²`。
- **fallback = STE**（硬前向 + 软反向），非首选。
- 注意：router 现为 `router_weight_mode: identity`（裸 cos-sim），软分配需额外定义 `softmax(cos_sim/τ_route)`，多一个 `τ_route`（可复用 `routing_contrastive_temperature` 或单设）。
- **⚠ 已存在实现**：Switch 式 `Σ f_i·P_i` = `models_ProMoE_TC.py:135-155` 的 `load_balance_loss`（`α=load_balance_loss_coef=0` 关着），且它对 identity 路由已用 `scores_for_aux=F.softmax(cond_weights)`（τ=1）算 `Pi`——**这正好回答 `τ_route` 从哪来**（复用它，别另造）。臂 c 首选 = "复用/打开这条现成损失"而非从零写。
- **⚠ 梯度冲突**：均衡项把 `P_i` 推向均匀（去特化），InfoNCE 同时把同一组 `cluster_centers/embedding` 推向高准确度（锐化）——**同一批参数上方向相反**，需给均衡项一个小相对权重并观察是否打架。
- **⚠ proxy 失配**：真实路由是 identity 硬 top-k（不过 softmax），`P_i=softmax(cos/τ_route)` 只是软代理；正则一个路由实际不用的分布未必真能均衡硬分配 —— 尽量让 `f_i/P_i` 取自与 `load_balance_loss` 相同的 `scores_for_aux`。

**方案 b —— 偏置超参**：
- **形式**：候选列 j 上 **加** `logits[i,j] += τ_adj·log(n_j+ε)`（logit-adjustment / balanced-softmax 方向）。
- **⚠ 类别错配 caveat**：本 InfoNCE 是**双射匹配**（M×M、`labels=arange`，每列恰是某一行的唯一正样本）→ CE 的标签先验**本就均匀**；`n_j` 是"簇大小/特征可靠度"、**并非类别频率 `P(y=j)`**。故 Ren-2020 的"免超参 Bayes 一致"保证**不能直接套用**——这里是把 count 当经验偏置，`τ_adj` 应视为**要扫的超参**（`{0.5,1,2}`），而非"恒 1 免调"。
- **⚠ 温度耦合**：喂给 CE 的是 `sim/τ`（`τ=0.07`）→ `sim∈[-1,1]` 放大到 `logits∈[±14]`，而 `log(n_j)` 只有几个单位 → `+log(n_j)` 加在 `/τ` **之后**是被 τ 稀释的弱修正。要么把偏置加在 `sim` 上（`/τ` **之前**），要么用 `τ_adj` 显式调强度。
- **⚠ 方向只是"可能对"**：抬高高频列 logit → 对角特化压力变小（利均衡）；**但**同时放大其它 center 对该高频列 `mean_j` 的**离对角排斥**（反均衡的抵消项）。净效果**存疑、需实测**，勿当"已证明正确"。
- **稳定**：`log(n_j + ε)` 或 `log(n_j + 1)`，防空簇 `n_j=0` 爆 `-inf`。

**方案 a —— `w_i` 形式**：`w_i ∝ 1/n_i`（或更柔和的 `1/√n_i`、`1/log(1+n_i)`），detach，且**归一化**（如 `Σ w_i = M`，保总损失尺度不变，否则偷偷改变有效 lr）。方向是**欠载加权**（见"方向陷阱"），STE-free。

**三方案共用的横切细节（这些定了才能公平消融）**：
- **count 统计口径**：`n_i` 是 per-batch、DDP 下为 **per-GPU 局部**。待拍：是否 `all_reduce` 成**全局** count；是否对 `n_i` 做 **EMA** 平滑（per-batch count 抖动大，structured-batch 时尤甚）。
- **空簇 / 欠载这一头**：现状 `compute_routing_contrastive_loss` **跳过空簇、零牵引**；只压过载则只治一头。a/b 要覆盖需把全部 K 簇纳入（不跳过空簇 + ε 平滑）；c 用软 count 天然覆盖。待拍：是否取消"跳过空簇"。
- **数值稳定**：`1/n_i`、`log(n_i)` 均需 eps / 平滑（`n_i=0`）。
- **均衡强度 / 量级**：c 是"加一项"，需相对 `routing_contrastive_lam=1` 的权重；a/b 强度藏在 `w_i` 尺度 / `τ_adj`。三者都要确认与现有对比损失的相对量级。

**待你确认的默认**：a/b 不用 STE（count 当 detached 调制）；c 默认软 count / Switch 式，STE 仅 fallback；b 用 balanced-softmax 免超参，需旋钮时退 logit-adjustment 扫 `τ_adj`。`count` 口径（局部/全局/EMA）与是否纳入空簇，留设计阶段先定并对全部臂统一。


---

# 改进组三：自适应 FFN 深度 —— 按 token 难度跳过/加深（token-adaptive FFN depth，MoD 式）

> 本组**状态：fixed_q v1 已实现 + 验证；未提交、未训练**。实现 = 自包含 `models/models_ProMoE_TC_adepth.py`（`SparseMoeBlock` + `depth_gate=Linear(d,1)` + `deepen_gain` 零初始化 + step buffer）+ 注册 `train.py:56/93` + 4 config（扫 `depth_q∈{0.1,0.2,0.3,0.4}`）+ 4 script/slot（`2026_07_01/` 9.1/9.2/10.1/10.2）+ 4 describe.txt。验证：**step-0 前向逐比特 = base（max|Δ|=0.0）**、**算力守恒（routed rows 256==256）**、梯度到 depth_gate/deepen_gain/experts/cluster_centers、q>0 生效、**codex NO FINDINGS**；自测抓到并修了"s 全等时 topk 重叠→不守恒"的真 bug（改用单次 argsort disjoint 切片）。**dynamic_reg = v2 未做**（待 count-tying 防坍缩修正）。目标：每个 block 对每个 cond token 用一个轻量 **linear 门控**判断它"该走几次 FFN"——简单 token **跳过**、难 token **加深（走 2 次）**，把跳过省下的算力**再分配**给难 token，**总算力守恒**（方案 b）。

## 1. 背景与思想

- 定位：**Mixture-of-Depths（MoD，DeepMind 2024）** 特化到 ProMoE 的 **cond-token routed-expert FFN** 路径；亦近 adaptive computation / early-exit / token pruning。
- 每个 block、每个 cond token：linear 门控输出**单个"难度分" `s_t`**；按 `s_t` 把 cond token 分三档（**方案 (b)，算力守恒**）：
  - **skip**（最易）→ 0 次 FFN，走残差；
  - **normal**（中间）→ 1 次（照常）；
  - **deepen**（最难）→ 2 次（第二次对**更新后的表示**再走**同一个 top-1 routed expert**）。
- 直觉：简单 token 不需要那么多 FFN，省下的算力给难 token。

## 2. 锁定决策（已与用户确认）

| 项 | 决定 |
|---|---|
| 额外 FFN 形态 | **加深**（第二次走同一 top-1 expert），**不加宽**（避免 top-K>1 路由大改） |
| shared expert | **不动**（照常处理所有 token） |
| uncond token | **不受影响** |
| 守恒 | **`#deepen == #skip == k`**，每步耦合 → `k·0 + (1−2k/N)·1 + k·2 = N` 恒等 |
| step-0 | init **人人恰好 1 次 FFN**（deepen 第二次**零初始化** + 初始不跳）→ 与 base 前向逐比特一致 |

## 3. 分配机制（关键：logits 只用来「排序」，不是 per-token argmax）

**门控输出单个标量 `s_t`**（不是 3 类 argmax——per-token argmax 会让 skip/deepen 数量失控、破坏守恒）。按 `s_t` 在 block 的 cond-token 池里**排序**：top-k → deepen，bottom-k → skip，中间 → normal。**`s_t` 决定"谁"，`k` 决定"多少个"。** 两种 `alloc_mode`（config 开关）：

- **`fixed_q`（稳，v1 兜底）**：固定配额 `q`，top-q/bottom-q。`q·0+(1−2q)·1+q·2=N` 按构造守恒。`q∈(0,0.5)`，`q=0` 退回 base。GPU 友好（静态形状）。
- **`dynamic_reg`（贴用户直觉，v2）**：`k` **动态 = 门控预测的跳过数**，deepen 耦合到同一个 `k`。⚠ **无约束会坍缩**（常见 `k→0` 退回 base：跳任何 token 都略伤 MSE，而"腾算力"的正向梯度信号很弱）→ **必须加软"目标率"正则**（惩罚 `|平均 skip 率 − 目标|`，或 Switch 式占用辅助 loss）锚住 `k`；目标率可从 0 退火上升（顺便满足 step-0）。变长 dispatch（像 MoE 变负载），GPU 可做但不如 `fixed_q` 高效。

## 4. 门控如何学 + step-0

- `s_t` **端到端学出来**（不是预先标注难度）：deepen 第二次 FFN 贡献用 `sigmoid(s_t)` 门控加权，梯度回流——**加深确有用则分被推高**，排序自然把有用的送 top-k（MoD 机制）。top-k/阈值不可导 → 用门控权重乘贡献使可导，必要时加辅助 loss。
- **step-0 逐比特一致**：deepen 第二次 FFN **零初始化**（初始贡献=0）+ 初始 `q`（或目标率）=0（不跳）→ 人人恰好 1 次 FFN = base；之后 `q`/门控渐学，才真正"跳易 / 加难"。

## 5. 待定（设计阶段拍板）

- **排序池**：逐图（每图内排 cond token）vs 批展平（EC-BC 式）—— 倾向逐图更稳。
- `fixed_q` 的 `q` 初值（如 0.25）；`dynamic_reg` 的目标率与退火曲线。
- deepen 第二次是"同一 expert 权重原样再走一遍"（纯加深、零新参数）还是带一个独立小投影（零初始化）；两者 step-0 都要求初始贡献=0。
- 架构 = 自包含变体文件 `models/models_ProMoE_TC_<name>.py` + `alloc_mode`/`q`/`target_rate` 等开关（同前两组约定）；是否与组一/组二叠加 —— 默认先独立。
- 可训练性 & 稳定性细节（软门控 vs straight-through、正则权重）留实现时定。


---

# 改进组四：无损路由负载均衡（Loss-Free Balancing，DeepSeek arXiv 2408.15664）

> 本组**状态：已实现 + 验证；未提交、未训练**。实现 = 自包含 `models/models_ProMoE_TC_lossfree.py`（**无新可训练参数**，只改 `compute_router` + 一个 `expert_bias` buffer）+ 注册 `train.py:57/95` + 3 config（扫 `bias_update_rate u∈{1e-4,1e-3,1e-2}`）+ 3 script/slot（`2026_07_01/` 11.1/11.2/12.1）+ 3 describe.txt。

## 1. 背景与思想
- 论文核心：为均衡用**辅助损失**（Switch 式）会注入**干扰梯度**、伤性能（"小 α 不均衡 / 大 α 伤性能"两难）。**Loss-Free** 不用辅助损失：给每个专家一个**偏置 `b_i`**，加到门控分上、**只用于 top-K 选择**；输出权重用**无偏置分** → **零干扰梯度**。
- `b_i` 每 batch 后按负载**非梯度更新**：`b_i += u·sign(c̄ − c_i)`（欠载抬、过载压）。论文消融：**sign > 幅度**、**加法 > 乘法**、`u=1e-3` 最佳、用历史负载避因果泄漏、sigmoid 门 > softmax。效果 MaxVio 0.5~0.72 → ~0.04，且 ppl 更低。

## 2. 锁定决策（已与用户确认）
| 项 | 决定 |
|---|---|
| 偏置作用 | 加在 cond 的 **identity cos-sim 选择分**上，**只影响 top-1 选择**；输出权重用无偏置 cos-sim |
| 更新规则 | **仅 sign**（论文默认）；加法偏置 |
| count 口径 | **全局 all_reduce**（跨卡一致 + 全局负载；无条件调用避 DDP 死锁） |
| u 扫描 | `{1e-4, 1e-3, 1e-2}` ×3 run |
| 对比损失 | **保留不变**（均衡与"路由准确"正交，这是相对改进组二的干净之处） |
| uncond / step-0 | uncond 不受影响；`b_i` buffer 初始 0 ⇒ step-0 前向逐比特 = base，**零新可训练参数** |

## 3. 验证（已过）
- **step-0 前向 = base 逐比特一致**（`use_lossfree_bias` False/True，eval，max|Δ|=0.0）；额外 key 仅 `expert_bias`。
- **零干扰梯度**：`expert_bias.requires_grad=False`、`.grad=None`；主梯度到 cluster_centers/experts/input 正常。
- **偏置更新方向**：过载 → 降、欠载 → 升。
- **均衡有效**：MaxVio 0.219 →（u=1e-3~1e-2）0.125，明显低于固定无偏置基线；u 过大（0.05/0.2）后期震荡/坍缩（复现论文结论）。

## 4. 待定 / 后续
- 若某个 u 明显更好，可细扫 u。
- 与改进组二（lbcontra）**互斥/对比**：两者都解"负载均衡"——组二用辅助 loss（有干扰梯度），组四用无损偏置（零干扰）；跑完可直接对比谁的均衡/FID 更好。
- 是否要"偏置作用在 softmax 分"的变体（论文说更难调）、或 EMA 平滑 count —— 暂不做。

---

# 四个有效点的组合实验：heterogeneous experts + routing/expert separation

> 这是下一轮组合实验的设计记录。核心消融严格复用已经有收益的机制：异构专家、历史 token-count diagonal LS-Reg、历史输出正则，以及参数正则在异构宽度下的明确几何扩展。`capacity` 版本的责任均衡不是历史结果，单独作为后续假设，不能混入核心结论。当前 4--7 号卡上的 Loss-Free 和缺失结果补测完成前，不启动本组训练。

## 1. 可检验的假设

1. **异构容量是否本身带来收益。** routed experts 的中间宽度按 1x--3x 排列，并让总 routed 中间宽度尽量贴近 Base（历史整数取整后只差 5 个 hidden units），先排除“总 FLOPs 变多”的解释。
2. **路由均衡与专家解耦是否互补。** 历史 LS-Reg 只改变 routing InfoNCE 的 diagonal；输出和参数正则分别约束函数行为与参数几何。组合是否有额外收益必须由逐项消融回答。
3. **不同宽度不能直接做参数向量拼接。** 异构专家的参数长度不同，参数臂使用固定长度、按宽度归一化的参数签名；这是一项明确的实现扩展，不冒充历史等宽 raw-parameter 实验。
4. **输出与参数视角是否提供互补信息。** `H+O+P` 和 `H+R+O+P` 分别检验不带/带路由均衡时两种解耦信号是否叠加。

## 2. 逐步消融（全部 training from scratch）

现有历史结果提供单点参考；新一轮增加必要的 H-anchored 组合臂，所有臂使用同一 `global_seed=0`、全局 batch 256、相同学习率/数据/采样协议，并训练到 300K 和 500K。历史单点 R/O/P 结果只作先验，不能替代本批次的 H+O、H+P、H+O+P 对照。

| 臂 | 异构宽度 | 责任感知 LS-Reg | 输出签名 | 参数签名 | 目的 |
|---|---:|---:|---:|---:|---|
| H | 是 | 否 | 否 | 否 | 异构专家本身的对照 |
| H-R | 是 | 是（历史 token-count diagonal） | 否 | 否 | 检验路由均衡在异构专家上的增益 |
| H-O | 是 | 否 | 是 | 否 | 检验输出正则在没有 R 时是否仍有效 |
| H-P | 是 | 否 | 否 | 是 | 检验参数正则在没有 R 时是否仍有效 |
| H-O-P | 是 | 否 | 是 | 是 | 检验两种专家解耦视角在没有 R 时是否互补 |
| H-R-O | 是 | 是 | 是 | 否 | 再加入专家函数输出解耦 |
| H-R-P | 是 | 是 | 否 | 是 | 再加入宽度无关的参数解耦 |
| H-R-OP | 是 | 是 | 是 | 是 | 完整组合；输出/参数两视角共享同一专家身份损失 |

`H` 采用与历史 `hetero_expert_NoPenalty` 相同的 1x--3x 容量排序和整数向下取整规则；B 规模下 routed experts 的中间宽度为 `[768, 907, 1047, 1186, 1326, 1466, 1605, 1745, 1885, 2024, 2164, 2304]`，总和为 18,427。它与历史实现使用同一宽度序列，因此这是**同一容量假设的等 routed-FLOPs 复现/组合基线**，不能把收益归因于额外计算量。O 使用历史有效输出正则的 L2 形式，并覆盖全部六个 MoE blocks；P 使用适用于不同宽度专家的固定长度参数几何签名，并在 block-3 复用历史 `exp(-L2/temperature)` 目标（temperature=0.7）。核心 R 臂使用历史 token-count diagonal LS-Reg。`capacity` responsibility 版本若要做，另开 `HR-capacity` 计划，不与本表的 R 混称。

## 3. 需要报告的证据

- 生成质量：300K/500K 的 FID 和 IS（CFG 1.0、1.5）。
- 路由：每个 block 的 token 负载、`n_i` 的变异系数、路由熵和 top-1 margin；若做后续 capacity 臂，再增加 `n_i*m_i` 的变异系数。
- 专家分工：输出 RMS、归一化输出两两距离、参数签名两两距离/相似度、有效秩；检查是否通过增大输出尺度投机。
- 计算公平性：总参数量、激活参数量、实际 FLOPs，以及不同专家的宽度和使用率。
- 稳定性：辅助 loss 的数值、非有限梯度、300K 到 500K 的趋势；任何早期 checkpoint 只作诊断，不作结论。

## 4. 预先的否决条件

- 若后续 `HR-capacity` 只降低 token 计数的 CV，却提高 `n_i*m_i` 的 CV，则说明容量目标定义错了，停止继续叠加 O/P。
- 若 O/P 使输出 RMS 明显上升而归一化距离没有改善，判定为尺度投机，不把 FID 的偶然变化算作收益。
- 只有完整组合同时改善生成指标和至少一个责任/分工指标，且逐项移除消融能解释增益来源，才把它作为期刊主线；否则保留为负结果。

## 5. 组合后的统一假设（补测和八臂矩阵完成后才执行）

四个点不能只被写成四个互不相关的开关。异构专家改变了每个专家一次前向的计算量，所以只看 token 数并不能说明负载是否真的均衡。下一阶段只保留一个可以被实验推翻的假设：

- 记专家 `i` 的中间宽度为 `m_i`，一个训练批次实际分到它的 token 数为 `n_i`。定义计算责任 `r_i = n_i * m_i`，再除以所有专家责任的平均值。
- `r_i > 1` 表示这个专家承担了过多计算，`r_i < 1` 表示承担不足。路由对比损失的均衡修正应使用这个量，而不是只把每个专家的 token 数拉到一样；修正量只作为训练期的 detached 信号，不改变推理时的额外路径。
- 输出正则和参数正则也使用同一份责任记录来校准强度，避免大专家因为参数更多而在正则里占主导。这样四个点共同回答一个问题：在固定总计算量下，如何把 token 分给不同容量的专家，并让每个专家学到不同的功能。

矩阵完成后，最多先开四个新臂：`H-Rcap`、`H-Rcap-O`、`H-Rcap-P`、`H-Rcap-O-P`。它们仍然 fresh、seed 0、300K/500K、同一评估协议；若 `r_i` 的变异系数没有下降，或生成质量与专家分工没有同步改善，就停止这条统一路线，不再继续堆叠正则。
