# design-todo: ProMoE_TC_B_dagfuse — shared↔conditional 单向融合（DAG-MoE 风格）

> 状态：**已实现并通过验证（核心三臂 + 脚手架）；未提交、未训练**。架构采用方案 (b)：新建自包含 `models/models_ProMoE_TC_dagfuse.py` + `fusion_arm` 开关。
> 验证：`py_compile` ✅ / import ✅ / 四向一致性 ✅ / 输出目录碰撞守卫 ✅ / **step-0 与 base `ProMoE_TC` 前向逐比特一致（none + 三臂 max|Δ|=0.0）** ✅ / 三臂 query-set 语义 ✅ / uncond 不受影响 ✅ / **codex 独立审查 NO FINDINGS** ✅。真实训练 smoke 待在训练服务器上跑（本机无 promoe/GPU 环境，仅做了代码级前向等价验证）。
> 按用户要求**本版只做核心三臂、不加任何额外旋钮/改动**（见末尾「本版明确不做」）。

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
    # 因子化边：silu/gelu(g_src[i]+g_tgt[j]) == σ(W_edge·concat(x_i,x_j)) 的 [W_s|W_t] 拆分
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
- **step-0 证明**：up-proj=0 ⇒ 每次迭代 `upd=0` ⇒ `X=residual` 不变 ⇒ `C_new=C, S_new=S` ⇒ 块输出 = C+S，与 base 逐比特一致；base ckpt 可 `strict=False` 加载。
- **1/K 残差说明（重要纠正）**：DAG-MoE 的 `1/K·x`（Eq 6）是**块级**残差，作用是把层输入作为节点基底；在 ProMoE 里块级残差由 `DiTBlock`（`x = x + gate_mlp * x_mlp`）在 MoE 块**外部**处理，且这里节点是原始专家输出 C、S，**故 Eq 6 的 1/K 项结构上不适用、正确做法是省略**；只保留模块内部残差（Eq 11）。

**参数量**（d=768, d_g=64, L=1，每个 MoE 块）：RMSNorm 768 + down 49,152 + combined 16,384 + up 49,152 ≈ **0.115M/块**；DiT-B interleave → 6 个 MoE 块 → **≈0.69M**（约 B 模型的 0.5%）。三臂相同。
**计算开销**：每 cond token 两节点 ≈0.23M MAC，约为「routed MoeMLP + shared MoeMLP」的个位数 %，量级与论文 ~1.5%(L=1) 一致；只在 cond token（CFG 下约 90%）上付出。

## 4. 实现清单（file-by-file）

- [ ] **模型**：新建 `models/models_ProMoE_TC_dagfuse.py`，作为 `models_ProMoE_TC.py` 的**自包含拷贝**（仿 `models_ProMoE_TC_anchor.py` / `_proto_choice.py` 模式）：
  - 新增 `FusedRMSNorm(d)`（fp32 规约，bf16-safe）与 `DAGFuseModule`（持 `L` 组 `down/combined/up` + `norms`，4 个 `nn.ModuleList`；**up-proj 列表命名 `fusion_up_projectors`** 以便 `utils.TrainingMonitor` 的 `*projectors` 规则自动挂梯度统计）。
  - `SparseMoeBlock.__init__` 读取 `fusion_arm`(默认 `"none"`)、`fusion_dim`(64)、`fusion_num_iter`(1)；`fusion_arm!="none"` 时建 `self.dag_fuse` 并 `assert use_shared_expert`。`cluster_centers/experts/shared_expert/_init_weights` 不变。
  - **插桩点**：`SparseMoeBlock.forward` 中，routed 循环算出 `final_output`(=C, [B,S,d]) 且 `shared_output=shared_expert(hidden_states)` 后，把 `final_output += shared_output`（约 195 行）替换为：展平 C、S 到 `[B*S,d]`；`out = C_flat + S_flat`；`cond_mask = (flat_labels != 1000)`；若 `fusion_arm!="none"` 且 cond 非空：gather `Cc,Sc` → `(Cc_new,Sc_new)=self.dag_fuse(Cc,Sc)` → `out.index_copy_(0, cond_pos, Cc_new+Sc_new)`；reshape 回 `[B,S,d]` 作为 `final_output`。**routing-contrastive 段与 `return final_output, loss` / `AddAuxiliaryLoss` 约定保持不变**；guard cond 数为 0 时跳过。
- [ ] **注册**：`train.py` 加 `from models.models_ProMoE_TC_dagfuse import DiT as ProMoE_TC_dagfuse`，`model_dict["ProMoE_TC_B_dagfuse"] = (ProMoE_TC_dagfuse, "DiT_B_config")`（仿第 48–49、78–79 行）。`sample.py` 自动合并。
- [ ] **配置**（从 `configs/004_ProMoE_B.yaml` 拷贝，超参严格对齐 base；只改 `model_name` 并在 `DiT_B_config.MoE_config` 加 `fusion_*`）：
  - `configs/004_ProMoE_B_dagfuse_condfromshared.yaml` → `fusion_arm: cond_from_shared`
  - `configs/004_ProMoE_B_dagfuse_sharedfromcond.yaml` → `fusion_arm: shared_from_cond`
  - `configs/004_ProMoE_B_dagfuse_bidirectional.yaml` → `fusion_arm: bidirectional`
  - 三者均 `model_name: "ProMoE_TC_B_dagfuse"`、`fusion_dim: 64`、`fusion_num_iter: 1`。
- [ ] **脚本**（`scripts/template.sh` 模式，只改 `CONFIG`/`LOG`，训练入口 `train.py`）：`scripts/dagfuse/run_B_dagfuse_condfromshared_train_sample_eval.sh` 等 3 个。
- [ ] **GPU slot**：对每个脚本 `scripts/_run_times/new_run.sh --script ... --gpus 4 --dry-run` 预览后写入（3 个 4-GPU slot，会自动 patch 各 YAML 的 `gpu_ids` 并生成 per-date wrapper）。

## 5. 验证

- [ ] `python -m py_compile models/models_ProMoE_TC_dagfuse.py train.py`
- [ ] `python scripts/check_output_dir.py --config <每个 yaml>`（强制输出目录碰撞守卫）
- [ ] 四向一致性：`model_dict` ↔ `models/` ↔ `configs/` ↔ `scripts/`
- [ ] **step-0 数值等价**：`fusion_arm="none"` 的前向须与 base `ProMoE_TC_B` 逐比特一致；且任一臂在初始化时融合输出 == `C+S`。
- [ ] 短 smoke 训练（几十步不发散）；跑完清理 smoke 产物。

## 6. 本版明确不做（推迟，仅记录，勿实现）

> 用户指示「先不加这些额外的改动」。以下都不写进代码/配置，留待三臂出信号后再单独评估：

- `fusion_operator=film`（AdaLN-Zero FiLM / DiNeFu 备选算子）。
- `fusion_use_cond`（把 `c=t_emb+y_emb` 注入融合门的 timestep 条件化，需一行 `DiTBlock` 改动）。
- `fusion_edge_tie`（`combined_proj` 4·d_g→2·d_g 的门控残差对照）。
- `fusion_gate_act=sigmoid` / `fusion_update_clamp`（bf16 逃生阀）。
- `fusion_detach_key`（方向纯度消融）。
- 任何额外的 base/对照配置（base 已在别处跑过）。

## 7. 风险提示（实现时留意）

- bf16：`gate*value` 是无界积，靠 RMSNorm 限输入 + up-proj 零初始化 + `max_grad_norm=0.5` 兜底；本算子**无 cos-sim/normalize+bmm**，不属于已记录的那类 loss 尖峰崩溃。头 ~10k 步看 `monitor/grad/fusion*`。
- ckpt resume：新 `fusion_*` 参数是额外 key，加载 base 须 `strict=False`（本版 fresh 训练，主要影响后续若要 warm-start）。
- 公平性：三臂配置除 `fusion_arm` 外必须与 `configs/004_ProMoE_B.yaml` 完全一致（lr/batch/steps/data/sample 设置）。
