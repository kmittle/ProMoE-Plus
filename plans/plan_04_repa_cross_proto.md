# 实验 4: Naive REPA + Prototype 路由相似度交叉对齐

## 实验目标

在 naive REPA 单层对齐的基础上，利用 conditional expert 的路由信息实现交叉对齐。**不引入任何新的可学习模块**，直接复用路由阶段已有的 prototype 余弦相似度作为交叉对齐权重。

## 基座模型

`models/models_ProMoE_TC_repa.py`（naive REPA，单层对齐于 `encoder_depth`）

## 核心思想

- 在 prototypical routing 中，每个 cond token 与其被分配到的 prototype (cluster center) 计算余弦相似度。
- 对于被分到同一个 expert e 的 token a 和 b：
  - token a 与**自身** DINO token 的对齐权重固定为 **1**
  - token a 与 **token b 对应 DINO token** 的对齐权重为 `cos_sim(a, proto_e) × cos_sim(b, proto_e)`
- 直觉：两个 token 与 prototype 的相似度都高，说明它们在语义上高度相关，应该更强地互相对齐。

## 新增模块

无新增模块。仅需从 `SparseMoeBlock` 中提取路由阶段的余弦相似度。

## 模型修改

### SparseMoeBlock

在 `compute_router` 中缓存路由信息：

```python
def compute_router(self, hidden_states, labels):
    # ... 现有逻辑 ...
    if cond_mask.any():
        cond_positions = torch.where(cond_mask)[0]
        cond_input = flat_input[cond_positions]
        input_norm = F.normalize(cond_input, p=2, dim=1)
        cluster_norm = F.normalize(self.cluster_centers, p=2, dim=1)
        cos_sim = input_norm @ cluster_norm.T  # (num_cond, num_experts)
        # ... topk 选择 ...

    # 缓存路由信息
    self._expert_indices = expert_indices  # (N, T, top_k)

    # 缓存每个 token 与其 top-1 expert prototype 的余弦相似度
    # 需要构建完整的 (N*T,) 向量
    proto_sim = torch.zeros(batch_size * seq_len, device=device)
    if cond_mask.any():
        top1_idx = topk_idx[:, 0]  # (num_cond,)
        top1_sim = cos_sim[torch.arange(len(top1_idx)), top1_idx]  # (num_cond,)
        proto_sim[cond_positions] = top1_sim
    self._proto_sim = proto_sim.view(batch_size, seq_len)  # (N, T)

    return router_weights, expert_indices, load_balance_loss
```

### DiT.__init__

```python
# projector 保持不变，无新增模块
self.projectors = nn.ModuleList([
    build_repa_projector(hidden_size, projector_dim, z_dim) for z_dim in z_dims
])
```

### DiT.forward

```python
def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
    # ... patchify, embedding 同基座 ...
    cross_align_loss = torch.tensor(0.0, device=x.device)
    for i, block in enumerate(self.blocks):
        x = block(x, c, labels)
        if self.training and self.projectors is not None and (i + 1) == self.encoder_depth:
            if teacher_all_z is not None:
                expert_indices = block.mlp._expert_indices  # (N, T, top_k)
                proto_sim = block.mlp._proto_sim             # (N, T)

                # 构建 prototype 相似度权重矩阵
                cross_weights = self._build_proto_cross_weights(
                    expert_indices, proto_sim, labels, N, T
                )

                teacher_z = teacher_all_z[-1]  # teacher 最后一层
                z_proj = self.projectors[0](x.reshape(-1, D)).reshape(N, T, -1)
                cross_align_loss = self.compute_cross_align_loss(
                    z_proj, teacher_z, expert_indices, labels, cross_weights
                )

    x = self.final_layer(x, c)
    x = self.unpatchify(x)
    if not self.training:
        return x
    return x, cross_align_loss
```

## 交叉对齐权重构建

```python
def _build_proto_cross_weights(self, expert_indices, proto_sim, labels, N, T):
    """
    构建基于 prototype 余弦相似度的交叉对齐权重矩阵。

    规则:
    - W[n, i, i] = 1（自身对齐权重固定为 1）
    - W[n, i, j] = proto_sim[n, i] * proto_sim[n, j]
      （仅当 i, j 同 expert 且 both conditional）
    - 其余为 0

    Args:
        expert_indices: (N, T, top_k)
        proto_sim: (N, T) 每个 token 与其 top-1 expert prototype 的余弦相似度
        labels: (N,) class labels
        N, T: batch size, num tokens
    Returns:
        W: (N, T, T)
    """
    # 同 expert 掩码
    top1_experts = expert_indices[:, :, 0]  # (N, T)
    expert_match = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))  # (N, T, T)

    # conditional 掩码
    cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)  # (N, T)
    pair_cond = cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)  # (N, T, T)

    # 外积: sim_i * sim_j
    outer_sim = proto_sim.unsqueeze(2) * proto_sim.unsqueeze(1)  # (N, T, T)

    # 组合: 同 expert + conditional + 外积权重
    W = outer_sim * expert_match.float() * pair_cond.float()  # (N, T, T)

    # 对角线强制为 1（对 conditional token）
    diag_mask = torch.eye(T, device=W.device).unsqueeze(0).expand(N, -1, -1)
    cond_diag = diag_mask * cond_mask.unsqueeze(2).float()  # (N, T, T)
    W = W * (1 - diag_mask) + cond_diag  # 非对角线保持外积，对角线设为 1

    return W
```

## 交叉对齐损失计算

复用与实验 1-3 相同的 `compute_cross_align_loss`。由于 `_build_proto_cross_weights` 已经包含了 expert_match 和 cond 掩码，`compute_cross_align_loss` 中的掩码操作实际上是冗余的，但保留以保持一致性。

## 配置参数

```yaml
DiT_B_config:
  repa_config:
    enc_type: "dinov2-vit-b"
    encoder_depth: 4
    z_dims: [768]
    projector_dim: 2048
    cross_align_type: "proto"
    # 无额外参数

repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

## 训练脚本

同实验 1-3，注册到 `train_with_MoS_repa.py`。

```python
"ProMoE_TC_REPA_CROSS_PROTO_B": (ProMoE_TC_REPA_CrossProto, "DiT_B_config"),
```

## 文件清单

| 类型 | 路径 |
|------|------|
| 模型 | `models/models_ProMoE_TC_repa_cross_proto.py` |
| 配置 | `configs/004_ProMoE_B_repa_cross_proto.yaml` |
| 脚本 | `scripts/repa/run_B_repa_cross_proto_train_sample_eval.sh` |
| 训练入口 | `train_with_MoS_repa.py`（新增 model_dict 条目） |

## 注意事项

1. **无新增可学习参数**：这是 4 种策略中最轻量的，不引入任何额外模块，仅复用路由阶段已有的余弦相似度。训练开销基本与 naive REPA 一致。
2. **余弦相似度范围**：`cos_sim` 在 `router_weight_mode="identity"` 时范围为 `[-1, 1]`。两个负值相乘会变正值，需要注意。如果 `router_weight_mode="softmax"`，`cos_sim` 是 softmax 后的概率分布，外积值会很小。建议使用原始余弦相似度（identity mode）计算外积，而非 softmax 后的值。
3. **对角线处理**：自身对齐权重固定为 1（由 implementation plan 明确要求），不受 prototype 相似度影响。
4. **退化情况**：如果某 expert group 只有 1 个 token，权重矩阵在该 token 行仅有对角线为 1，完全等价于标准 1-to-1 REPA。
5. **梯度流**：prototype 相似度的梯度会通过 `_proto_sim` 流回 `cluster_centers` 和输入 token，这为 prototype 学习提供了额外的对齐信号。
6. **Teacher 特征差异**：同实验 1，使用 `extract_all_teacher_block_features()` 提取的 `teacher_all_z[-1]` 与原始 REPA 的 `extract_teacher_features()` 输出存在微小差异（缺少 final layer norm），projector 应能自适应。
