# 实验 8: MoS 多层对齐 + Prototype 路由相似度交叉对齐

## 实验目标

在 MoS naive choice 多层对齐的基础上，利用 conditional expert 的路由信息实现交叉对齐。**不引入新的可学习模块**，直接复用每个 MoE block 路由阶段的 prototype 余弦相似度作为交叉对齐权重。

## 基座模型

`models/models_ProMoE_TC_repa_MoS_naive_choice.py`（MoS naive choice，多层对齐）

## 核心思想

- 在每个 `align_blocks` 中的 MoE block 路由阶段，每个 cond token 与其被分配到的 prototype (cluster center) 计算余弦相似度。
- 对于被分到同一个 expert e 的 token a 和 b：
  - `w(a, a) = 1`（自身对齐权重固定为 1）
  - `w(a, b) = cos_sim(a, proto_e) × cos_sim(b, proto_e)`
- 不同 block 的路由独立，因此每个对齐 block 使用其对应 MoE block 的路由相似度。

## 新增模块

无新增可学习模块。

## 模型修改

### SparseMoeBlock

在 `compute_router` 中缓存路由信息（同实验 4，但需要注意 MoS 中每个 MoE block 独立缓存）：

```python
# 在 compute_router 末尾:
self._expert_indices = expert_indices  # (N, T, top_k)

# 缓存每个 token 与其 top-1 expert prototype 的原始余弦相似度
proto_sim = torch.zeros(batch_size * seq_len, device=device)
if cond_mask.any():
    top1_idx = topk_idx[:, 0]
    top1_sim = cos_sim[torch.arange(len(top1_idx)), top1_idx]
    proto_sim[cond_positions] = top1_sim
self._proto_sim = proto_sim.view(batch_size, seq_len)  # (N, T)
```

### DiT.__init__

无新增模块。保留 MoS 的所有原有组件（BlockRouter、mos_projectors 等）。

### DiT.forward

```python
def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
    # ... patchify, embedding, BlockRouter 预计算 ...

    mos_repa_loss = torch.tensor(0.0, device=x.device)
    for i, block in enumerate(self.blocks):
        x = block(x, c, labels)
        if self.training and routing_weights is not None and i in self.align_block_to_idx:
            align_idx = self.align_block_to_idx[i]

            if block.use_moe:
                expert_indices = block.mlp._expert_indices  # (N, T, top_k)
                proto_sim = block.mlp._proto_sim             # (N, T)
                cross_weights = self._build_proto_cross_weights(
                    expert_indices, proto_sim, labels, N, T
                )
            else:
                # dense block: 无路由信息，退化为标准 MoS（1-to-1 对齐）
                cross_weights = torch.eye(T, device=x.device).unsqueeze(0).expand(N, -1, -1)

            block_loss = self.compute_cross_mos_repa_loss(
                x, align_idx, routing_weights, teacher_all_z,
                expert_indices if block.use_moe else None,
                labels, cross_weights, N, T, D
            )
            mos_repa_loss = mos_repa_loss + block_loss

    # ... averaging, final_layer, unpatchify ...
```

### _build_proto_cross_weights

与实验 4 完全一致：

```python
def _build_proto_cross_weights(self, expert_indices, proto_sim, labels, N, T):
    """
    构建 prototype 余弦相似度交叉对齐权重矩阵。

    w(i, i) = 1
    w(i, j) = cos_sim(i, proto_e) * cos_sim(j, proto_e)  (同 expert e, i≠j)
    w(i, j) = 0  (不同 expert 或 unconditional)
    """
    top1_experts = expert_indices[:, :, 0]
    expert_match = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))
    cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)
    pair_cond = cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)

    outer_sim = proto_sim.unsqueeze(2) * proto_sim.unsqueeze(1)
    W = outer_sim * expert_match.float() * pair_cond.float()

    # 对角线强制为 1
    diag_mask = torch.eye(T, device=W.device).unsqueeze(0).expand(N, -1, -1)
    cond_diag = diag_mask * cond_mask.unsqueeze(2).float()
    W = W * (1 - diag_mask) + cond_diag

    return W
```

## 交叉对齐 MoS 损失计算

与实验 5、6、7 的 `compute_cross_mos_repa_loss` 一致。

## 配置参数

```yaml
DiT_B_config:
  repa_config:
    enc_type: "dinov2-vit-b"
    num_teacher_blocks: 12
    z_dims: [768]
    projector_dim: 2048
    align_blocks: [2, 3, 4]      # 0-indexed blocks 2,3,4
    mos_top_k: 2
    mos_random_prob: 0.05
    cross_align_type: "proto"
    # 无额外参数

repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

## 训练脚本

注册到 `train_with_MoS_repa.py`。

```python
"ProMoE_TC_REPA_MoS_CROSS_PROTO_B": (ProMoE_TC_REPA_MoS_CrossProto, "DiT_B_config"),
```

## 文件清单

| 类型 | 路径 |
|------|------|
| 模型 | `models/models_ProMoE_TC_repa_MoS_naive_choice_cross_proto.py` |
| 配置 | `configs/004_ProMoE_B_repa_MoS_naive_choice_cross_proto.yaml` |
| 脚本 | `scripts/MoS_repa/run_B_repa_mos_cross_proto_train_sample_eval.sh` |
| 训练入口 | `train_with_MoS_repa.py`（新增 model_dict 条目） |

## 注意事项

1. **无新增可学习参数**：8 个实验中最轻量的 MoS 变体，新增计算仅为 `_build_proto_cross_weights` 中的外积和掩码操作。
2. **每个 block 使用其自身的路由相似度**：不同 MoE block 的 `SparseMoeBlock` 独立路由，`_proto_sim` 也不同。同一对 token 在不同 block 可能被分到不同 expert，对应不同的交叉对齐权重。
3. **余弦相似度符号**：`router_weight_mode="identity"` 时，`cos_sim` 范围为 `[-1, 1]`。两个负值相乘变正值。建议确认使用 identity mode 时的行为是否符合预期。如需强制非负，可使用 `cos_sim.clamp(min=0)` 或 `(cos_sim + 1) / 2`。
4. **dense block fallback**：对于 `align_blocks` 中的 dense block（无路由信息），权重矩阵退化为单位矩阵（仅自身对齐），等价于标准 MoS REPA 的 1-to-1 对齐。
5. **梯度流**：`proto_sim` 来自路由阶段的 `cos_sim`，其梯度同时影响 `cluster_centers` 和输入 token embedding。交叉对齐损失通过 `proto_sim` 为 prototype learning 提供了额外信号，可能改善 expert specialization。
6. **与 routing contrastive loss 的关系**：routing contrastive loss 鼓励 cluster centers 与对应 token 均值方向一致；prototype 交叉对齐鼓励高相似度 token 互相对齐。两者通过 `cluster_centers` 梯度互相影响，可能有协同效果。
