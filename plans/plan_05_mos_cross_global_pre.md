# 实验 5: MoS 多层对齐 + 全图 Pre-Block Attention 交叉对齐

## 实验目标

在 MoS naive choice 多层对齐的基础上，利用 conditional expert 的路由信息实现交叉对齐。对齐位置和 BlockRouter 机制不变（每个对齐 block 的输出处，BlockRouter 选择对齐哪些 teacher blocks），但每个 token 不再仅与自身位置的 teacher token 做 1-to-1 对齐，而是与同一张图、同一个 conditional expert 的 token 对应的 teacher token 交叉对齐，权重由**全图 pre-block attention map** 决定。

## 基座模型

`models/models_ProMoE_TC_repa_MoS_naive_choice.py`（MoS naive choice，多层对齐）

## 核心思想

- 在 patchify 之后、进入 DiT blocks 之前，用 **2 层可学习 transformer block** 预处理所有 token，然后 QKV + scaled dot-product → 全图 attention map `(N, T, T)`。
- 这个 attention map 一次计算，**所有 `align_blocks` 中的对齐 block 复用**。
- 在每个对齐 block 处，交叉对齐 loss 的计算结合 BlockRouter（选择 teacher block）和 attention map（选择 teacher token 位置）。

## 新增模块

### GlobalPreAttention

与实验 1 完全相同：

```python
class GlobalPreAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, num_blocks=2, mlp_ratio=4.0, qk_norm=False):
        # 2层 RouterTransformerBlock + QK 投影
    def forward(self, x):
        # → attn_map: (N, T, T)
```

## 模型修改

### SparseMoeBlock

缓存 `self._expert_indices`（同实验 1-4）。

### DiT.__init__

在 MoS 基座的初始化基础上，新增 `GlobalPreAttention`：

```python
# 新增（在 repa_config 解析区域，与 BlockRouter 同级）:
self.global_pre_attn = GlobalPreAttention(
    hidden_size=hidden_size,
    num_heads=repa_config.get('align_attn_num_heads', 8),
    num_blocks=2,
    mlp_ratio=repa_config.get('align_attn_mlp_ratio', 4.0),
    qk_norm=qk_norm,
)

# 保留 MoS 的所有原有组件:
# self.block_router (BlockRouter), self.mos_projectors, self.align_blocks 等
```

### DiT.forward

```python
def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
    y = context
    if len(x.shape) != 4:
        x = x.squeeze(2)

    x = self.x_embedder(x) + self.pos_embed
    N, T, D = x.shape
    t = self.t_embedder(timestep)
    y, labels = self.y_embedder(y, self.training)
    c = t + y

    # 1. BlockRouter 预计算路由权重（与 MoS 基座相同）
    routing_weights = None
    if self.training and self.block_router is not None and teacher_all_z is not None:
        routing_weights = self.block_router(x, c)  # (N, T, m, n_align)

    # 2. 计算全图 pre-block attention map
    cross_attn_map = None
    if self.training and self.global_pre_attn is not None and teacher_all_z is not None:
        cross_attn_map = self.global_pre_attn(x)  # (N, T, T)

    # 3. 过 DiT blocks
    mos_repa_loss = torch.tensor(0.0, device=x.device)
    for i, block in enumerate(self.blocks):
        x = block(x, c, labels)
        if self.training and routing_weights is not None and i in self.align_block_to_idx:
            align_idx = self.align_block_to_idx[i]
            # 获取该 block 的路由信息
            expert_indices = block.mlp._expert_indices if block.use_moe else None
            block_loss = self.compute_cross_mos_repa_loss(
                x, align_idx, routing_weights, teacher_all_z,
                expert_indices, labels, cross_attn_map, N, T, D
            )
            mos_repa_loss = mos_repa_loss + block_loss

    if self.training and routing_weights is not None and len(self.align_blocks) > 0:
        mos_repa_loss = mos_repa_loss / len(self.align_blocks)

    x = self.final_layer(x, c)
    x = self.unpatchify(x)
    if not self.training:
        return x
    return x, mos_repa_loss
```

## 交叉对齐 MoS 损失计算

替换基座的 `compute_mos_repa_loss`：

```python
def compute_cross_mos_repa_loss(self, x, align_idx, routing_weights, teacher_all_z,
                                 expert_indices, labels, cross_weights, N, T, D):
    """
    结合 BlockRouter（选 teacher block）+ 交叉对齐（选 teacher token 位置）。

    Args:
        x: (N, T, D) 当前 block 输出
        align_idx: projector/router 列索引
        routing_weights: (N, T, m, n_align) BlockRouter 输出
        teacher_all_z: (m, N, T, D_z) 所有 teacher block 特征
        expert_indices: (N, T, top_k) 该 block 的路由分配（MoE block 时有效）
        labels: (N,) class labels
        cross_weights: (N, T, T) 交叉对齐权重矩阵
        N, T, D: batch/token/dim sizes
    Returns:
        block_loss: scalar
    """
    # 1. 投影 student 特征
    z_proj = self.mos_projectors[align_idx](x.reshape(-1, D)).reshape(N, T, -1)

    # 2. BlockRouter 权重 → 选 teacher block（与基座相同）
    block_weights = routing_weights[:, :, :, align_idx]  # (N, T, m)
    m = teacher_all_z.shape[0]
    top_k = min(self.mos_top_k, m)

    if self.training and torch.rand(1).item() < self.mos_random_prob:
        rand_indices = torch.randperm(m, device=x.device)[:top_k]
        select_idx = rand_indices.view(1, 1, top_k).expand(N, T, top_k)
    else:
        _, select_idx = torch.topk(block_weights, k=top_k, dim=-1)

    # 3. 构建 expert 掩码 + 交叉对齐权重
    if expert_indices is not None:
        # MoE block: 使用路由信息
        top1_experts = expert_indices[:, :, 0]
        expert_mask = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))
        cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)
        pair_cond = cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)
        W = cross_weights * expert_mask.float() * pair_cond.float()
    else:
        # Dense block（align_blocks 包含非 MoE block 时）: 退化为全图交叉对齐
        cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)
        pair_cond = cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)
        W = cross_weights * pair_cond.float()

    # 行归一化
    row_sum = W.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    W = W / row_sum  # (N, T, T)

    # 4. 对每个选中的 teacher block 计算交叉对齐 cos_sim
    z_proj_norm = F.normalize(z_proj, dim=-1)  # (N, T, D_z)
    total_loss = torch.tensor(0.0, device=x.device)

    for k_idx in range(top_k):
        teacher_block_idx = select_idx[:, :, k_idx]  # (N, T)
        # 逐 teacher block 处理（select_idx 可能逐 token 不同）
        # 为效率，按 teacher_block_idx 的唯一值分组
        unique_blocks = select_idx[:, :, k_idx].unique()
        k_loss = torch.tensor(0.0, device=x.device)

        for tb in unique_blocks:
            token_mask = (select_idx[:, :, k_idx] == tb)  # (N, T)
            teacher_z = teacher_all_z[tb]  # (N, T, D_z)
            teacher_norm = F.normalize(teacher_z, dim=-1)

            # 交叉对齐: cos_sim_matrix[n, i, j] = cos(proj[n,i], teacher[n,j])
            cos_sim_matrix = torch.bmm(z_proj_norm, teacher_norm.transpose(1, 2))  # (N, T, T)

            # 加权求和: 每个 token 的交叉对齐相似度
            cross_sim = (W * cos_sim_matrix).sum(dim=-1)  # (N, T)

            # 用 BlockRouter 权重加权（仅该 teacher block 被选中的 token）
            selected_block_weight = torch.gather(
                block_weights, dim=-1,
                index=select_idx[:, :, k_idx:k_idx+1]
            ).squeeze(-1)  # (N, T)

            k_loss = k_loss + (-(cross_sim * selected_block_weight) * token_mask.float()).sum()

        total_loss = total_loss + k_loss

    num_cond = (labels != 1000).sum() * T
    block_loss = total_loss / num_cond.clamp(min=1)
    return block_loss
```

**注意**：上面的实现为了清晰用了循环。实际实现时可以优化：对于 `mos_top_k=2` 且 teacher block 数量有限的情况，循环次数很少，性能可接受。

## 配置参数

在 MoS 基座 YAML 基础上新增：

```yaml
DiT_B_config:
  repa_config:
    enc_type: "dinov2-vit-b"
    num_teacher_blocks: 12
    z_dims: [768]
    projector_dim: 2048
    align_blocks: [2, 3, 4]      # 同基座 (0-indexed blocks 2,3,4)
    mos_top_k: 2
    mos_random_prob: 0.05
    cross_align_type: "global_pre"
    align_attn_num_heads: 8
    align_attn_mlp_ratio: 4.0

repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

## 训练脚本

注册到 `train_with_MoS_repa.py`（自然适配，model 返回 `(pred, mos_repa_loss)`）。

```python
"ProMoE_TC_REPA_MoS_CROSS_GLOBAL_PRE_B": (ProMoE_TC_REPA_MoS_CrossGlobalPre, "DiT_B_config"),
```

## 文件清单

| 类型 | 路径 |
|------|------|
| 模型 | `models/models_ProMoE_TC_repa_MoS_naive_choice_cross_global_pre.py` |
| 配置 | `configs/004_ProMoE_B_repa_MoS_naive_choice_cross_global_pre.yaml` |
| 脚本 | `scripts/MoS_repa/run_B_repa_mos_cross_global_pre_train_sample_eval.sh` |
| 训练入口 | `train_with_MoS_repa.py`（新增 model_dict 条目） |

## 注意事项

1. Pre-block attention map 一次计算，所有 `align_blocks` 复用。额外计算量为一个 `GlobalPreAttention` forward pass。
2. `align_blocks` 中可能包含非 MoE block（dense block）。Dense block 没有路由信息，此时交叉对齐退化为全图交叉对齐（所有 conditional token 互相对齐，无 expert 分组约束）。
3. 对于每个对齐 block，交叉对齐 cos_sim 矩阵大小为 `(N, T, T)`，需要对每个选中的 teacher block 分别计算。如果 `mos_top_k=2`，每个 block 需 2 次 `(N, T, T)` bmm，总共 `len(align_blocks) * mos_top_k` 次。
4. 交叉对齐损失仍然使用 `proj_coeff` 作为系数，与 MoS 基座一致。
