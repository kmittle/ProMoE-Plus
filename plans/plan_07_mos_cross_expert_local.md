# 实验 7: MoS 多层对齐 + Expert-Local Attention 交叉对齐

## 实验目标

在 MoS naive choice 多层对齐的基础上，利用 conditional expert 的路由信息实现交叉对齐。Attention map 在每个对齐 block 的输出处计算，且**仅在同一张图、同一个 conditional expert 的 token 之间**计算（与实验 6 的全图 attention 对比）。多 block 场景使用 1 层 transformer 预处理。

## 基座模型

`models/models_ProMoE_TC_repa_MoS_naive_choice.py`（MoS naive choice，多层对齐）

## 核心思想

- 在每个 `align_blocks` 中的 block 输出后，用 **1 层可学习 transformer block** 预处理（多 block 场景），然后做 QKV + masked scaled dot-product，attention **仅在同一张图、同一个 conditional expert 的 token 之间** 计算。
- mask 使得 softmax 只在 expert group 内归一化，组外位置的 attention weight 为 0。
- 没有 pre-block 变体（必须在路由之后才知道分组）。

## 新增模块

### ExpertLocalAttention (1层变体)

```python
class ExpertLocalAttention(nn.Module):
    """
    Expert-local masked attention。多 block 对齐场景使用 1 层 transformer。
    """
    def __init__(self, hidden_size, num_heads=8, num_blocks=1, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            RouterTransformerBlock(hidden_size, num_heads, mlp_ratio, qk_norm)
            for _ in range(num_blocks)
        ])
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** -0.5

    def forward(self, x, expert_local_mask):
        """
        Args:
            x: (N, T, D) block 输出特征
            expert_local_mask: (N, T, T) bool，同图同 expert 的 token 对为 True
        Returns:
            attn_map: (N, T, T) expert-local attention weights
        """
        for block in self.blocks:
            x = block(x)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_logits = attn_logits.masked_fill(~expert_local_mask, float('-inf'))
        attn_map = F.softmax(attn_logits, dim=-1)
        attn_map = attn_map.nan_to_num(0.0)
        return attn_map
```

## 模型修改

### SparseMoeBlock

缓存 `self._expert_indices`（同前）。

### DiT.__init__

```python
# 每个对齐 block 有独立的 ExpertLocalAttention（1层）
self.expert_local_attns = nn.ModuleList([
    ExpertLocalAttention(
        hidden_size=hidden_size,
        num_heads=repa_config.get('align_attn_num_heads', 8),
        num_blocks=1,   # 多 block 对齐 → 1 层
        mlp_ratio=repa_config.get('align_attn_mlp_ratio', 4.0),
        qk_norm=qk_norm,
    )
    for _ in range(num_align_blocks)
])

# 保留 MoS 的所有原有组件
```

### DiT.forward

```python
def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
    # ... patchify, embedding, BlockRouter 预计算 ...

    mos_repa_loss = torch.tensor(0.0, device=x.device)
    for i, block in enumerate(self.blocks):
        x = block(x, c, labels)
        if self.training and routing_weights is not None and i in self.align_block_to_idx:
            align_idx = self.align_block_to_idx[i]
            expert_indices = block.mlp._expert_indices if block.use_moe else None

            # 构建 expert-local mask
            expert_local_mask = self._build_expert_local_mask(
                expert_indices, labels, N, T
            )

            # 计算 expert-local attention map
            cross_attn_map = self.expert_local_attns[align_idx](x, expert_local_mask)

            block_loss = self.compute_cross_mos_repa_loss(
                x, align_idx, routing_weights, teacher_all_z,
                expert_indices, labels, cross_attn_map, N, T, D
            )
            mos_repa_loss = mos_repa_loss + block_loss

    # ... averaging, final_layer, unpatchify ...
```

### _build_expert_local_mask

与实验 3 完全一致：

```python
def _build_expert_local_mask(self, expert_indices, labels, N, T):
    if expert_indices is None:
        # dense block: 所有 conditional token 互相可见
        cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)
        return cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)

    top1_experts = expert_indices[:, :, 0]
    expert_match = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))
    cond_mask = (labels != 1000).unsqueeze(1).expand(-1, T)
    pair_cond = cond_mask.unsqueeze(2) & cond_mask.unsqueeze(1)
    return expert_match & pair_cond
```

## 交叉对齐 MoS 损失计算

与实验 5、6 的 `compute_cross_mos_repa_loss` 一致。由于 `ExpertLocalAttention` 输出已经是 expert-local 的，`compute_cross_mos_repa_loss` 中的 expert_mask 冗余但保留。

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
    cross_align_type: "expert_local"
    align_attn_num_heads: 8
    align_attn_mlp_ratio: 4.0

repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

## 训练脚本

注册到 `train_with_MoS_repa.py`。

```python
"ProMoE_TC_REPA_MoS_CROSS_EXPERT_LOCAL_B": (ProMoE_TC_REPA_MoS_CrossExpertLocal, "DiT_B_config"),
```

## 文件清单

| 类型 | 路径 |
|------|------|
| 模型 | `models/models_ProMoE_TC_repa_MoS_naive_choice_cross_expert_local.py` |
| 配置 | `configs/004_ProMoE_B_repa_MoS_naive_choice_cross_expert_local.yaml` |
| 脚本 | `scripts/MoS_repa/run_B_repa_mos_cross_expert_local_train_sample_eval.sh` |
| 训练入口 | `train_with_MoS_repa.py`（新增 model_dict 条目） |

## 注意事项

1. **不同 block 的路由不同**：MoS 中每个 MoE block 有独立的 `SparseMoeBlock`，因此不同 `align_blocks` 处的 expert 分配不同。这意味着同一对 token 在不同 block 处可能属于不同 expert，对应不同的 expert-local mask。
2. **expert-local mask 每个 block 重新构建**：因为每个 MoE block 的路由是独立的，不能复用上一个 block 的 mask。
3. **变长组的 attention**：不同 expert 的 token 数量差异大，expert-local softmax 的有效序列长度不同。对于大 group，attention 更分散；小 group，attention 更集中。
4. **dense block fallback**：如果 `align_blocks` 包含非 MoE block，所有 conditional token 互相可见（无 expert 分组），等价于全图 attention 变体。
5. **nan 处理**：unconditional token 不在任何 group 中（mask 全 False），softmax 结果为 nan，用 `nan_to_num(0.0)` 处理后这些 token 对损失没有贡献。
