# 实验 6: MoS 多层对齐 + 全图 Per-Block Attention 交叉对齐

## 实验目标

在 MoS naive choice 多层对齐的基础上，利用 conditional expert 的路由信息实现交叉对齐。与实验 5 不同，attention map 不在 patchify 后计算，而是在**每个对齐 block 的输出处**分别计算。由于是多 block 对齐（MoS），每个 block 使用 **1 层**可学习 transformer block 预处理（非 2 层）。

## 基座模型

`models/models_ProMoE_TC_repa_MoS_naive_choice.py`（MoS naive choice，多层对齐）

## 核心思想

- 在每个 `align_blocks` 中的 block 输出后，用 **1 层可学习 transformer block** 预处理，然后 QKV + scaled dot-product → 该 block 专属的 attention map `(N, T, T)`。
- 每个对齐 block 有独立的 attention map，捕获该 block 输出特征中的 token 关系。
- 与实验 5 的区别：attention 基于 DiT block 输出（更丰富的语义），但需要为每个对齐 block 单独计算。
- 多 block 场景使用 1 层 transformer（而非 2 层），以控制参数量和计算量。

## 新增模块

### BlockAlignAttention (1层变体)

```python
class BlockAlignAttention(nn.Module):
    """
    对 block 输出做 1 层 transformer 预处理 + QK attention map 计算。
    多 block 对齐场景使用 1 层。
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

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_map = F.softmax(attn_logits, dim=-1)
        return attn_map
```

## 模型修改

### SparseMoeBlock

缓存 `self._expert_indices`（同前）。

### DiT.__init__

```python
# 每个对齐 block 有独立的 BlockAlignAttention（1层）
self.block_align_attns = nn.ModuleList([
    BlockAlignAttention(
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
    # ... patchify, embedding, BlockRouter 预计算 routing_weights ...
    # （无 pre-block attention 计算，与实验 5 不同）

    mos_repa_loss = torch.tensor(0.0, device=x.device)
    for i, block in enumerate(self.blocks):
        x = block(x, c, labels)
        if self.training and routing_weights is not None and i in self.align_block_to_idx:
            align_idx = self.align_block_to_idx[i]
            expert_indices = block.mlp._expert_indices if block.use_moe else None

            # 为该 block 计算独立的 attention map
            cross_attn_map = self.block_align_attns[align_idx](x)  # (N, T, T)

            block_loss = self.compute_cross_mos_repa_loss(
                x, align_idx, routing_weights, teacher_all_z,
                expert_indices, labels, cross_attn_map, N, T, D
            )
            mos_repa_loss = mos_repa_loss + block_loss

    # ... averaging, final_layer, unpatchify ...
```

## 交叉对齐 MoS 损失计算

与实验 5 的 `compute_cross_mos_repa_loss` 完全一致。

## 配置参数

```yaml
DiT_B_config:
  repa_config:
    enc_type: "dinov2-vit-b"
    num_teacher_blocks: 12
    z_dims: [768]
    projector_dim: 2048
    align_blocks: [3, 4, 5]
    mos_top_k: 2
    mos_random_prob: 0.05
    cross_align_type: "global_block"
    align_attn_num_heads: 8
    align_attn_mlp_ratio: 4.0

repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

## 训练脚本

注册到 `train_with_MoS_repa.py`。

```python
"ProMoE_TC_REPA_MoS_CROSS_GLOBAL_BLOCK_B": (ProMoE_TC_REPA_MoS_CrossGlobalBlock, "DiT_B_config"),
```

## 文件清单

| 类型 | 路径 |
|------|------|
| 模型 | `models/models_ProMoE_TC_repa_MoS_naive_choice_cross_global_block.py` |
| 配置 | `configs/004_ProMoE_B_repa_MoS_naive_choice_cross_global_block.yaml` |
| 脚本 | `scripts/MoS_repa/run_B_repa_mos_cross_global_block_train_sample_eval.sh` |
| 训练入口 | `train_with_MoS_repa.py`（新增 model_dict 条目） |

## 注意事项

1. **1 层 vs 2 层**：实验 2（naive REPA 单 block）使用 2 层 transformer 预处理，本实验（MoS 多 block）使用 1 层。这是 implementation plan 中的明确要求：多 block 对齐时用 1 层，以控制新增参数量。
2. **参数量**：每个对齐 block 有独立的 `BlockAlignAttention`。如果 `align_blocks = [3, 4, 5]`（3 个 block），共有 3 × (1层 transformer + QK proj) 的新增参数。
3. **计算量**：每个对齐 block 需要一次 `BlockAlignAttention` forward + 一次 `(N, T, T)` cos_sim_matrix 计算。总计算量 = `len(align_blocks) × (BlockAlignAttention + cross_mos_loss)`。
4. **共享 vs 独立**：当前设计为每个对齐 block 独立的 `BlockAlignAttention`。如果参数量成为瓶颈，可以考虑所有对齐 block 共享一个 `BlockAlignAttention`（但语义上不同 block 的输出特征分布不同，独立更合理）。
5. 对于 `align_blocks` 中的 dense block（非 MoE），退化逻辑同实验 5。
