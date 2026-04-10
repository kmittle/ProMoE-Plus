# 实验 2: Naive REPA + 全图 Per-Block Attention 交叉对齐

## 实验目标

在 naive REPA 单层对齐的基础上，利用 conditional expert 的路由信息实现交叉对齐。与实验 1 不同，attention map 不在 patchify 后计算，而是在**对齐 block 的输出处**用 2 层可学习 transformer block 预处理后计算（单 block 对齐场景使用 2 层）。Attention 仍然在全图所有 token 上计算。

## 基座模型

`models/models_ProMoE_TC_repa.py`（naive REPA，单层对齐于 `encoder_depth`）

## 核心思想

- 在 `encoder_depth` 处的 block 输出后，先用 **2 层可学习 transformer block** 对 block 输出 token 做预处理，再做 QKV 投影 + scaled dot-product，得到 block-wise attention map `(N, T, T)`。
- 这个 attention map 基于经过 DiT 处理后的特征计算，比 pre-block（实验 1）有更丰富的语义信息。
- 由于是单 block 对齐（naive REPA），使用 2 层 transformer block 做预处理。

## 新增模块

### BlockAlignAttention

```python
class BlockAlignAttention(nn.Module):
    """
    对 block 输出做 transformer 预处理 + QK attention map 计算。
    单 block 对齐场景使用 2 层（多 block 场景使用 1 层，见 MoS 变体）。
    """
    def __init__(self, hidden_size, num_heads=8, num_blocks=2, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            RouterTransformerBlock(hidden_size, num_heads, mlp_ratio, qk_norm)
            for _ in range(num_blocks)
        ])
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** -0.5

    def forward(self, x):
        """
        Args:
            x: (N, T, D) block 输出特征
        Returns:
            attn_map: (N, T, T) softmax-normalized attention weights
        """
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

同实验 1：缓存 `self._expert_indices`。

### DiT.__init__

```python
# 新增:
self.block_align_attn = BlockAlignAttention(
    hidden_size=hidden_size,
    num_heads=repa_config.get('align_attn_num_heads', 8),
    num_blocks=2,   # 单 block 对齐 → 2 层
    mlp_ratio=repa_config.get('align_attn_mlp_ratio', 4.0),
    qk_norm=qk_norm,
)
# projector 保持不变
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
                # 在 block 输出处计算 attention map
                cross_attn_map = self.block_align_attn(x)  # (N, T, T)
                expert_indices = block.mlp._expert_indices
                teacher_z = teacher_all_z[-1]  # teacher 最后一层
                z_proj = self.projectors[0](x.reshape(-1, D)).reshape(N, T, -1)
                cross_align_loss = self.compute_cross_align_loss(
                    z_proj, teacher_z, expert_indices, labels, cross_attn_map
                )

    x = self.final_layer(x, c)
    x = self.unpatchify(x)
    if not self.training:
        return x
    return x, cross_align_loss
```

## 交叉对齐损失计算

与实验 1 的 `compute_cross_align_loss` 完全一致（同专家掩码 + 行归一化 + 加权负余弦相似度）。

## 配置参数

```yaml
DiT_B_config:
  repa_config:
    enc_type: "dinov2-vit-b"
    encoder_depth: 4
    z_dims: [768]
    projector_dim: 2048
    cross_align_type: "global_block"
    align_attn_num_heads: 8
    align_attn_mlp_ratio: 4.0

repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

## 训练脚本

同实验 1，注册到 `train_with_MoS_repa.py`。

```python
"ProMoE_TC_REPA_CROSS_GLOBAL_BLOCK_B": (ProMoE_TC_REPA_CrossGlobalBlock, "DiT_B_config"),
```

## 文件清单

| 类型 | 路径 |
|------|------|
| 模型 | `models/models_ProMoE_TC_repa_cross_global_block.py` |
| 配置 | `configs/004_ProMoE_B_repa_cross_global_block.yaml` |
| 脚本 | `scripts/repa/run_B_repa_cross_global_block_train_sample_eval.sh` |
| 训练入口 | `train_with_MoS_repa.py`（新增 model_dict 条目） |

## 注意事项

1. 与实验 1 相比，attention map 基于 DiT block 输出特征，包含更丰富的语义信息（已经过自注意力和 MoE 处理）。
2. 计算顺序：先过 DiT block → 获取路由信息 → `BlockAlignAttention` 处理 block 输出 → 计算 attention map → 投影 + 交叉对齐 loss。注意 `BlockAlignAttention` 的输入是 block 的原始输出 `x`，不是投影后的 `z_proj`。
3. `BlockAlignAttention` 使用 2 层 transformer block，因为只需要处理一个对齐 block（单 block 场景允许更深的预处理）。
4. 同实验 1，`encoder_depth` 对应的 block 必须是 MoE block（`interleave=True` 时 `encoder_depth` 为偶数）。
5. `BlockAlignAttention` 的梯度会通过 block 输出 `x` 回传到 DiT blocks，这是期望的行为（让 DiT 学到有利于交叉对齐的表示）。
6. **Teacher 特征差异**：同实验 1，使用 `extract_all_teacher_block_features()` 提取的 `teacher_all_z[-1]` 与原始 REPA 的 `extract_teacher_features()` 输出存在微小差异（缺少 final layer norm），projector 应能自适应。
