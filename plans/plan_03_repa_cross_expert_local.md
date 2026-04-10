# 实验 3: Naive REPA + Expert-Local Attention 交叉对齐

## 实验目标

在 naive REPA 单层对齐的基础上，利用 conditional expert 的路由信息实现交叉对齐。与实验 2 不同，attention map **仅在同一张图且同一个 conditional expert 的 token 之间**计算，而非全图所有 token。这样 attention 权重更加聚焦于路由相关的 token 关系。

## 基座模型

`models/models_ProMoE_TC_repa.py`（naive REPA，单层对齐于 `encoder_depth`）

## 核心思想

- 必须在 MoE 路由之后才知道哪些 token 被分到同一个 expert，因此**没有 pre-block 变体**（与实验 1 的区别）。
- 在 `encoder_depth` 处的 block 输出后，用 **2 层可学习 transformer block** 预处理（单 block 对齐场景），但计算 attention 时，通过 mask 限制为仅同一张图、同一个 conditional expert 的 token 之间。
- 不同图的 token 加噪程度不同，不适合直接做 attention 计算，所以强制同图约束。

## 新增模块

### ExpertLocalAttention

```python
class ExpertLocalAttention(nn.Module):
    """
    对 block 输出做 transformer 预处理 + expert-local masked attention map 计算。
    attention 仅在同一张图、同一个 conditional expert 的 token 之间计算。
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

    def forward(self, x, expert_local_mask):
        """
        Args:
            x: (N, T, D) block 输出特征
            expert_local_mask: (N, T, T) bool，True 表示两个 token
                属于同一张图且同一个 conditional expert
        Returns:
            attn_map: (N, T, T) expert-local softmax-normalized attention weights
                （不在同一组的位置为 0）
        """
        for block in self.blocks:
            x = block(x)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (N, T, T)
        # 把不在同一个 expert group 的位置设为 -inf
        attn_logits = attn_logits.masked_fill(~expert_local_mask, float('-inf'))
        attn_map = F.softmax(attn_logits, dim=-1)  # (N, T, T)
        # 处理全为 -inf 的行（token 所在组只有自己，softmax → nan → 置 0）
        attn_map = attn_map.nan_to_num(0.0)
        return attn_map
```

## 模型修改

### SparseMoeBlock

同实验 1、2：缓存 `self._expert_indices`。

### DiT.__init__

```python
# 新增:
self.expert_local_attn = ExpertLocalAttention(
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
                expert_indices = block.mlp._expert_indices  # (N, T, top_k)

                # 构建 expert-local mask: (N, T, T)
                expert_local_mask = self._build_expert_local_mask(
                    expert_indices, labels, N, T
                )

                # 计算 expert-local attention map
                cross_attn_map = self.expert_local_attn(x, expert_local_mask)

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

### _build_expert_local_mask

```python
def _build_expert_local_mask(self, expert_indices, labels, N, T):
    """
    构建 expert-local 掩码：仅同一张图、同一个 conditional expert 的 token 对为 True。

    注意：batch 中每个样本是一张图，所以 (N, T, T) 中不同 N 维度的样本
    天然是独立的，不存在跨图的 token pair。

    Args:
        expert_indices: (N, T, top_k)
        labels: (N,)
        N, T: batch size, num tokens
    Returns:
        mask: (N, T, T) bool
    """
    top1_experts = expert_indices[:, :, 0]  # (N, T)
    # 同 expert: (N, T, T)
    expert_match = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))

    # conditional mask: 排除 unconditional 样本的所有 token
    cond_mask = (labels != 1000)  # (N,)
    token_cond = cond_mask.unsqueeze(1).expand(-1, T)  # (N, T)
    pair_cond = token_cond.unsqueeze(2) & token_cond.unsqueeze(1)  # (N, T, T)

    return expert_match & pair_cond
```

## 交叉对齐损失计算

与实验 1、2 的 `compute_cross_align_loss` 相同。由于 `ExpertLocalAttention` 输出的 attention map 已经是 expert-local 的（非同组位置为 0），`compute_cross_align_loss` 中的 expert_mask 实际上是冗余的，但保留以保持接口一致。

## 配置参数

```yaml
DiT_B_config:
  repa_config:
    enc_type: "dinov2-vit-b"
    encoder_depth: 4
    z_dims: [768]
    projector_dim: 2048
    cross_align_type: "expert_local"
    align_attn_num_heads: 8
    align_attn_mlp_ratio: 4.0

repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

## 训练脚本

同实验 1、2，注册到 `train_with_MoS_repa.py`。

```python
"ProMoE_TC_REPA_CROSS_EXPERT_LOCAL_B": (ProMoE_TC_REPA_CrossExpertLocal, "DiT_B_config"),
```

## 文件清单

| 类型 | 路径 |
|------|------|
| 模型 | `models/models_ProMoE_TC_repa_cross_expert_local.py` |
| 配置 | `configs/004_ProMoE_B_repa_cross_expert_local.yaml` |
| 脚本 | `scripts/repa/run_B_repa_cross_expert_local_train_sample_eval.sh` |
| 训练入口 | `train_with_MoS_repa.py`（新增 model_dict 条目） |

## 注意事项

1. Expert-local attention 的 softmax 在每行仅对同组 token 计算（其余 mask 为 `-inf`）。如果某个 token 在其 expert group 内只有自己（组大小为 1），softmax 退化为 `[1.0]`，等价于标准 1-to-1 REPA。
2. 不同 expert 的 group 大小差异可能很大，导致 attention 的有效范围不同（某些 expert 可能有大量 token，某些很少）。
3. 由于 attention 仅在组内计算，组间 token 关系无法被捕获。与实验 2（全图 attention）相比，这减少了信息量但更聚焦于路由相关性。
4. 实现中 `nan_to_num` 处理 softmax 全 `-inf` 的情况（unconditional token 不在任何 cond group 中）。
5. Expert-local mask 的构建开销为 `O(N * T * T)`，与 cos_sim 矩阵相同。
6. **Teacher 特征差异**：同实验 1，使用 `extract_all_teacher_block_features()` 提取的 `teacher_all_z[-1]` 与原始 REPA 的 `extract_teacher_features()` 输出存在微小差异（缺少 final layer norm），projector 应能自适应。
