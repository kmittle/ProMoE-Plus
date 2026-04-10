# 实验 1: Naive REPA + 全图 Pre-Block Attention 交叉对齐

## 实验目标

在 naive REPA 单层对齐的基础上，利用 conditional expert 的路由信息实现交叉对齐。对齐位置不变（transformer block 输出之后），但不再是 1-to-1 对齐，而是让同一张图、同一个 conditional expert 的 token 互相与对方对应的 DINO token 对齐，权重由**全图 pre-block attention map** 决定。

## 基座模型

`models/models_ProMoE_TC_repa.py`（naive REPA，单层对齐于 `encoder_depth`）

## 核心思想

- 在 patchify 之后、进入 DiT transformer block 之前，用 **2 层可学习 transformer block** 预处理所有 token，然后做 QKV 投影 + scaled dot-product，得到全图 attention map `(N, T, T)`。
- 这个 attention map 一次计算，在后续的 `encoder_depth` 对齐时复用。
- 在对齐时，对于 token i，其与 token j 对应 DINO token 的对齐权重为 `attn_map[n, i, j]`，但**仅限于 token i 和 j 属于同一张图且被分到同一个 conditional expert**。

## 新增模块

### GlobalPreAttention

```python
class GlobalPreAttention(nn.Module):
    """
    2层 transformer + QK 投影，输出全图 attention map。
    在 patchify 后、DiT blocks 前调用一次，后续对齐复用。
    """
    def __init__(self, hidden_size, num_heads=8, num_blocks=2, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        # 2层 pre-norm transformer block 预处理
        self.blocks = nn.ModuleList([
            RouterTransformerBlock(hidden_size, num_heads, mlp_ratio, qk_norm)
            for _ in range(num_blocks)
        ])
        # 单独的 Q, K 投影用于计算 attention map
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** -0.5

    def forward(self, x):
        """
        Args:
            x: (N, T, D) patchified embedding + pos_embed
        Returns:
            attn_map: (N, T, T) softmax-normalized attention weights
        """
        for block in self.blocks:
            x = block(x)
        Q = self.q_proj(x)  # (N, T, D)
        K = self.k_proj(x)  # (N, T, D)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (N, T, T)
        attn_map = F.softmax(attn_logits, dim=-1)  # (N, T, T)
        return attn_map
```

可复用 `models_ProMoE_TC_repa_MoS_naive_choice.py` 中的 `RouterTransformerBlock`。

## 模型修改

### SparseMoeBlock

在 `compute_router()` 末尾（return 之前）缓存路由信息，供 DiT 级别获取：

```python
# 在 compute_router() 中，return 之前:
self._expert_indices = expert_indices   # (N, T, top_k)
```

不改变 `forward()` 和 `compute_router()` 的返回值。

### DiT.__init__

```python
# 新增（在 repa_config 解析区域）:
self.global_pre_attn = GlobalPreAttention(
    hidden_size=hidden_size,
    num_heads=repa_config.get('align_attn_num_heads', 8),
    num_blocks=2,
    mlp_ratio=repa_config.get('align_attn_mlp_ratio', 4.0),
    qk_norm=qk_norm,
)

# projector 保持不变（单个，在 encoder_depth 处使用）
self.projectors = nn.ModuleList([
    build_repa_projector(hidden_size, projector_dim, z_dim) for z_dim in z_dims
])
```

### DiT.forward

签名改为接受 teacher 特征：

```python
def forward(self, x, timestep, context, teacher_all_z=None, **kwargs):
```

流程：

```python
x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
N, T, D = x.shape
t = self.t_embedder(timestep)
y, labels = self.y_embedder(y, self.training)
c = t + y

# 1. 计算全图 pre-block attention map（仅训练时）
cross_attn_map = None
if self.training and self.global_pre_attn is not None and teacher_all_z is not None:
    cross_attn_map = self.global_pre_attn(x)  # (N, T, T)

# 2. 正常过 DiT blocks
cross_align_loss = torch.tensor(0.0, device=x.device)
for i, block in enumerate(self.blocks):
    x = block(x, c, labels)
    # 3. 在 encoder_depth 处计算交叉对齐损失
    if self.training and self.projectors is not None and (i + 1) == self.encoder_depth:
        if teacher_all_z is not None and cross_attn_map is not None:
            # 获取路由信息（从该 block 的 SparseMoeBlock）
            expert_indices = block.mlp._expert_indices  # (N, T, top_k)
            # 取 teacher 最后一层特征（与原始 naive REPA 对齐目标一致）
            teacher_z = teacher_all_z[-1]  # (N, T, D_z)
            # 投影 student 特征
            z_proj = self.projectors[0](x.reshape(-1, D)).reshape(N, T, -1)
            # 计算交叉对齐损失
            cross_align_loss = self.compute_cross_align_loss(
                z_proj, teacher_z, expert_indices, labels, cross_attn_map
            )

x = self.final_layer(x, c)
x = self.unpatchify(x)

if not self.training:
    return x
return x, cross_align_loss
```

注意：`encoder_depth` 处的 block 必须是 MoE block（`use_moe=True`），否则没有路由信息。在 `interleave=True` 时，`encoder_depth` 必须为偶数（对应 0-indexed 奇数位置 1, 3, 5... 的 MoE block，例如 `encoder_depth=4` → 0-indexed block 3）。需要在 `__init__` 中加 assert 验证。

## 交叉对齐损失计算

```python
def compute_cross_align_loss(self, z_proj, teacher_z, expert_indices, labels, cross_weights):
    """
    Args:
        z_proj: (N, T, D_z) 投影后的 student 特征
        teacher_z: (N, T, D_z) teacher 特征
        expert_indices: (N, T, top_k) expert 分配
        labels: (N,) class labels
        cross_weights: (N, T, T) 交叉对齐权重（来自 attention map）
    Returns:
        loss: scalar
    """
    N, T, D_z = z_proj.shape

    # 归一化
    z_proj_norm = F.normalize(z_proj, dim=-1)
    teacher_norm = F.normalize(teacher_z, dim=-1)

    # 余弦相似度矩阵: (N, T, T)
    cos_sim = torch.bmm(z_proj_norm, teacher_norm.transpose(1, 2))

    # 构建同专家掩码: (N, T, T)
    top1_experts = expert_indices[:, :, 0]  # (N, T)
    expert_mask = (top1_experts.unsqueeze(2) == top1_experts.unsqueeze(1))  # (N, T, T)

    # 排除 unconditional 样本
    cond_mask = (labels != 1000)  # (N,)
    token_cond = cond_mask.unsqueeze(1).expand(-1, T)  # (N, T)
    pair_cond = token_cond.unsqueeze(2) & token_cond.unsqueeze(1)  # (N, T, T)

    # 最终权重 = attention权重 * 同专家 * 都是conditional
    W = cross_weights * expert_mask.float() * pair_cond.float()  # (N, T, T)

    # 行归一化
    row_sum = W.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    W = W / row_sum

    # 加权负余弦相似度
    num_cond_tokens = token_cond.sum()
    loss = -(W * cos_sim).sum() / num_cond_tokens.clamp(min=1)

    return loss
```

## 配置参数

YAML 新增字段（在 `DiT_B_config.repa_config` 下）：

```yaml
DiT_B_config:
  repa_config:
    enc_type: "dinov2-vit-b"
    encoder_depth: 4            # 对齐的 block index（必须是 MoE block）
    z_dims: [768]
    projector_dim: 2048
    cross_align_type: "global_pre"   # 标记策略类型
    align_attn_num_heads: 8          # GlobalPreAttention 的 head 数
    align_attn_mlp_ratio: 4.0        # GlobalPreAttention 的 MLP ratio
```

顶层 `repa_config` 不变：

```yaml
repa_config:
  enc_type: "dinov2-vit-b"
  proj_coeff: 0.5
```

## 训练脚本

注册到 `train_with_MoS_repa.py` 的 `model_dict`，复用其训练循环（model 返回 `(pred, cross_align_loss)`，训练循环乘以 `proj_coeff`）。虽然 naive REPA 只用一个 teacher block，但提取全部 teacher block features 的额外开销极小（teacher 冻结，前向快速）。

```python
# train_with_MoS_repa.py model_dict 新增:
from models.models_ProMoE_TC_repa_cross_global_pre import DiT as ProMoE_TC_REPA_CrossGlobalPre
"ProMoE_TC_REPA_CROSS_GLOBAL_PRE_B": (ProMoE_TC_REPA_CrossGlobalPre, "DiT_B_config"),
```

## 文件清单

| 类型 | 路径 |
|------|------|
| 模型 | `models/models_ProMoE_TC_repa_cross_global_pre.py` |
| 配置 | `configs/004_ProMoE_B_repa_cross_global_pre.yaml` |
| 脚本 | `scripts/repa/run_B_repa_cross_global_pre_train_sample_eval.sh` |
| 训练入口 | `train_with_MoS_repa.py`（新增 model_dict 条目） |

## 注意事项

1. `encoder_depth` 对应的 block 必须是 MoE block（`interleave=True` 时 `encoder_depth` 为偶数，对应 0-indexed 奇数位置 1,3,5...），否则无路由信息。在 `__init__` 中需要 assert `use_moe_flag[encoder_depth - 1] == True`。
2. `GlobalPreAttention` 的输入是 patchify 后的特征（尚未经过任何 DiT block），语义信息有限，但通过 2 层 transformer 预处理可以捕获初步的空间关系。
3. 交叉对齐的 cos_sim 矩阵大小为 `(N, T, T)`，对于 256×256 图像 patch_size=2 时 T=256，占用 `N*256*256*4` bytes ≈ N*0.25MB，batch_size=256 时约 64MB，可以接受。
4. 权重矩阵行归一化后，loss 的量纲与标准 REPA 一致（负余弦相似度的均值）。
5. unconditional 样本（`labels==1000`）的 token 不参与交叉对齐损失，这些 token 的 REPA 对齐信号完全由 routing contrastive loss 通过 `AddAuxiliaryLoss` 机制间接提供。
6. **Teacher 特征差异**：本实验注册在 `train_with_MoS_repa.py`，使用 `extract_all_teacher_block_features()` 提取 teacher 特征（各 block 原始输出，无 final layer norm），而原始 naive REPA 使用 `extract_teacher_features()` 提取的是经过 final layer norm 的最后一层输出（`x_norm_patchtokens`）。使用 `teacher_all_z[-1]` 与原始 REPA 的 teacher 特征存在微小差异（缺少 final norm），但 projector 应能自适应。
