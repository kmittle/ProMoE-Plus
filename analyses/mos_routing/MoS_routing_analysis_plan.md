# MoS Router Block Selection Analysis — Implementation Plan

## 1. Goal

Analyze and visualize MoS router behavior: for each aligned generation block, the router assigns tokens routing weights over teacher (DINOv2) blocks. Core questions:

- **Which teacher blocks** does each generation block prefer?
- **How do routing patterns vary** across tokens (spatial position), diffusion timesteps, and ImageNet classes?
- **Token-wise vs block-wise**: how much do routing weights differ across tokens within the same image?

---

## 2. MoS Model Variants & Routing Interfaces

| Model File | Router Class | Output Shape | Granularity | Timing | model_dict Key |
|---|---|---|---|---|---|
| `models_ProMoE_TC_repa_MoS.py` | `AdaLNRouter` (per-block) | `(N, T, m)` per block | token-wise | inside `compute_mos_repa_loss` | `ProMoE_TC_REPA_MoS_B` |
| `models_ProMoE_TC_repa_MoS_naive.py` | `BlockRouter` (global) | `(N, T, m, n)` | token-wise | forward 开头一次性 | `ProMoE_TC_REPA_MoS_Naive_B` |
| `models_ProMoE_TC_repa_MoS_naive_choice.py` | `BlockRouter` (global) | `(N, T, m, n')` | token-wise | forward 开头一次性 | `ProMoE_TC_REPA_MoS_Naive_Choice_B` |
| `models_ProMoE_TC_repa_MoS_naive_choice_.py` | `BlockRouter` (sep t/y) | `(N, T, m, n)` | token-wise | forward 开头一次性 | `ProMoE_TC_REPA_MoS_Naive_Choice_Sep_B` |
| `models_ProMoE_TC_repa_MoS_naive_choice_blockwise.py` | `BlockRouter` (pooled) | `(N, m, n)` → expand | block-wise | forward 开头一次性 | `ProMoE_TC_REPA_MoS_Naive_Choice_Blockwise_B` |
| `models_ProMoE_TC_repa_MoS_choice_per_block.py` | `PerBlockRouter` (per-block) | `(N, T, m)` per block | token-wise | 每个 aligned block 输出后 | `ProMoE_TC_REPA_MoS_Choice_PerBlock_B` |

> **Note**: `models_ProMoE_TC_repa_multi_align.py` 的 `AlignCoefficientPredictor` 输出 sigmoid 系数而非 teacher block 路由权重, 不属于本分析范围.

### 统一输出格式

```python
routing_data: Dict[int, Tensor]
# key: DiT block index (e.g., 2, 3, 4)
# value: (N, T, m) routing weights (softmax-normalized over m teacher blocks)
```

转换规则:
- `(N, T, m, n)` 模型: 按 align_idx 切片 `[:, :, :, align_idx]` → `(N, T, m)`
- `(N, T, m)` per-block 模型: 直接使用
- Blockwise 模型: expand 后格式相同 (值在 T 维恒等)

---

## 3. Directory Structure

遵循 `analyses/` 现有模式: 入口脚本 `run_*.py` + 同名 `.md` 使用文档 + `analyses/<subdir>/` 共享模块.

```
analyses/
├── mos_routing/                          # 新建共享模块目录
│   ├── __init__.py
│   ├── extract.py                        # 路由权重提取 (hook-based, model-agnostic)
│   ├── aggregate.py                      # 统计聚合 (per-block, per-timestep, per-class, variance)
│   └── plotting.py                       # 可视化 (heatmap, spatial map, bar chart)
├── run_mos_routing_analysis.py           # 入口脚本
└── run_mos_routing_analysis.md           # 使用文档
```

### 复用现有模块

| 现有模块 | 复用内容 |
|---|---|
| `analyses/t_SNE/checkpoint_utils.py` | `load_runtime_cfg`, `resolve_config_from_checkpoint`, `parse_checkpoint_step` |
| `analyses/t_SNE/imagenet_utils.py` | `get_imagenet_class_names`, `sample_class_ids`, `slugify_class_name` |
| `analyses/t_SNE/sampling.py` | `build_model` (加载模型 + EMA 权重), `compute_analysis_steps` |
| `analyses/t_SNE/model_registry.py` | 已包含 `train_with_MoS_repa.py` 的 `model_dict`, 无需额外注册 |
| `analyses/heatmap/sample_specs.py` | `HeatmapSampleSpec`, `build_heatmap_sample_specs` (用于 spatial map 样本选择) |

---

## 4. Module Design

### 4.1 `mos_routing/extract.py` — 路由权重提取

**核心思路**: forward hook 捕获路由权重, 无需修改模型代码. 参照 `heatmap/capture.py` 的 `DynamicRepaWeightCapture` 和 `t_SNE/routing_capture.py` 的 `TokenRoutingCapture` 模式.

```python
class MoSRoutingCapture:
    """
    Model-agnostic MoS routing weight capture via forward hooks.
    
    Pattern follows DynamicRepaWeightCapture (heatmap/capture.py):
    - __init__: detect model type, register hooks
    - enable()/disable(): toggle capture
    - get_routing_data() -> Dict[int, Tensor]: return captured {block_idx: (N, T, m)}
    """

    def __init__(self, model):
        self.model = model
        self.model_type = self._detect_model_type()
        # MoS model aligns ALL blocks (no align_blocks attr); others have explicit align_blocks
        if hasattr(model, 'align_blocks') and model.align_blocks:
            self.align_blocks = list(model.align_blocks)
            self.align_block_to_idx = dict(model.align_block_to_idx)
        elif self.model_type == 'mos':
            depth = len(model.blocks)
            self.align_blocks = list(range(depth))
            self.align_block_to_idx = {i: i for i in range(depth)}
        else:
            self.align_blocks = []
            self.align_block_to_idx = {}
        self._enabled = False
        self._captured = {}  # {block_idx: (N, T, m)}
        self._handles = []
        self._register_hooks()

    def _detect_model_type(self) -> str:
        """Auto-detect from model attributes (no config needed)."""
        model = self.model
        if hasattr(model, 'per_block_routers') and model.per_block_routers is not None:
            return 'per_block'
        if hasattr(model, 'mos_routers') and model.mos_routers is not None:
            return 'mos'
        if hasattr(model, 'block_router') and model.block_router is not None:
            # Distinguish blockwise vs token-wise by checking class name
            module_name = type(model).__module__
            if 'blockwise' in module_name:
                return 'blockwise'
            return 'global'  # covers naive, naive_choice, naive_choice_sep
        raise RuntimeError(f"Cannot detect MoS model type from {type(model).__name__}")

    def _register_hooks(self):
        """Register forward hooks based on model_type."""
        if self.model_type == 'global' or self.model_type == 'blockwise':
            # Hook on block_router.forward → capture (N, T, m, n)
            self._handles.append(
                self.model.block_router.register_forward_hook(self._global_router_hook)
            )
        elif self.model_type == 'per_block':
            # Hook on each PerBlockRouter → capture (N, T, m)
            for align_idx, router in enumerate(self.model.per_block_routers):
                block_idx = self.align_blocks[align_idx]
                self._handles.append(
                    router.register_forward_hook(self._make_per_block_hook(block_idx))
                )
        elif self.model_type == 'mos':
            # Hook on each AdaLNRouter → capture logits, apply softmax externally
            for block_idx, router in enumerate(self.model.mos_routers):
                self._handles.append(
                    router.register_forward_hook(self._make_adaln_hook(block_idx))
                )

    def _global_router_hook(self, module, inputs, output):
        """Capture (N, T, m, n) from BlockRouter, slice per align_block."""
        if not self._enabled:
            return
        # output: (N, T, m, n) — already softmax-normalized
        routing_weights = output.detach()
        for block_idx, align_idx in self.align_block_to_idx.items():
            self._captured[block_idx] = routing_weights[:, :, :, align_idx].cpu()  # (N, T, m)

    def _make_per_block_hook(self, block_idx):
        def hook(module, inputs, output):
            if not self._enabled:
                return
            self._captured[block_idx] = output.detach().cpu()  # (N, T, m)
        return hook

    def _make_adaln_hook(self, block_idx):
        def hook(module, inputs, output):
            if not self._enabled:
                return
            # AdaLNRouter returns logits (N, T, K); need softmax
            import torch.nn.functional as F
            self._captured[block_idx] = F.softmax(output.detach(), dim=-1).cpu()  # (N, T, m)
        return hook

    def enable(self):
        self._enabled = True
        self._captured = {}

    def disable(self):
        self._enabled = False

    def get_routing_data(self) -> Dict[int, torch.Tensor]:
        return dict(self._captured)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
```

**关键发现: 路由权重不依赖 `teacher_all_z`**

所有 router (`BlockRouter`, `PerBlockRouter`, `AdaLNRouter`) 的输入都是 `(x, c)` — 即 patchified latent 和条件向量。`teacher_all_z` 仅用于 MoS loss 计算 (cosine similarity), 不参与路由权重的生成。

但 forward 中有守卫条件 `if self.training and teacher_all_z is not None`, 因此分析时需要:
1. 设置 `model.train()` 激活路由代码路径
2. 传入 **dummy `teacher_all_z`** (shape `(m, N, T, D_teacher)`, 值为 zeros) 满足非 None 条件
3. 使用 `torch.no_grad()` 包裹 — loss 计算结果无意义但无副作用

这意味着分析时**不需要加载 teacher encoder**, 大幅简化流程。

**`MoS` 模型兼容**: `models_ProMoE_TC_repa_MoS.py` 没有 `align_blocks` 属性 (所有 block 都参与对齐). `_detect_model_type` 返回 `'mos'` 时, `align_blocks` fallback 为 `list(range(depth))`.

### 4.2 `mos_routing/aggregate.py` — 统计聚合

```python
@dataclass
class BlockRoutingStats:
    """Per-generation-block routing statistics, accumulated online."""
    block_idx: int
    mean_weights: np.ndarray           # (m,) average routing weight per teacher block
    top1_freq: np.ndarray              # (m,) frequency of being top-1
    topk_freq: np.ndarray              # (m,) frequency of being in top-k
    entropy: float                     # mean routing entropy over all tokens
    token_variance: float              # mean variance across tokens (T dim)
    num_samples: int                   # total tokens accumulated

class OnlineRoutingAggregator:
    """
    Accumulates routing statistics online (no full tensor storage).
    Call update() per batch, then finalize() to get BlockRoutingStats.
    """
    def __init__(self, align_blocks: List[int], num_teacher_blocks: int, top_k: int = 2):
        ...

    def update(self, routing_data: Dict[int, Tensor], timesteps: Tensor = None, labels: Tensor = None):
        """Accumulate one batch of routing data."""
        ...

    def finalize(self) -> Dict[int, BlockRoutingStats]:
        ...

class PerTimestepRoutingAggregator:
    """
    Accumulates routing statistics keyed by denoising step index.
    Each step maintains its own OnlineRoutingAggregator.
    Used for timestep-level histogram visualization (图表 3).
    """
    def __init__(self, align_blocks, num_teacher_blocks, analysis_steps, top_k=2):
        # analysis_steps: list of denoising step indices, e.g. [50, 100, 150, 200, 250]
        ...

    def update(self, routing_data, step_idx):
        """Accumulate one batch of routing data for a specific denoising step."""
        ...

    def finalize(self) -> Dict[int, Dict[int, BlockRoutingStats]]:
        """Returns {step_idx: {block_idx: stats}}"""
        ...
```

### 4.3 `mos_routing/plotting.py` — 可视化函数

输出 SVG + PNG 双格式 (与 `analyses/heatmap/plotting.py` 和 `analyses/t_SNE/plotting.py` 一致).

#### 图表 1: Per-block Teacher Block Selection Histogram (核心图)

每个 aligned generation block 一张柱状图, **汇总所有 timestep/类别/样本/token** 的路由统计.

```
横轴: Teacher block index (0..m-1)
纵轴: Top-1 selection frequency (该 teacher block 被选为 top-1 的比例)
```
- 每个 aligned block (如 block 2, 3, 4) 各一张子图
- 直观展示每个 generation block 偏好哪些 teacher blocks

#### 图表 2: All-blocks Aggregated Teacher Block Selection Histogram

将所有 aligned generation blocks 的路由统计合并为一张柱状图, **汇总所有 timestep/类别/样本/token**.

```
横轴: Teacher block index (0..m-1)
纵轴: Top-1 selection frequency (汇总所有 aligned blocks 的 token)
```
- 展示整体路由偏好: 哪些 teacher blocks 被选中最多

#### 图表 3: Per-block Histogram × Timestep (小多图)

展示 **频率直方图随 denoising timestep 的变化**, 使用小多图布局.

```
行: Aligned generation block (如 block 2, 3, 4)
列: Denoising timestep 采样点 (由 analysis_every 控制, 默认 5 列: step 50/100/150/200/250)
每个子图: 柱状图, 横轴 teacher block index, 纵轴 top-1 selection frequency
```
- 揭示不同去噪阶段的路由偏好变化
- 默认 `analysis_every=50` → 5 列; 若需更细 → `--analysis-every 25` → 10 列
- 每个子图统计量: 20 classes × 5 samples × 256 tokens = 25,600 tokens, 对 m=12 的直方图充分

#### 图表 4: Spatial Routing Map

```
对于指定图片, 将 (H/p × W/p) 的 patch grid 可视化:
每个 patch 的颜色 = top-1 selected teacher block index
每个 aligned generation block 一行, 每个 denoising step 一列
```
- 展示 token-wise routing 的空间模式
- 使用离散 colormap (m 种颜色)
- 参照 `run_token_choice_expert_heatmap.py` 的 spatial heatmap overlay 模式

#### 图表 5: Token Variance Bar Chart

```
横轴: Aligned generation block
纵轴: Token-wise routing weight variance (mean across samples)
多组 bar: 不同 model variants (token-wise vs blockwise, blockwise 的 variance ≈ 0)
```
- 量化 token-wise 路由的多样性

#### 图表 6: Routing Entropy

```
横轴: Aligned generation block
纵轴: Routing entropy (softmax weight 的信息熵)
多条线: 不同 model variants / checkpoints
```
- Entropy 高 = 路由均匀; Entropy 低 = 集中在少数 teacher blocks

---

## 5. Entry Script `run_mos_routing_analysis.py`

参照 `run_token_choice_expert_heatmap.py` 的结构: argparse CLI → 加载 config/checkpoint → 构建模型 → 采样循环 → 聚合 → 可视化.

### CLI 参数

与 `run_tokenwise_tsne.py` / `run_compute_flops.py` 一致，采用 `--ckpt` required 模式。config YAML 从 ckpt 路径自动推断 (`resolve_config_from_checkpoint`).

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|---|---|---|---|---|
| `--ckpt` | str | **是** | — | checkpoint 路径 (.pth) |
| `--seed` | int | 否 | 42 | 随机种子 |
| `--num-classes` | int | 否 | 20 | 随机采样的 ImageNet 类别数 |
| `--class-ids` | str | 否 | None | 逗号分隔的指定类别 ID (与 `--num-classes` 互斥) |
| `--samples-per-class` | int | 否 | 5 | 每个类别生成的样本数 |
| `--analysis-every` | int | 否 | 50 | 每 N 个 denoising step 采集一次路由权重 |
| `--plots` | str | 否 | all | 要生成的图表 (逗号分隔: `per_block_hist,all_blocks_hist,timestep,spatial,variance,entropy`) |
| `--vae-path` | str | 否 | None | 本地 VAE 路径 |
| `--overwrite` | flag | 否 | False | 覆盖已有输出 |

**默认值设计依据** (与现有分析脚本对齐):

- `num_classes=20`, `samples_per_class=5`: 与 `run_repa_dyna_heatmap.py` 一致
- `analysis_every=50`: 与所有现有分析脚本一致, `sample_steps=250` 时产生 5 个 timestep 采样点 `[50, 100, 150, 200, 250]`
- 默认统计量: 20 × 5 = 100 images × 256 tokens = **25,600 tokens/timestep point**, 对 12 个 teacher block 的频率直方图充分
- 需更细 timestep 粒度时: `--analysis-every 25` → 10 个点

### 单模型分析

```bash
python analyses/run_mos_routing_analysis.py \
  --ckpt outputs/ProMoE_TC_REPA_MoS_Naive_Choice_B/004_ProMoE_B_repa_MoS_naive_choice_b3_5/checkpoints/ckpt_step_500000.pth
```

```bash
python analyses/run_mos_routing_analysis.py \
  --ckpt outputs/ProMoE_TC_REPA_MoS_Naive_Choice_B/004_ProMoE_B_repa_MoS_naive_choice_b3_5/checkpoints/ckpt_step_500000.pth \
  --class-ids 0,207,971 \
  --samples-per-class 10 \
  --plots per_block_hist,spatial,variance
```

### 流程

```
1. 解析 CLI 参数
2. 从 ckpt 路径推断 config YAML (复用 t_SNE/checkpoint_utils.resolve_config_from_checkpoint)
3. 加载 runtime config (复用 t_SNE/checkpoint_utils.load_runtime_cfg)
4. 加载模型 + EMA 权重 (复用 t_SNE/sampling.build_model)
5. 构造 dummy_teacher_all_z (zeros, shape (m, N, T, D_teacher), 从 repa_config 读取 m 和 D_teacher)
6. 创建 MoSRoutingCapture + OnlineRoutingAggregator + PerTimestepRoutingAggregator
7. 设置 model.train(), 对每个 class 运行 denoising sampling loop:
   a. 采样初始噪声 z_T
   b. 对每个 denoising step t:
      - capture.enable()
      - model.forward(z_t, t, class_label, teacher_all_z=dummy_teacher_all_z)
      - capture.disable()
      - routing_data = capture.get_routing_data()
      - 若 t 在 analysis_steps 中:
        · aggregator.update(routing_data)
        · timestep_aggregator.update(routing_data, step_idx=t)
        · 对 spatial map 样本保存完整路由权重
      - 用模型输出继续 denoising
8. stats = aggregator.finalize(); timestep_stats = timestep_aggregator.finalize()
9. 生成可视化并保存
```

> **Note**: 不需要加载 teacher encoder. 路由权重仅由 `(x, c)` 决定, dummy `teacher_all_z` 仅用于触发 forward 中的路由代码路径.

### 输出路径

从 ckpt 路径解析 `run_root = ckpt_path.parent.parent`，输出保存到:

```
{run_root}/sample/step{N}/mos_routing/
```

即:
```
outputs/{model_name}/{custom_cfg_name}/sample/step{N}/mos_routing/
```

与现有分析脚本同级 (`t-sne/`, `heatmap/`, `flops_eval/`)。

实现方式: 在 `mos_routing/` 模块中新增 `resolve_mos_routing_output_dir(ckpt_path)`:

```python
def resolve_mos_routing_output_dir(ckpt_path: Path) -> Path:
    step = parse_checkpoint_step(ckpt_path)
    run_root = ckpt_path.parent.parent
    return run_root / "sample" / f"step{step}" / "mos_routing"
```

输出目录内文件示例:
```
mos_routing/
├── per_block_hist_block2.svg       # 图表 1: block 2 的 teacher block 频率直方图
├── per_block_hist_block3.svg
├── per_block_hist_block4.svg
├── all_blocks_hist.svg             # 图表 2: 所有 blocks 聚合直方图
├── per_block_hist_by_timestep.svg  # 图表 3: 小多图 (block × timestep)
├── spatial_routing_class000.svg    # 图表 4: spatial routing map
├── token_variance.svg              # 图表 5
├── routing_entropy.svg             # 图表 6
└── metadata.yaml                   # 运行参数记录
```

### 数据缓存策略

- **在线聚合**: `OnlineRoutingAggregator` / `PerTimestepRoutingAggregator` 累积均值/方差/频率/entropy, 不保存全量 tensor
- **Spatial map**: 仅对前几个 class 的前几个 sample 保存完整 `(T, m)` 路由权重 (由 `--class-ids` / `--num-classes` 和 `--samples-per-class` 控制)
- **中间结果**: 参照 `heatmap/expert_sampling.py` 的 `save_partial_*` / `load_partial_*` / `merge_*_partials` 模式, 支持多 GPU 分区处理后合并

---

## 6. Key Technical Considerations

### 6.1 触发路由代码路径

MoS 模型的 forward 中路由受守卫条件保护: `if self.training and teacher_all_z is not None`. 分析时需要:
- `model.train()` 激活路由代码路径
- 传入 dummy `teacher_all_z` = `torch.zeros(m, N, T, D_teacher)` 满足非 None 条件
- `torch.no_grad()` 包裹整个 forward — dummy tensor 导致的 MoS loss 无意义但无副作用
- **不需要加载 teacher encoder** — 路由权重完全由 `(x, c)` 决定

dummy tensor 的 shape 参数从 model 的 `repa_config` 读取:
- `m` = `num_teacher_blocks` (default 12 for dinov2-vit-b)
- `D_teacher` = `z_dims[0]` (default 768)
- `N`, `T` 从实际输入 batch 确定

### 6.2 EMA 权重

`build_model()` (from `t_SNE/sampling.py`) 已处理 EMA 权重加载 (`ema_model_state_dict`). 与采样/评估使用相同权重.

### 6.3 显存管理

- 每次只处理一个 batch, hook 中 `.detach().cpu()` 及时释放 GPU tensor
- `torch.no_grad()` 避免计算图和显存占用

### 6.4 AdaLNRouter 特殊处理

`MoS` 模型的 `AdaLNRouter` 返回 logits 而非 softmax 权重 (softmax 在 `compute_mos_repa_loss` 中执行). Hook 需要在捕获后手动 `F.softmax(logits, dim=-1)`.

### 6.5 PerBlockRouter 的 forward 时机

`MoS Choice PerBlock` 的路由发生在 block 输出后, hook 自然捕获正确的 post-block 路由.

### 6.6 MoS 模型无 `align_blocks`

`models_ProMoE_TC_repa_MoS.py` 对所有 block 做对齐 (无 `align_blocks` 属性). `MoSRoutingCapture` 在 `model_type == 'mos'` 时 fallback 为 `align_blocks = list(range(depth))`.

---

## 7. Implementation Order

| 阶段 | 内容 | 优先级 |
|---|---|---|
| **Phase 1** | `mos_routing/extract.py`: `MoSRoutingCapture` (hook + auto-detect + unified format) | P0 |
| **Phase 2** | `mos_routing/aggregate.py`: `OnlineRoutingAggregator` (block-level stats) | P0 |
| **Phase 3** | `mos_routing/plotting.py`: per-block histogram (图表 1) + all-blocks histogram (图表 2) | P0 |
| **Phase 4** | `run_mos_routing_analysis.py` + `run_mos_routing_analysis.md`: 单模型入口 | P0 |
| **Phase 5** | `aggregate.py`: per-timestep 统计 + `plotting.py`: per-block histogram × timestep 小多图 (图表 3) | P1 |
| **Phase 6** | `plotting.py`: spatial routing map (图表 4) + token variance bar chart (图表 5) | P1 |
| **Phase 7** | `plotting.py`: routing entropy (图表 6) | P2 |
| **Phase 8** | 多模型对比模式 (分别运行各 ckpt, 对比各自输出目录的聚合结果) | P2 |
| **Phase 9** | 更新 `analyses/README.md` 注册新入口 | P0 (随 Phase 4 一起) |
