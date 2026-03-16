# `compute_flops.py`

## 作用

`compute_flops.py` 是 `compute_FLOPs/` 目录下用于直接启动的入口脚本。
它会根据 checkpoint 路径解析对应的 YAML 配置，使用 YAML 中的采样步数设置进行采样，并统计 FLOPs、激活参数量以及专家激活频率。

## 统计内容

- 采样过程中的 conditional forward FLOPs
- 模型激活参数量统计
- 每个 MoE block 的整体专家激活频率
- 所有已追踪 block 的平均专家激活频率
- 每隔 `N` 个去噪 step 保存一次的分 step 专家激活频率

## 输出目录

结果默认保存在：

```text
outputs/<model_name>/<config_name>/sample/step<ckpt_step>/flops_eval/
```

典型输出包括：

- `flops_result.txt`
- `expert_freq_block_<block_idx>.png`
- `expert_freq_average.png`
- `step-050/`、`step-100/` 等分 step 子目录

每个 `step-xxx/` 子目录中会包含：

- 当前 step 下各个 block 的专家激活频率柱状图
- `expert_freq_average.png`
- `expert_frequencies.txt`

## 主要参数

- `ckpt`：必需，checkpoint 路径
- `--num_samples_per_class`：每个 ImageNet 类生成多少个样本，默认 `5`
- `--seed`：随机种子，默认 `0`
- `--guide_scale`：CFG scale，默认 `1.0`
- `--save_every_steps`：每隔多少个去噪 step 保存一次分 step 专家频率，默认 `50`

## 使用示例

```bash
python compute_FLOPs/compute_flops.py \
  outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --num_samples_per_class 5 \
  --guide_scale 1.0 \
  --save_every_steps 50
```

## 当前目录组织

`compute_FLOPs/` 根目录下只保留直接运行的入口脚本：

- `compute_flops.py`

其余可复用模块按功能拆分为子目录：

- `config/`：checkpoint、YAML、模型构建相关工具
- `tracking/`：专家激活与激活参数量追踪
- `profiling/`：FLOPs 统计
- `visualization/`：专家频率可视化
