# Analyses Usage

`analyses/` 目录只放任务入口脚本；具体分析逻辑和可复用工具模块放在对应子目录中，例如 `analyses/t_SNE/`。

## Current Entry Scripts

### `run_tokenwise_tsne.py`

用途：

- 对指定 checkpoint 做 token-wise 的路由 t-SNE 分析。
- 自动从 checkpoint 路径反推对应 YAML 配置文件。
- 默认随机种子为 `42`。
- 默认从 1000 个 ImageNet 类中随机抽取 20 个类。
- 采样时固定 `CFG=1.0`，采样步数等超参数从 YAML 读取。
- 每隔一定去噪步数抓取各个 MoE block 的 token 表征，并以 token 选择的 top-1 conditional expert 作为标签绘制 t-SNE。

默认输出位置：

- `outputs/<model_name>/<config_stem>/sample/step<step>/t-sne/token-wise/`

基础用法：

```bash
python analyses/run_tokenwise_tsne.py \
  --ckpt outputs/ProMoE_TC_S/004_ProMoE_S/checkpoints/ckpt_step_500000.pth
```

常用参数：

- `--ckpt`: 必填，checkpoint 路径。
- `--seed`: 随机种子，默认 `42`。
- `--num-classes`: 随机抽取多少个 ImageNet 类，默认 `20`。
- `--class-ids`: 手动指定类 id，格式如 `1,5,23`；提供后将覆盖随机抽样。
- `--analysis-every`: 每隔多少个去噪 step 做一次分析，默认 `50`。
- `--analysis-batch-size`: 每个 worker 一次并行分析多少个类，默认 `4`。
- `--perplexity`: 手动指定 t-SNE perplexity；默认自适应。
- `--overwrite`: 如果目标 SVG 已存在则重新生成。

示例：

```bash
python analyses/run_tokenwise_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --seed 42 \
  --num-classes 20 \
  --analysis-every 50
```

```bash
python analyses/run_tokenwise_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --class-ids 7,12,56,207 \
  --analysis-batch-size 2 \
  --overwrite
```

运行前提：

- 需要可用 CUDA GPU。
- 使用哪些 GPU 会优先从 YAML 中的 `sample_gpu_ids` 或 `gpu_ids` 读取。
- 需要安装仓库依赖，以及绘图依赖 `matplotlib` 和 `scikit-learn`。

## Subdirectories

### `t_SNE/`

这里存放 `run_tokenwise_tsne.py` 用到的工具模块，不直接作为任务入口执行。当前包含：

- `checkpoint_utils.py`: checkpoint、YAML、输出目录、GPU 配置解析。
- `imagenet_utils.py`: ImageNet 类名与类别抽样工具。
- `model_registry.py`: 统一模型注册表。
- `routing_capture.py`: MoE block 路由与 block 输出采集。
- `sampling.py`: 采样与分析 step 调度。
- `plotting.py`: token-wise t-SNE 绘图与 SVG 保存。
