# `run_expert_function_consistency_probe_batch.py`

## 目的

这个入口验证一个只与 MoE 有关的机制假设：相对共享专家而言，某个路由
专家的函数如果更跟随图像内容而不是固定空间位置，它是否也更适合负责该
token 的去噪。它不引入 teacher，不做 DINO/REPA 表征对齐，也不直接声称
能够改善 FID。

确认实验固定使用 Base seed-0 的 EMA step 50K、block 3、三个噪声水平、
四个单 patch 平移和 24 张此前探针没有使用过的 ImageNet 图像。每个
图像/噪声/平移 cell 抽取 8 个 token，并把每个 token 分别强制给全部
12 个等宽路由专家。强制路由只改变专家 ID，保留原生 top-1 gate weight，
所以候选之间的激活 FLOPs 相同。每个 native/candidate 对在同一次前向中运行，
总 batch 仍不超过 24；这避免不同专家批次组成带来的 BF16 no-op 数值偏差。

## 锁定门槛

统计单位是图像，不是 token。每张图先聚合 96 个 token 观测，再对 24 个
图像均值做 200,000 次 cluster bootstrap。通过需要同时满足：

- 主指标的图像均值 Spearman 不低于 0.10，且 95% bootstrap CI 下界大于 0；
- 至少 15/24 张图的主指标为正，三个 sigma 的图像级均值分别为正；
- 主指标相对原生 router affinity 的图像均值 Spearman 增益不低于 0.05，
  且增益的 95% bootstrap CI 下界大于 0；
- 每张图至少 95% 的 token 有有效秩相关；
- 原生 gate weight 为正，并通过 no-op 与 forced/unforced 数值检查。

阈值、样本、latent、seed 和 checkpoint SHA256 都不是 CLI 参数。runner
会同时校验 canonical checkpoint 与本地权重副本，锁定 24 个 latent 的内容
SHA256 和配置。写协议前，它会在 CPU 上实际构建目标模型，再动态记录当时已
加载的全部项目内源码、静态协议文件及对应第三方 distribution 版本；这里锁定
的是“目标模型构建的 import 闭包”，不夸大为 CUDA kernel 等完整运行时闭包。
它先原子写入 `protocol.json` 和 `protocol.sha256`，之后才允许生成第一个 case；
每个 case 也保存协议 SHA256，已有协议或 case 与当前源码、数据或执行契约不一致
时会直接拒绝。

## 运行

先只锁定并检查协议：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_expert_function_consistency_probe_batch.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_50000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_50000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --devices cuda:4,cuda:5,cuda:6,cuda:7 \
  --output-dir /home/dev/promoe-probes/function-transport-gate-v1 \
  --prepare-only
```

确认协议后去掉 `--prepare-only`。四个独立进程各占一张 GPU，每张卡按固定
顺序处理六个 case。单个 case 先原子写入 pending JSON；worker 在 checkpoint、
config、manifest、protocol、源码和全部 latent 的运行后 hash 复验通过后，才给 pending 写入内容绑定的
seal，并把该组结果原子发布为正式 JSON。进程中断后可用完全相同的命令恢复：已有
正式结果会完整复验；只有已通过 worker 尾验的 sealed pending 才会复用，未封印的
pending 一律丢弃并重新计算。

## 输出与裁决

```text
/home/dev/promoe-probes/function-transport-gate-v1/
  protocol.json
  protocol.sha256
  cases/01_*.json ... cases/24_*.json
  summary.json
```

`summary.json` 在再次复验全部锁定输入后原子发布；已有正式 summary 只能验证复用，
不能被不同的重算结果覆盖。它保存全部安全检查、机制门槛、每个 sigma 和每张图的统计。
任何一项失败都会写完汇总并以状态 1 退出；失败意味着停止这条假设，而不是
根据确认集结果修改指标或门槛。即使通过，也只能据此设计近等 FLOPs 的 MoE
方法，最终性能结论仍必须来自本项目的 OpenAI 50K FID/IS 协议。
