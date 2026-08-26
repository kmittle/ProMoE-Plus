# `run_timestep_utility_probe_batch.py`

## 目的

这是 natural-input timestep-utility 机制诊断的预注册批处理入口。它在任何新样本结果产生前，同时锁定 8 个 discovery images 和另 24 个 confirmatory images。两组都排除以前 routing probes 使用过的 78 个 labels，且类别、latent 和随机种子都由 SHA256 规则确定。

协议固定 Base seed-0 step 100K、EMA 权重、0-index MoE blocks `1/5/11`（12-block Base 的早/中/晚 MoE 层）、sigmas `0.2/0.5/0.8`、每个 cell 8 个固定 token、2 个 weight-sensitivity tokens，以及 12 个同宽专家的完整反事实网格。主结果保持原生 route weight 与激活 FLOPs 不变。prepare 阶段会先构建真实模型并验证深度、MoE block、`top_k=1` 和 identity router weights，验证失败时不会写入 protocol。

## 两个独立 gate

- `routing_accuracy_gap_passed`：要求 native regret 足够大、native 很少为 oracle、router/utility 相关低，并且严格保持原生专家计数的联合重分配在真实 forward 上稳定改善 MSE。只有它通过，才允许设计 utility-aware MoE routing。
- `stage_structure_passed`：额外要求 utility ranks 随 sigma 的变化明显大于 router ranks 的变化。只有它也通过，才允许把后续方法叙事写成 timestep-conditioned routing。

最终统计单位是 image。置信区间对 image 做 200,000 次 cluster bootstrap；不能把 token 当独立样本扩大显著性。

## 先锁协议

```bash
python analyses/run_timestep_utility_probe_batch.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_100000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_100000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/natural-timestep-utility-gate-v1 \
  --prepare-only
```

`protocol.json` 会记录 checkpoint/config/manifest/latent/source SHA256、环境、GPU 分配、全部阈值和两个 split。协议存在后，任何输入或源码变化都会使运行拒绝继续。

## 运行

```bash
python analyses/run_timestep_utility_probe_batch.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_100000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_100000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/natural-timestep-utility-gate-v1 \
  --split discovery
```

只有 discovery 的安全控制和 routing gap 通过后，才运行相同命令并改成 `--split confirmatory`。代码会重新读取 discovery cases、重算 gate，并核对 summary 与每个 case 的 SHA256；缺少 summary、内容不一致或 gate 未通过时，confirmatory 会在启动 worker 前拒绝执行。

四个 worker 固定使用 `cuda:4,cuda:5,cuda:6,cuda:7`，每张卡顺序处理自己的 cases。新结果先写入带内容哈希的 pending 文件；checkpoint、config、manifest、protocol、latent 和源码在 worker 尾部全部复核后才 seal 并原子发布。联合 assignment 中原生路由相对自身也必须保持 MSE 与输出精确不变，否则安全门失败。
