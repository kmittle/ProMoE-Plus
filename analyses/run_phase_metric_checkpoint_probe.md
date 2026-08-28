# Phase-Metric 50K checkpoint gate

## 目的

这个探针回答一个窄而关键的问题：Phase-Metric 是否真的把 token 路由给了更合适的专家。

它不是从 Base checkpoint 继续训练，也不把 50K MSE 当作最终生成性能。候选模型和 Base 对照都必须使用 seed 0，从 step 0 独立训练到 50K。探针只冻结这两个 fresh checkpoint，在相同图像、噪声和算力下做路由反事实。正式结论仍然来自 fresh 0→300K/500K 训练和 OpenAI evaluator。

## 预注册设计

- 复用已经锁定的 8 张 discovery 图像和 24 张 confirmatory 图像。
- 每张图像测试 sigma 0.2、0.5、0.8，以及第 2、6、12 个 Transformer block（代码索引 1、5、11）。
- 对候选 checkpoint 做 `selection × weight` 的 2×2 分解：Phase/Base 选择专家，Phase/Base 产生输出权重。
- 加入 shuffled-phase 对照。它在同一图像的三个 noised state 之间轮换 phase timestep，保留 phase timestep 的直方图，但打乱 noised state 与 phase timestep 的对应关系。
- 只在 Phase 路由和 Base 路由不同的 token 上做精确反事实。替换专家 ID 时保持 Phase 原生路由权重不变，因此激活专家数和计算量不变。
- 使用同一步数、同一 seed 的 fresh Base checkpoint 比较总体去噪 MSE。
- discovery 通过后，程序才允许执行 confirmatory。confirmatory 通过后，Phase-Metric 才能继续作为正式 500K 候选。

门槛在唯一的 canonical 文件 `analyses/routing_metric/manifests/phase_metric_50k_gate_v1.json` 中固定，入口同时锁定该文件的 SHA256，不接受替代 `--spec`。核心要求包括：Phase 必须产生足够多的实际 route flip；精确反事实中 Phase 路由的平均收益、胜率和 selection 主效应不能为负；Phase 必须优于 Base 路由、shuffled phase 和独立 Base checkpoint；confirmatory 还要求关键收益的单侧 95% bootstrap 下界不小于 0。

## 运行

两个 `ckpt_step_50000.pth` 都生成并完整写盘后运行：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_phase_metric_checkpoint_probe.py \
  --candidate-ckpt outputs/ProMoE_TC_B/004_ProMoE_B_phase_metric/checkpoints/ckpt_step_50000.pth \
  --base-ckpt outputs/ProMoE_TC_B/004_ProMoE_B_phase_metric_base_s0/checkpoints/ckpt_step_50000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/phase-metric-50k-gate-v1 \
  --split discovery \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --num-threads 4
```

只有 discovery 命令返回 0 且 `gate.passed` 为 `true` 时，才运行：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_phase_metric_checkpoint_probe.py \
  --candidate-ckpt outputs/ProMoE_TC_B/004_ProMoE_B_phase_metric/checkpoints/ckpt_step_50000.pth \
  --base-ckpt outputs/ProMoE_TC_B/004_ProMoE_B_phase_metric_base_s0/checkpoints/ckpt_step_50000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/phase-metric-50k-gate-v1 \
  --split confirmatory \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --num-threads 4
```

如果权重需要从 NAS 复制到本地，可分别传入 `--candidate-weights-ckpt` 和 `--base-weights-ckpt`。本地文件必须与对应 canonical checkpoint 逐字节相同。

## 输出与解释

输出目录包含：

- `protocol.json` / `protocol.sha256`：checkpoint、fresh step-0 日志前缀、配置、样本、源代码和运行环境的锁定快照。
- `discovery-result.json` / `discovery-result.sha256`：discovery 原始结果和门控结论。
- `confirmatory-result.json` / `confirmatory-result.sha256`：通过 discovery 后才可能生成的确认结果。

同一输出目录由进程锁串行化。结果 JSON 和 SHA256 先作为一对 pending 文件校验，再发布为最终文件；若进程恰好在两次原子重命名之间退出，下次相同调用会先验证并恢复这对文件，而不会重跑或覆盖结果。

任一门槛失败时程序返回非零退出码。discovery 通过只产生 `authorize_confirmatory`，confirmatory 通过才产生 `authorize_continue_training`。失败时结论是停止或重构 Phase-Metric，而不是继续消耗到 500K。即使门控通过，它也只证明“路由机制方向成立”，不能代替 300K/500K FID、sFID、IS、Precision 和 Recall。

## English summary

This locked probe compares two independently trained, fresh step-50K checkpoints. It separates expert selection from router-output weighting, runs fixed-weight equal-compute expert counterfactuals on route-disagreement tokens, and checks native phase correspondence against a shuffled-phase control. Passing discovery unlocks confirmatory execution; passing both authorizes continuation, but does not replace the formal 300K/500K generation evaluation.
