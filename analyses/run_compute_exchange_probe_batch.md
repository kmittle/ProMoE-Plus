# 同专家计算交换诊断

`run_compute_exchange_probe_batch.py` 在冻结的 ProMoE-TC Base 200K EMA 权重上检验一个具体问题：保持目标 block 的 top-1 expert ID、route weight、shared expert 和 uncond expert 不变时，能否把同一 expert 的一次 routed FFN 计算从低价值 token 转给高价值 token，并降低完整 suffix 的去噪误差。

这只是进入训练前的可证伪诊断，不是生成指标、训练收益或创新性结论。它不使用 DINO，也不做表征对齐。

## 干预

对路由到同一 expert `e` 的 donor `d` 和 receiver `r`：

```text
donor:   remove w_d E_e(h_d)
receiver: add w_r E_e(h_r + w_r E_e(h_r))
```

每个 expert 独立设置 `k_e=min(floor(0.1*n_e+0.5), floor(n_e/2))`，donor 和 receiver 数量相同且不重叠。因此每个 expert 的逻辑 FFN pass 数逐项不变。诊断代码为了构造反事实会重算 expert 输出，所以这里的 isoFLOP 是候选方法的解析语义，不是诊断脚本自身的实际耗时。

候选池固定为 64 个完整的 expert-wise assignment。四个可部署性对照在同一候选池上各自强制选择一个候选：first-order downstream utility、matched random、router margin 和 within-expert rolled utility。它们不共享 first-order 的停止决定，因此对照差异只来自排序信号。Exact oracle 只报告上界，不能单独授权训练。

## 锁定协议

入口只接受 GPU 4-7、固定 Base-200K checkpoint 哈希和已经 push 的 clean `origin/repa`。它复用旧 cycle gate 的 8 张 plumbing 图做纯安全检查，并从所有旧 manifest 未使用的类别中确定性选择 24 张 discovery 图和 48 张 confirmatory 图。plumbing 结果不会保存任何 compute-exchange efficacy。

先创建协议：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_compute_exchange_probe_batch.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_200000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/within-expert-compute-exchange-base200k-v1 \
  --prepare-only
```

协议准备完成后，在当前 tmux session 的新 window 中依次运行：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_compute_exchange_probe_batch.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_200000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/within-expert-compute-exchange-base200k-v1 \
  --split plumbing
```

只有 `plumbing-summary.json` 的 safety gate 通过，`discovery` 才会解锁；只有 discovery 全部门槛通过，`confirmatory` 才会解锁。候选 batch 的 native 行与另一次同 batch-shape 的 no-op forward 比较；单样本与双样本 forward 的数值差只记录为诊断，不进入 safety gate。后两步只需把 `--split` 分别改成 `discovery` 和 `confirmatory`。任何门失败都必须停止，不能在看到该 split 后放宽阈值或更换 quota。

## 主要门槛

Discovery 要求 image-level mean gain 至少 `1e-4`、单侧 95% LCB 大于 0、至少 16/24 张图为正，并在 per-transferred-pass gain 上显著优于 matched random 和 router margin。Exact oracle 至少要在 18/24 张图上存在正空间。

Confirmatory 还要求 mean gain 的 LCB 至少 `5e-5`、至少 32/48 张图为正、明显优于 rolled utility、恢复至少 25% 的 oracle gain，并且三个 block 与三个 sigma 的点估计全部为正。主要单侧检验使用 Holm-Bonferroni 校正。

即使 confirmatory 通过，也只授权训练一个轻量 forward-only scorer 并在第二个 checkpoint 或 seed 上复核，不直接授权 501K ImageNet 长训。
