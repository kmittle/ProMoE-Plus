# RCL 与专家责任批量检查

这个入口只回答一个问题：ProMoE 的 RCL 在整理当前专家分组时，是否会通过同一个 cosine 分数把 routed expert 的输出强度推向更高的去噪误差。

它不训练新模型，也不使用 DINO 或 teacher 做表征对齐。即使全部通过，也只说明找到了一个值得修的 ProMoE 机制问题，不能直接证明新方法有效或有创新性。

## 检查怎样做

程序使用 Fresh Base step 300K 的在线权重，不使用 EMA。36 张 query 图沿用 Fresh 路由审计在看结果前固定的图片，另外固定选择 256 个不与 query 类别重合的 support 类别。

每张 GPU 对应一组 64 张 support 图，用一个固定的、近似训练时每个 rank 本地 batch 的诊断 batch。其中固定 6 张改成 unconditional，剩余 58 张用于计算 RCL 的直接 prototype 梯度。每张 support 图有自己预先固定的 logit-normal 噪声时刻。四张卡算完后，程序像 DDP 一样把四份梯度取平均；所有 query 都使用这一份 256 图平均梯度。

每张 query 检查 6 个 MoE block 和 3 个噪声位置。程序固定原 expert ID，分别沿下面几种方向移动 prototype：

1. 不移动，用来检查程序本身是否改变结果。
2. 沿 diffusion 梯度下降，用来检查正确符号是否真的降低 MSE。
3. 沿正确 support 分组的 RCL 梯度下降。
4. 把 support 的 token-expert 对应关系打乱 16 次，但严格保持每位 expert 的 token 数，再分别下降。

所有更新使用相同的 Frobenius 步长，并恢复每个 prototype 原来的范数。正确 RCL 还会运行半步，检查一阶解释和精确 MSE 是否相符。这里只隔离 RCL 直接更新 prototype 的路径；RCL 通过 hidden state 影响前面网络的另一条梯度不在本检查结论里。

## 三道门

- `plumbing` 用 4 张图，只检查哈希、no-op、固定 expert ID、梯度恒等式和半步线性关系，不公布汇总效果。
- `discovery` 用事先固定的 8 张图。它不通过时程序不会打开确认集。
- `confirmatory` 使用另外 24 张图，要求正确分组相对保持负载的 shuffle 同时更改善独立 query 的分组几何、又更损害精确去噪 MSE，并覆盖至少 4 个 MoE block 和 2 个噪声位置。

置信区间以 query 图片为单位计算。token、block、噪声位置和 16 次 shuffle 都不会被当成额外独立样本。结论只针对这一个预先固定的四卡 support 更新，不会声称已经覆盖所有可能的训练 batch。

## 执行

只有分析分支通过 `$check 1`、提交并推送，而且 Fresh 300K wrapper 已暂停、GPU 0-3 空闲后才能准备协议：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_rcl_responsibility_probe_batch.py prepare \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_fresh_routing_audit_s0/checkpoints/ckpt_step_300000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

命令会打印并保存 `output_dir`。推荐用一条命令跑完整门控，这样每张 GPU 只加载一次模型、只计算一次 support 梯度：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_rcl_responsibility_probe_batch.py run-gate \
  --output-dir <output_dir>
```

中断后可以重跑同一命令。已有 case 会先逐项验哈希再复用。`run-split` 只用于单独修复某一阶段；分开调用会重新加载模型，因此正常运行不应使用它。
`run-gate` 或 `run-split` 若得到 `passed: false`，命令会返回非零退出码；这能让独立调用者发现门控失败，而 relay 会把这种结果记录为“门未通过”并按脚本策略恢复训练。

## 结果边界

协议会封存 checkpoint、训练 run ID、从 step 0 开始的记录、数据清单、query/support latent 和每张 support 图的噪声时刻、Git 提交、配置和源码哈希。程序还会精确核对 manifest 的全部字段。任何一项变化都会拒绝混用结果。

确认集通过后，下一步仍要先做新颖性复审，再提出利用该冲突的 MoE 方法。正式对照必须从 step 0 训练，并包含正确依据、保持统计量不变的打乱依据和去掉新机制三条对照。
