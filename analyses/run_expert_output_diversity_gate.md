# 专家输出正则 50K 止损检查

这个检查只回答一件事：tau=5 的专家输出正则，是否真的让不同专家学成了不同的函数。

旧版本比较的是“每位专家各自收到的 token 的平均输出”。这有一个漏洞：即使专家本身完全一样，只要它们收到的 token 不同，平均输出也会不同。另一个漏洞是，专家可以单纯把输出整体放大，让欧氏距离变大，但没有学到更不同的功能。

这里把 Base 和 tau=5 的专家喂入完全相同的 hidden state，再比较它们的输出。这样，输入差异不能冒充专家差异。检查同时报告归一化后的距离、有效秩和输出大小，因此也能识别“只放大输出”的情况。

## 固定设置

- 两个 checkpoint 必须是同一个 step，正式止损点是 50K。
- 两个 checkpoint 必须来自固定的 Base `_v2` 和 tau=5 `_v2` 输出目录；checkpoint 内必须记录干净、已 push 的代码、配置、数据顺序和训练进度。两组的 seed、batch 顺序、数据身份、公共训练源码、软件环境和 GPU 型号必须一致，但 run ID 必须不同。`train.py` 只允许预先锁定的来源记录注册差异。
- 使用已经在 tau=5 训练前锁定的 Fresh Base manifest 中 8 张 discovery 图像。
- 使用 sigma `0.2/0.5/0.8` 和全部六个 MoE block `1/3/5/7/9/11`。
- 每个图像、噪声和 block 固定抽取 32 个 token。
- Base 专家和 tau=5 专家分别在 Base hidden state 与 tau=5 hidden state 上运行，再对两个方向取平均。
- 四张 GPU 各处理两张图，结果按图像配对汇总；置信区间以图像为单位重采样。

## 预先固定的继续条件

必须同时满足：

1. 与训练目标直接对应的 routed-pooled 专家距离至少增加 10%，且配对置信区间下界大于 0；真正参与训练的 `mean(exp(-distance/5))` 也必须下降，且配对置信区间上界小于 0。
2. 相同输入、去掉输出尺度后的专家距离至少增加 3%，且配对置信区间下界大于 0。
3. 专家输出有效秩的平均变化为正，并且至少 6/8 张图为正。
4. 相同输入下的输出 RMS 不能增加超过 15%，防止靠放大数值作弊。
5. tau=5 在每张图、每个噪声和每个 MoE block 上都必须至少使用两个专家；只有一个专家收到 token 就直接视为路由塌缩。
6. 路由归一化熵最多下降 0.02，最大专家占比最多增加 0.03。
7. 固定图像上的去噪 MSE 最多恶化 3%。

任一条件失败，就停止 tau=5，不继续消耗到 300K。全部通过也只说明机制值得继续，不代表生成质量已经改善；最终结论仍必须来自 300K/500K、CFG 1.0/1.5、每组 50K 图片的 OpenAI FID/IS。

## 运行命令

先把两个 50K checkpoint 复制到本地盘，避免分析反复读取 NAS，然后暂停 tau=5 训练并使用它的四张卡。工具会逐字节比较 NAS 原文件和本地副本的 SHA-256；如果复制错了文件或文件不完整，它会直接停止：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_expert_output_diversity_gate.py \
  --base-ckpt outputs/ProMoE_TC_B/004_ProMoE_B_fresh_routing_audit_s0_v2/checkpoints/ckpt_step_50000.pth \
  --variant-ckpt outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_output_tau5_fresh_s0_v2/checkpoints/ckpt_step_50000.pth \
  --base-weights-ckpt /home/dev/promoe-probes/tau5-gate/base_step50000.pth \
  --variant-weights-ckpt /home/dev/promoe-probes/tau5-gate/tau5_step50000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --devices cuda:4,cuda:5,cuda:6,cuda:7 \
  --output analyses/archvied_analyses/2026-08-31/expert-output-tau5-50k/result.json
```

结果里的 `comparison.passed` 是总判定；每个条件的观测值和门槛都保存在 `comparison.checks`，完整逐图、逐噪声、逐 block 数值保存在 `cases`。

正式模式会拒绝任何不同的 checkpoint、manifest、图像数量、sigma、block、token 数、GPU、线程或 bootstrap 设置。只有调试工具时才可加 `--exploratory`；探索模式允许改设置，但会把 `comparison.passed` 写成 `null`，不能拿来决定是否继续训练。
