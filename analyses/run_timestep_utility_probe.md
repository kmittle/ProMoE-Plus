# `run_timestep_utility_probe.py`

## 目的

该脚本检查 ProMoE 的 prototype router 是否真的选中了对当前 token、当前噪声阶段最有用的专家。输入是训练分布中的 VAE posterior sample 与固定 Gaussian noise，不做平移、翻转、teacher feature 或表征对齐。

它对固定 token 穷举同宽的 12 个 routed experts，并用完整模型的 denoising MSE 定义真实 utility。主要干预保持 token 原生 top-1 route weight 不变，因此只替换 expert identity，激活专家数和推理计算量不变。

## 三类证据

1. 路由准确性：报告 native-is-oracle rate、prototype affinity 与真实 utility 的 Spearman 相关、以及 native 相对 oracle 的 regret。
2. 容量约束：用 Hungarian assignment 构造三种联合路由。其中 `native_capacity_oracle` 严格保持 sampled tokens 在每个专家上的原生计数，再用一次真实 multi-token forward 验证 MSE，而不是只相加单 token 收益。
3. 阶段结构：同一 latent、noise 和 token 在多个 sigma 下重复测量，报告 utility rank Spearman、专家两两次序反转率，以及 native/oracle expert 的切换率。

`candidate` 和 `unit` route-weight 口径只在预先固定的少量 token 上做敏感性分析。它们不能取代保持原生 weight 的主结果。

## 示例

```bash
python analyses/run_timestep_utility_probe.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_100000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_100000.pth \
  --latent /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz/n01440764/example.latent.npz \
  --label 0 \
  --sigmas 0.2,0.5,0.8 \
  --block-indices 1,5,11 \
  --num-token-probes 8 \
  --sensitivity-token-count 2 \
  --exact-batch-size 24 \
  --device cuda:4
```

默认输出位置为：

```text
outputs/<model_name>/<config>/sample/step<step>/timestep_utility_probe/
```

该脚本只是 frozen-checkpoint 机制诊断。单图结果、oracle 上界或局部 MSE 变化都不能当作 FID 改进，也不能直接授权训练新方法。正式结论需要在运行前锁定 case manifest、checkpoint、源码 hash、阈值和 image-cluster bootstrap 规则。
