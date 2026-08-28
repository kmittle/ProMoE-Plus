# `run_phase_default_probe.py`

## 目的

该诊断回答一个窄而关键的问题：Default MoE 用一个全局历史均值代替未执行专家输出时，是否因为扩散阶段混合而产生系统性误差。它不是新模型训练，也不报告 FID。

脚本固定使用 Base-200K 的 EMA 权重。16 张校准图只根据原生 top-1 路由实际执行的专家输出构造均值；32 张独立确认图才用于评价。对每个 block、sigma 和图像，脚本穷举少量固定 token 的 12 个专家输出，并比较四种近似：

1. `zero`：未执行专家没有梯度，是当前稀疏反传的参照。
2. `global`：每个 block、expert 只有一个跨阶段均值，对应 Default MoE 的核心假设。
3. `phase`：每个 block、expert、扩散阶段各有一个均值。
4. `shuffled_phase`：循环错配阶段标签，参数量和样本量与 `phase` 相同，用于排除“只是多放了几组 buffer”的解释。

## 指标

- 未选专家输出的相对平方误差。
- 未选专家 local dense score-gradient 与真实值的余弦相似度。
- 将 score-gradient 通过归一化 cosine router 的 Jacobian 映射后，prototype-center 梯度与真实值的余弦相似度。

统计先在每张图的 15 个 block/sigma cell 内求均值，再对 32 张图做 200,000 次 image bootstrap，不能把 token 当独立样本。预注册同时约束 phase sketch 的绝对误差/梯度余弦、相对 global 的增益和相对 shuffled-phase 的增益，避免“两个都很差但差值显著”也被判为通过。具体门槛和 latent SHA256 位于 `analyses/phase_default/manifests/phase_default_gate_v1.json`。只有全部门槛通过，才允许实现 phase-conditioned default 的 fresh 0→500K 训练。

这里测量的是冻结 checkpoint 下、使用当前专家输出均值的上界。即使通过，也不能直接证明在线 EMA 或最终 FID 会改进；失败则说明该路线在投入正式训练前需要停止或重构。

## 运行

```bash
python analyses/run_phase_default_probe.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_200000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/phase-default-gate-base200k-v1 \
  --device cuda:0
```

输出目录包含 `protocol.json`、`protocol.sha256`、`result.json` 和 `result.sha256`。已有结果不会被覆盖。
