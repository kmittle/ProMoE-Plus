# Base/Loss-Free Cross-Checkpoint Credit Gate

这个分析回答一个限定很窄的问题：当每个 routed block 的 token 数量已经明显更均衡时，不同 expert 每个 token 收到的去噪梯度是否仍然不均衡。

它只分析一对冻结的 Base 与 Loss-Free EMA checkpoint（训练步数相同），不训练模型，也不生成图片。Loss-Free 是诊断对照，不是本文的新方法；通过本 gate 也不能说明 FID 会提高。

## 为什么需要新 probe

Base probe 把无偏原型相似度的 argmax 当作原生路由。Loss-Free 会在“选择 expert”时加入不参与梯度的 `expert_bias`，但仍用无偏相似度作为 route weight。因此，无偏 argmax 与实际 route 不同是预期行为，不能记为错误。

新 probe 分别检查：

- route ID 等于“无偏相似度 + Loss-Free bias”的 argmax；
- route weight 等于所选 expert 的无偏相似度；
- 同一冻结输入重复计算得到完全相同的 route；
- 无偏 argmax 分歧只作为 Loss-Free 实际重路由量报告。

## 两种信用

设 token `t` 的 routed-expert 输入是 `x_t`，MoE 输出的 suffix gradient 是 `g_t`，原生 route weight 是 `w_t`。输出侧信用为：

```text
v_t = w_t * g_t
c_t = ||v_t||^2
```

对项目中的两层 `MoeMLP`，参数侧信用不做随机近似，而是精确计算逐 token 经验 Fisher trace：

```text
z_t = W_up x_t + b_up
a_t = GELU_tanh(z_t)
u_t = (W_down^T v_t) * GELU_tanh'(z_t)
p_t = ||v_t||^2 (||a_t||^2 + 1) + ||u_t||^2 (||x_t||^2 + 1)
```

两个 `+1` 分别是 `b_down` 和 `b_up` 的梯度贡献。runner 在读取 checkpoint 前用确定性 toy case 将闭式结果与逐 token autograd 对照，最大相对误差必须不超过 `1e-5`。

参数侧主指标是每个 image/block/sigma cell 内，active experts 的 `sum(c_t)` 与 `sum(p_t)` Spearman；先对同一图的 9 个 cell 求均值，再以 image 为单位 bootstrap。Base 和 Loss-Free 都必须满足 mean `>= 0.50`、单侧 95% LCB `>= 0.30`，每个 cell 至少有 3 个 active experts。

## 负载统计

全局负载不能用“每张图的 CV/Gini 再平均”判断。runner 对 block `1/5/11` 分别累加所有图和 sigma 的完整 12-expert count vector，再各自计算：

- CV `<= 0.20`；
- Gini `<= 0.12`；
- max/min `<= 2.0`；
- 相比同 case 的 Base，CV 和 Gini 每个 block 都至少下降 50%；
- 所有 expert 都被激活。

不同 block 的 expert ID 没有共同语义，任何统计都禁止跨 block 相加后再计算不均衡。

## 配对要求

脚本接受任意一对 Base（`ProMoE_TC_B`）与 Loss-Free（`ProMoE_TC_B_lossfree`）checkpoint，但两者必须是公平的一对，否则直接拒绝：

- 两个 checkpoint 都要带当前 `train.py` 写入的训练记录（trainer state v2）；更早、没有训练记录的 checkpoint 无法确认训练设置，会被拒绝。
- 训练步数、global seed、world size、global batch、`grad_mix` 和 learning rate 必须一致。
- sampler 记录必须完全相同，而且其中的数据集指纹要等于从 `--latent-root` 重算的有序 latent 指纹，也就是两个模型按同样的顺序读过同一份 encoded latent。
- 配置的 `model_name` 必须分别是 `ProMoE_TC_B` 和 `ProMoE_TC_B_lossfree`；模型结构检查还会确认只有 Loss-Free 一侧开启 `expert_bias`。

两个 checkpoint 的路径、SHA256、配置哈希和训练记录都会写进 protocol，之后每个阶段开始和结束时都会复核，文件被改动就停止。

## 阶段与门槛

case 由 `select_cases` 按固定 salt 从 1000 个类中确定性抽取，类别互不重叠：plumbing 8 张、discovery 32 张、confirmatory 64 张。每个阶段都在 Base 和 Loss-Free 两个 checkpoint 上各算一遍：

1. `plumbing`：只发布数值安全字段，隐藏 MSE 和信用结果；两个 checkpoint 都要通过安全检查。
2. `discovery`：只统计 token 数，判断 Loss-Free 的逐 block 负载门槛（上一节的 CV、Gini、max/min 与相对 Base 的下降幅度），暂不汇总信用效果。
3. `parameter`：对 discovery 顺序中的前 16 张图，在两个 checkpoint 上验证精确参数信用；通过后才汇总 discovery 的信用结果，两个 checkpoint 的信用门槛都要通过。
4. `confirmatory`：先过负载门槛，再查看信用结果；Base 和 Loss-Free 各自用自己的 discovery 结果检验排名稳定性。

门槛数值写在 `analyses/timestep_utility/credit_balance_batch.py` 和 `analyses/timestep_utility/credit_balance_cross_checkpoint.py` 中。任何门槛失败都停止，不能改阈值或跳过阶段。同一个 checkpoint 的 count 与 credit 两次独立计算，token 数必须逐 cell 完全一致。每个 case 与 summary 都有独立 seal，内容被修改或只剩单边文件时会直接失败。

## 运行

代码必须先通过检查、commit 并 push，且 `HEAD == origin/repa`。把 `BASE_RUN` / `LOSSFREE_RUN` 换成实际的配置名和 checkpoint 位置，先锁 protocol：

```bash
PY=/home/dev/miniforge3/envs/promoe/bin/python
PAIR=(
  --base-ckpt outputs/ProMoE_TC_B/BASE_RUN/checkpoints/ckpt_step_200000.pth
  --base-config configs/BASE_RUN.yaml
  --lossfree-ckpt outputs/ProMoE_TC_B_lossfree/LOSSFREE_RUN/checkpoints/ckpt_step_200000.pth
  --lossfree-config configs/LOSSFREE_RUN.yaml
  --output-dir analyses/archvied_analyses/YYYY-MM-DD/credit-balance-BASE_RUN-vs-LOSSFREE_RUN
)
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${PAIR[@]}" --prepare-only
```

再依次运行：

```bash
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${PAIR[@]}" --stage plumbing
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${PAIR[@]}" --stage discovery
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${PAIR[@]}" --stage parameter
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${PAIR[@]}" --stage confirmatory
```

每个阶段都要传同一组参数：protocol 记录了输入位置，任何一项变化都会与已锁的 protocol 不一致而失败。`--latent-root` 默认是 `/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz`。`--output-dir` 必须是仓库内、被 Git 忽略的目录（例如 `outputs/` 或 `analyses/archvied_analyses/` 下），否则参数解析阶段就会拒绝。GPU 固定为 4-7（`--devices` 只接受 `cuda:4,cuda:5,cuda:6,cuda:7`）。这是长任务，必须在当前 attached tmux session 的新 window 中运行。

## 与原封存 gate 的关系

这个脚本原本是封存 gate，只接受 Base seed-0 step-200K 与 Loss-Free `u=1e-2` step-200K 这一对，并依赖两份预注册文件和另一个 Base gate 的输出。那一对的 Loss-Free checkpoint、预注册文件和 Base gate 输出都已找不到，所以 2026-09-13 起改为通用分析：Base 一侧的结果由本脚本自己计算，不再读取其他 gate 的输出，也不再绑定预注册文件。它的结果不能当作原封存 gate 的重跑。

## 结论边界

完整通过只支持：在这一对 checkpoint 和锁定输入上，token-count balance 没有消除稳定的 routed-expert credit mismatch，而且 suffix-gradient energy 能预测真实参数侧经验 Fisher。

它不能支持：

- Loss-Free 本身是论文贡献；
- 梯度能量等于 expert 的语义知识或样本价值；
- frozen-checkpoint 相关性证明了因果优化收益；
- 通用梯度归一化足以构成 TPAMI 方法；
- 内部指标改善等于 OpenAI 50K-sample FID 改善。
