# Base/Loss-Free Cross-Checkpoint Credit Gate

这个分析回答一个限定很窄的问题：当每个 routed block 的 token 数量已经明显更均衡时，不同 expert 每个 token 收到的去噪梯度是否仍然不均衡。

它只分析冻结的 Base 与 Loss-Free step-200K EMA checkpoint，不训练模型，也不生成图片。Loss-Free 是诊断对照，不是本文的新方法；通过本 gate 也不能说明 FID 会提高。

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

## 封存依据与顺序

有效预注册由两个不可变文件共同定义：

```text
v1 SHA256 59ce95f39220511c510b589b78e69b0139c961aaa1d3e4e3f013c16312565a43
v2 SHA256 04ced5b1cebf371153c33c4f7b9cf703b58d430ee504d8d52c083a186f254b57
```

case manifest 复用 Base protocol：

```text
SHA256 9c25bd0144228e921be1a5491dafa32299356f5af00e0a5cc15d857a1eeef096
```

runner 会先逐字段核对两个预注册文件中的 seed、batch、阈值、固定路径和模型设置，重放训练端 cache-first 的 latent 枚举与哈希规则，同时要求训练采用的缓存序列与实际磁盘完整排序清单完全一致，再把所得 dataset identity 与 checkpoint 中的 sampler provenance 逐字段比较。随后才把两个预注册 hash、Base protocol hash、Base/Loss-Free checkpoint hash、Loss-Free trainer/sampler/RNG provenance、配置、latent manifest、项目源文件 hash、Git commit 和运行环境写入新 protocol。worker 在实际加载 checkpoint 前后及读取每个 latent 前后都会复核 hash，阶段汇总后还会再次复核全部输入。每个 case 与 summary 都有独立 seal，内容被修改或只剩单边文件时会直接失败。

阶段必须按以下顺序运行：

1. `plumbing`：8 张图，只发布数值安全字段，隐藏 MSE 和信用结果。
2. `discovery`：32 张图，只判断 Loss-Free 的逐 block 全局 count gate，暂不汇总信用效果。
3. `parameter`：对 discovery 顺序中的前 16 张图，在 Base 和 Loss-Free 上验证精确参数信用；通过后才汇总 discovery 的 count-adjusted output credit。
4. `confirmatory`：64 张类不重叠的新图；只有此前所有 gate 通过才解锁，先过 count gate，再查看信用结果。

任何门槛失败都停止，不能改阈值或跳过阶段。step-10K 等中间 checkpoint 只能放在另一个明确标为 trajectory diagnostic 的协议中，不能替代这里固定的 step-200K 结论。

## 运行

代码必须先通过检查、commit 并 push，且 `HEAD == origin/repa`。Loss-Free step-200K checkpoint 存在后，先锁 protocol：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_cross_checkpoint.py \
  --prepare-only
```

再依次运行：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_cross_checkpoint.py \
  --stage plumbing

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_cross_checkpoint.py \
  --stage discovery

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_cross_checkpoint.py \
  --stage parameter

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_cross_checkpoint.py \
  --stage confirmatory
```

默认使用 GPU 4-7，输出到：

```text
/home/dev/promoe-probes/credit-balance-lossfree-s0-200k-v2
```

这是长任务，必须在当前 attached tmux session 的新 window 中运行。

## 结论边界

完整通过只支持：在这一对 checkpoint 和锁定输入上，token-count balance 没有消除稳定的 routed-expert credit mismatch，而且 suffix-gradient energy 能预测真实参数侧经验 Fisher。

它不能支持：

- Loss-Free 本身是论文贡献；
- 梯度能量等于 expert 的语义知识或样本价值；
- frozen-checkpoint 相关性证明了因果优化收益；
- 通用梯度归一化足以构成 TPAMI 方法；
- 内部指标改善等于 OpenAI 50K-sample FID 改善。
