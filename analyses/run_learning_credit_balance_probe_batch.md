# Learning-Credit Balance Probe

这个 probe 检验一个 MoE 内部问题：expert 收到的 token 数量接近，并不代表它们收到的有效学习信号也接近。

它不训练模型，不改变路由，也不使用 REPA、DINO 或 teacher feature。冻结 checkpoint 上的结果只能说明机制是否存在，不能证明 FID 会改善。

## 学习信用

对一个 routed MoE block，记 denoising MSE 对该 block 输出中 token `t` 的梯度为 `g_t`，原生 top-1 route weight 为 `w_t`。probe 定义：

```text
token credit c_t = w_t^2 * ||g_t||_2^2
expert credit C_e = sum_{route(t)=e} c_t
expert credit rate R_e = C_e / N_e
```

`N_e` 是 expert `e` 收到的 token 数。`c_t` 是传到 routed-expert 输出的平方梯度能量，不是 router affinity、expert output magnitude 或 counterfactual routing utility。

主指标包括：

- `credit_rate_gini`：各 active expert 的 `R_e` Gini。
- `load_credit_tv`：token-count share 与 credit share 的总变差距离。
- `permutation_excess_tv`：固定完整 expert count vector，只在同一 cell 内打乱 token credit 后，观测 TV 超过置换零假设的部分。
- `rank_stability`：expert credit-rate 排名能否跨图像切分以及 discovery/confirmatory 保持。

另报告把 `w_t` 固定为 1 的结果，以区分 route weight 集中和 suffix-gradient 集中。普通 token-count CV/Gini 会并列报告，不能把已有 load imbalance 重新命名为 credit imbalance。

## 锁定门槛

预注册文件是：

```text
/home/dev/promoe-probes/credit-balance-gate-base200k-v1-preregister.json
SHA256 392be0136b046ebaef8f02dc3f05263925d2b5585fb4f26c2d817ee08abde5b9
```

协议固定 Base seed-0 step-200K EMA、zero-based blocks `1/5/11`、sigmas `0.2/0.5/0.8`，并用互不重叠的 ImageNet 类构造：

1. `plumbing`：8 张图，case JSON 和 summary 都只发布数值安全结果；每个 cell 在写盘前删除 MSE、credit 和其他 efficacy 字段。
2. `discovery`：32 张图，检验 count-adjusted credit imbalance 和 split-half expert-rank 稳定性。
3. `confirmatory`：64 张新图，复核效应、全部 block/sigma 分层和 discovery-to-confirmatory 排名稳定性。

统计单位始终是 image。每个 cell 的置换次数为 4096，聚合使用 200,000 次 image bootstrap；block/sigma 分层使用 Holm 校正。任何门槛失败都停止本路线，不能事后调阈值。

只有 confirmatory 完整通过，才允许从相同 Base-200K training state 做一个 20K-step 的 credit-normalized expert-gradient 因果 A/B。即使短训通过，也必须再做第二 seed 和最终 500K/50K-sample OpenAI evaluation。

## 运行

代码必须先通过检查、commit 并 push，且工作树干净、`HEAD == origin/repa`。先锁定协议：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_probe_batch.py \
  --prepare-only
```

随后按顺序运行：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_probe_batch.py \
  --split plumbing

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_probe_batch.py \
  --split discovery

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_learning_credit_balance_probe_batch.py \
  --split confirmatory
```

默认输出是 `/home/dev/promoe-probes/credit-balance-gate-base200k-v1`。runner 固定使用 GPU 4-7，每个 GPU worker 只加载一次模型。每个 case 和 summary 都绑定 protocol hash 并单独写 seal；完整兼容的 case 可以恢复，单边文件或 hash 不一致会直接失败。

长任务必须在当前 attached tmux session 的新 window 中运行。

## 解释限制

通过只能支持以下结论：在锁定的 ProMoE checkpoint 和输入分布上，token-count balance 没有消除稳定的 expert learning-credit imbalance。

不能据此声称：

- 所有 MoE 的 token balancing 都无效；
- 梯度能量等于 expert 的语义价值；
- frozen-checkpoint 统计已经改善生成质量；
- 一个通用 gradient normalization 技巧本身构成 TPAMI 贡献。
