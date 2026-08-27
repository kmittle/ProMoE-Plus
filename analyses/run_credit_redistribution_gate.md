# Credit-Rate Redistribution Gate

这个入口执行冻结的 Base seed-0 三臂 late-training 因果筛选。研究问题不是“让 token 数量更均衡”，而是：在 token 负载已经均衡后，专家是否仍收到不均衡的单位-token学习信用；如果存在，把固定的每层原始梯度预算按专家自身的 credit rate 重新分配，是否比不干预和错配反馈都更好。

三臂固定为 `measure_only_control`、`rotating_permuted_scale_control` 和 `matched_credit_rate_redistribution`。它们都从同一个 step-301000 checkpoint 开始，在 GPU 4-7 上依次训练 20K updates。训练输入、LR、global batch、采样顺序、FLOPs 和推理图保持一致。

## 执行顺序

所有命令从仓库根目录执行。长任务必须放在当前附着的 tmux session 新窗口中。

1. 代码通过 `$check 1`、提交并 push，且 `HEAD == origin/repa`、工作树干净后，物化 held-out 张量：

   ```bash
   /home/dev/miniforge3/envs/promoe/bin/python \
     analyses/run_credit_redistribution_gate.py materialize-heldout
   ```

2. 写入绑定 commit、源码、配置、launchers、数据、环境、checkpoint 和 held-out tensors 的 immutable protocol：

   ```bash
   /home/dev/miniforge3/envs/promoe/bin/python \
     analyses/run_credit_redistribution_gate.py write-protocol
   ```

3. GPU 4-7 空闲后，在 tmux 中运行三次 20-update preflight。它比较一个 transcript-only Base leg 和两个独立 measure-only replay。Base 与 measure-only 的 model、EMA、optimizer、trainer state 和完整输入 transcript 必须逐内容一致；normalizer/controller state 只在两次 measure-only replay 之间逐内容一致，因为 transcript-only Base leg 不安装信用控制器：

   ```bash
   /home/dev/miniforge3/envs/promoe/bin/python \
     analyses/run_credit_redistribution_gate.py preflight
   ```

4. Base/Loss-Free 的 plumbing、discovery、parameter 和 confirmatory gate 全部通过后，按固定顺序执行三份 launcher。每份 launcher 会先运行 `verify-launch`，并拒绝缺失前置结果、错误 commit、协议漂移、顺序错误或重复启动：

   ```bash
   bash scripts/credit_redistribution/run_B_credit_rate_measure_only_s0_301k_20k.sh
   bash scripts/credit_redistribution/run_B_credit_rate_permuted_s0_301k_20k.sh
   bash scripts/credit_redistribution/run_B_credit_rate_matched_s0_301k_20k.sh
   ```

5. 三臂全部完成后先做盲态 held-out evaluation。该命令只生成逐 case 的 sealed 原始量，不计算组间性能差异：

   ```bash
   /home/dev/miniforge3/envs/promoe/bin/python \
     analyses/run_credit_redistribution_gate.py evaluate
   ```

6. 所有 checkpoint、transcript、controller ledger 和 case seals 通过完整性检查后，只执行一次聚合揭盲：

   ```bash
   /home/dev/miniforge3/envs/promoe/bin/python \
     analyses/run_credit_redistribution_gate.py aggregate
   ```

7. 只有 efficacy 全部通过后才运行 ABBA throughput gate：

   ```bash
   /home/dev/miniforge3/envs/promoe/bin/python \
     analyses/run_credit_redistribution_gate.py throughput
   ```

## 输出

协议与分析工件写到 `/home/dev/promoe-probes/credit-normalized-expert-gradient-ab-v4/`，三臂 checkpoint 写到 `/home/dev/promoe-runs/credit-normalized-expert-gradient-ab-v4/`。这些目录不属于 Git 工作树。

本 gate 只支持一个 seed、一个冻结轨迹上的 20K continuation。即使通过，也只能授权 fresh multi-seed 训练，不能直接声称改善生成质量或构成 TPAMI 贡献。
