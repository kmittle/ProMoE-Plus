# MoE 专家学习信用再分配

本目录实现一组围绕 MoE 学习动态的受控实验，用来研究下面的问题：

> 当不同专家接收的 token 数量已经比较均衡时，它们获得的单位 token 学习信用是否仍然不均衡；在保持梯度总预算不变的前提下，把更多学习信用分配给长期信用不足的专家，能否改善模型训练。

`credit_redistribution/` 是这组实验的 Python 基础设施包，不是一个独立模型家族，也不是训练输出或数据缓存目录。它被 `train.py` 和 `analyses/run_credit_redistribution_gate.py` 调用，负责训练期信用测量与干预、冻结协议、可复现性校验、盲化评估和统计聚合；模型定义仍沿用仓库现有的 ProMoE 实现。

这里的 20K continuation 只回答“在同一个已训练状态上做后期干预是否有局部效应”，不能替代方法级比较。任何声称改善整体训练或生成质量的候选方法，都必须另建 fresh、相同预算的 step 0→500K、多 seed 实验；本目录的 continuation 输出不得作为 TPAMI 主结果。

这里的“学习信用”不是 token 数量，也不是语义标签。当前实验使用路由权重和 MoE 输出后续梯度构造代理量：

```text
token_credit = route_weight^2 * ||suffix_gradient||_2^2
credit_rate  = expert_credit_sum / expert_token_count
```

该代理量只用于分析和训练期梯度控制，不改变路由选择、模型参数量或推理图。

## 实验设计

冻结协议包含三条从同一 Base seed-0 checkpoint 出发的 20K-step continuation：

1. `measure_only_control`
   完整执行信用测量、EMA、日志和完整性检查，但专家梯度缩放严格为 1。
2. `rotating_permuted_scale_control`
   使用同一套自适应缩放机制，但循环打乱专家与缩放系数的对应关系，用于检验“专家匹配”本身是否重要。
3. `matched_credit_rate_redistribution`
   根据每个专家自己的长期信用率分配缩放系数。

对后两条实验臂，每个 MoE block 都会重新计算预算因子，使缩放前后的 routed-expert 原始梯度平方范数总和保持不变。该约束保持的是裁剪前的原始梯度预算，不声称保持 AdamW 更新量或参数位移。

## 文件职责

- `controller.py`：训练期路由捕获、信用计算、EMA、专家梯度缩放、预算校验和 AdamW 遥测。
- `transcript.py`：记录每个 rank 的样本、标签、潜变量、噪声和时间步，确保三条实验臂获得相同训练输入。
- `protocol_lock.py`：验证冻结的 v3 预注册及 v4 单字段修订。
- `protocol.py`：生成并校验绑定代码、配置、checkpoint、数据和环境的不可变实验协议。
- `heldout.py`：确定性选择并固化 held-out 潜变量与噪声张量。
- `evaluator.py`：执行盲化 held-out 去噪评估，并校验 checkpoint、训练记录和评估产物。
- `statistics.py`：按照预注册规则计算 MSE、信用率 Gini、token-load CV 和配对 bootstrap 区间。
- `orchestration.py`：编排前置 gate、预飞、三臂 continuation、评估和结果聚合。
- `benchmark.py`：测量控制器带来的训练吞吐开销。
- `serialization.py`、`state_digest.py`：提供确定性序列化、文件哈希和训练状态摘要。
- `tests/`：覆盖公式、预算守恒、确定性、断点续训、协议绑定及防篡改行为。

仓库级命令入口位于：

```text
analyses/run_credit_redistribution_gate.py
```

三条 continuation 的配置位于 `configs/`，启动脚本位于 `scripts/credit_redistribution/`。

运行产生的 checkpoint、协议、held-out 张量和评估结果写到配置或协议指定的仓库外路径，不应写入本目录。

## 研究边界

- 本目录研究的是 MoE 专家学习信用和优化动态，不是 REPA 或 DINO teacher 表征对齐。
- 当前干预不直接修改 router，也不增加模型参数或推理计算。
- 当前三臂实验只是单 checkpoint、单 seed 的因果筛选。即使通过，也只能支持继续开展从 step 0 开始的多 seed 实验，不能单独构成生成质量或 TPAMI 贡献结论。
- 在三条实验臂全部完成且完整性检查通过前，不允许提前聚合或查看 held-out efficacy。
- Base/Loss-Free 前置 gate 未全部通过时，不允许启动三臂 continuation。

冻结的科学规则以外部 v3/v4 预注册文件为准；本 README 只说明目录用途，不替代实验协议。
