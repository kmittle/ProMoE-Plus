# MoE 专家学习信用再分配

## 当前状态

本方向暂时只保留为研究假设。仓库里现有的 301K→321K continuation 方案已经停用，不允许继续启动，也不能作为论文方法结果。

现有代码和脚本是历史诊断工具。它们从同一个中途 checkpoint 出发，原本只想快速检查“把更多学习信号给长期偏弱的专家”是否有局部作用。这种实验无法说明完整训练的改进来自模型设计，还是来自起始 checkpoint 和后期训练技巧，所以不再按这套方案继续。

当前没有任何经过批准的 credit redistribution 训练命令。三份旧配置已经标记为归档；旧 gate 和 train.py 都会直接拒绝启动它们。未来若重新启动这条线，必须先实现一套新的 fresh-training 协议，并从 step 0 训练到 500K。

旧实验产生的 JSON、日志和统计结果统一放在 `archived_credit_redistribution/<日期>/`。这个归档目录由 Git 忽略，只用于保留已经做过但不能作为正式论文证据的结果。仍被 `train.py` 导入的 Python 源码留在本目录根部，避免基础训练因为整理文件而无法启动。

## 这个目录原本想研究什么

普通负载均衡只数每个专家收到了多少 token。但两个专家即使收到同样多的 token，它们收到的梯度也可能差很多。

这里把一个 token 经过专家后，从后面网络传回来的梯度大小当作“学习信用”的近似值：

```text
token_credit = route_weight^2 * ||suffix_gradient||_2^2
credit_rate  = expert_credit_sum / expert_token_count
```

直白地说，`credit_rate` 想回答的是：这个专家平均处理一个 token，实际得到了多强的学习信号。

旧 Base 200K 的冻结权重检查发现，这个量在不同专家之间确实不均匀，而且探索集和独立确认集上的专家排序较稳定。这个发现只能说明“问题可能存在”，不能说明任何再分配方法有效。

## 历史 continuation 为什么不能作为正式证据

旧方案包含三条从同一个 Base checkpoint 继续 20K step 的分支：

1. 只测量，不改变梯度。
2. 按专家自己的长期信用差异调整梯度。
3. 使用同一组调整倍率，但故意把倍率错配给其他专家。

这三条分支并没有形成可用的正式结果。即使以后把它们跑完，也只能说明某个成熟模型上的短期局部效应，不能证明从头训练时的生成质量会提高。

因此，以下内容都属于历史 continuation 基础设施，不是当前可启动实验：

- `analyses/run_credit_redistribution_gate.py`
- `configs/` 中带 credit redistribution continuation 设置的配置
- `scripts/credit_redistribution/` 中从中途 checkpoint 启动的脚本
- 本目录中绑定起始 checkpoint、起止 step 和旧预注册的编排代码

## 未来允许的干净实验

如果后续文献阅读和现有 fresh 实验结果仍支持这条假设，至少要重新实现下面三条分支：

| 分支 | 作用 |
| --- | --- |
| measure-only control | 完整测量信用，但不调整任何专家 |
| matched redistribution | 把调整量给到它真正对应的专家 |
| permuted control | 调整量完全相同，但故意打乱专家对应关系 |

三条分支必须同时满足：

1. 都从空输出目录的 step 0 开始。
2. 都不设置 `initial_checkpoint_path`。
3. seed、参数初始化、数据顺序、global batch 256、学习率 1e-4 和训练长度完全一致。
4. 都训练到 500K，并在 300K/500K、CFG 1.0/1.5 下各生成 50K 张图，用 OpenAI evaluator 计算 FID/IS。
5. 正确匹配必须明显胜过故意错配；否则只能说明普通梯度缩放有效，不能说明“专家信用匹配”有效。
6. 同时报告专家 token 数、平均学习信号、梯度预算、权重变化、路由分布、训练速度和推理代价。

在新模式实现前，不能简单修改旧 YAML 的训练步数来冒充 fresh 实验。旧 controller、protocol 和 orchestration 都写死了 continuation 的起点与检查逻辑，需要按 step-0 协议重新设计。

## 文件职责

下面是现有历史代码的职责，保留它们是为了追溯旧诊断，不代表已经批准继续运行：

- `controller.py`：捕获路由、计算信用近似值、维护滑动平均和调整专家梯度。
- `transcript.py`：记录训练输入，供旧三分支 continuation 对齐样本。
- `protocol_lock.py`、`protocol.py`：锁定旧实验的代码、配置、checkpoint、数据和环境。
- `heldout.py`、`evaluator.py`、`statistics.py`：生成旧 held-out 数据、评估局部去噪误差并汇总统计。
- `orchestration.py`：编排旧 gate、预飞、continuation、评估和聚合。
- `benchmark.py`：测量控制器的训练开销。
- `serialization.py`、`state_digest.py`：保存哈希和训练状态摘要。
- `tests/`：验证旧公式、预算守恒、恢复训练和协议绑定。

## 研究边界

- 这条线研究 MoE 专家的优化差异，不是 REPA，也不做 DINO 特征对齐。
- 只让信用不均匀度下降不算成功；生成质量必须同时改善。
- 单 checkpoint、单 seed 或短 continuation 不能进入论文主表。
- 如果方法最后只是给每个专家套一个普通 GradNorm，它的新颖性不足以支撑 TPAMI 扩展。
- 当前主线优先研究路由是否选对专家，以及如何把路由准度和负载均衡分开。credit redistribution 只作为后备假设。
