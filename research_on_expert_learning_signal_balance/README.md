# MoE 专家学习信用再分配

## 当前状态：实验已删除，只保留假设

这条线的**全部实验代码、配置和脚本已经从仓库删除**。

原因不是假设本身有问题，而是它唯一做过的实验是**接着另一个实验的 301K checkpoint 继续训练 20K step**。
本项目不承认这种实验：起点权重来自另一次训练，因素混杂，无法区分改进来自方法本身还是来自那个起始
checkpoint，所以它根本不算干净的消融，也不能作为论文证据。（同一个模型从 300K 续训到 500K 不属于
此列——那是同一条轨迹的延长，不是跨实验借权重。）

留在本目录的只有 `git_provenance.py`：它与本研究方向无关，是通用的 Git provenance 读取工具，
`train.py` 的 `STRICT_PROVENANCE_SOURCE_PATHS` 和 `analyses/expert_update_budget/audit.py` 的
`LOCKED_TRAINING_SOURCE_PATHS` 都按路径 + sha256 钉住了它，所以不能移动或改名。

一并删除的东西：

- `configs/004_ProMoE_B_credit_rate_{matched,measure_only,permuted}_s0_301k_20k.yaml`
- `scripts/credit_redistribution/`（三个 301K→321K 启动脚本）
- `analyses/run_credit_redistribution_gate.py` 及其 `.md`
- 本目录的 `controller.py`、`transcript.py`、`protocol.py`、`protocol_lock.py`、`heldout.py`、
  `evaluator.py`、`statistics.py`、`orchestration.py`、`benchmark.py`、`serialization.py`、
  `state_digest.py` 和 `tests/`
- `train.py` 中的 `credit_controller` / `transcript_recorder` / `throughput_timer` 三条通路，
  以及 `load_latest_checkpoint()` 的 `initial_checkpoint_path` 参数——那是「从外部 checkpoint
  起跑」的机制本身，删掉之后 `train.py` 在代码层面就不再提供这种起跑方式
- `config.py` 中的 `credit_redistribution_config` / `training_transcript_config` /
  `throughput_timer_config` 三个开关

旧实验产生的 JSON、日志和统计结果仍在仓库根目录下的
`credit_redistribution/archived_credit_redistribution/<日期>/`（Git 忽略，只作历史追溯，不是论文证据）。

## 这个方向原本想研究什么

普通负载均衡只数每个专家收到了多少 token。但两个专家即使收到同样多的 token，它们收到的梯度也可能差很多。

把一个 token 经过专家后，从后面网络传回来的梯度大小当作「学习信用」的近似值：

```text
token_credit = route_weight^2 * ||suffix_gradient||_2^2
credit_rate  = expert_credit_sum / expert_token_count
```

直白地说，`credit_rate` 想回答的是：这个专家平均处理一个 token，实际得到了多强的学习信号。

Base 200K 的冻结权重检查发现，这个量在不同专家之间确实不均匀，而且探索集和独立确认集上的专家排序较稳定。
这个发现只能说明「问题可能存在」，不能说明任何再分配方法有效。该检查本身是**只读 checkpoint 的探针**，
不是训练实验，仍然有效，代码在 `analyses/run_learning_credit_balance_probe_batch.py` 和
`analyses/run_learning_credit_balance_cross_checkpoint.py`。

## 未来允许的干净实验

如果后续文献阅读和现有 fresh 实验结果仍支持这条假设，必须从零重新实现下面三条分支：

| 分支 | 作用 |
| --- | --- |
| measure-only control | 完整测量信用，但不调整任何专家 |
| matched redistribution | 把调整量给到它真正对应的专家 |
| permuted control | 调整量完全相同，但故意打乱专家对应关系 |

三条分支必须同时满足：

1. 都从空输出目录的 step 0 开始，不得从任何已有 checkpoint 起跑。
2. seed、参数初始化、数据顺序、global batch 256、学习率 1e-4 和训练长度完全一致。
3. 先过 300K dual-CFG FID gate（CFG 1.0 和 1.5 都要胜过 fresh ProMoE-TC 基线），过了才训到 500K。
4. 正确匹配必须明显胜过故意错配；否则只能说明普通梯度缩放有效，不能说明「专家信用匹配」有效。
5. 同时报告专家 token 数、平均学习信号、梯度预算、权重变化、路由分布、训练速度和推理代价。

不要试图复活被删掉的 controller / protocol / orchestration：它们把 continuation 的起点、
起止 step 和旧预注册写死在代码里，按 step-0 协议必须重新设计。

## 研究边界

- 这条线研究 MoE 专家的优化差异，不是 REPA，也不做 DINO 特征对齐。
- 只让信用不均匀度下降不算成功；生成质量必须同时改善。
- 单 checkpoint、单 seed 或短 continuation 不能进入论文主表。
- 如果方法最后只是给每个专家套一个普通 GradNorm，它的新颖性不足以支撑 TPAMI 扩展。
- 当前主线优先研究路由是否选对专家，以及如何把路由准度和负载均衡分开。credit redistribution 只作为后备假设。
