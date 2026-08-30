# 专家更新预算审计

这项分析检查一个很具体的 MoE 问题：每个 expert 收到的 token 数量接近，并不代表它们在训练中真的获得了接近的参数更新量。

它只读取从零训练保存的 checkpoint，不修改模型，不做短程续训，也不使用 REPA、DINO 或 teacher。即使结果通过，也只允许继续做下一项冻结分析，不能直接声称方法有效。

## 为什么现有结果还不够

仓库已有的 learning-credit probe 测量了去噪损失传到 expert 输出端的梯度，并用单个 batch 的精确经验 Fisher 检查了参数侧梯度。它回答的是“当前输入给谁更强的学习信号”。

这里回答另一个问题：这些短期差异是否真的在 50K 到 300K 的训练轨迹里积累成长期、稳定的 expert 参数变化差异。若没有长期沉积，就没有理由为此设计新的训练机制。

## 测量内容

对同一个 block、同一个条件 expert，在相邻 checkpoint `a` 和 `b` 之间计算：

```text
参数位移 RMS = sqrt(sum((theta_b - theta_a)^2) / 参数个数)
相对参数位移 = sqrt(sum((theta_b - theta_a)^2) / sum(theta_a^2))
```

相对参数位移是主指标。它同时除掉了 expert 参数个数和参数本身的整体尺度。分析不会把不同 block 的 expert ID 混在一起。

这里必须使用 `model_state_dict`，不能使用 EMA 权重，因为 AdamW state 只对应实际参加反向传播的 raw model。EMA 仍然是生成评估使用的模型，但它不适合回答 optimizer 把更新给了谁。两个 checkpoint 的差只能测量净位移；如果参数在区间内来回移动，它会低估真实走过的总路程。

在后一个 checkpoint 上，分析还从 AdamW 的一阶、二阶矩计算当前的局部更新向量：

```text
m_hat = m / (1 - beta1^step)
v_hat = v / (1 - beta2^step)
局部更新向量 = lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta)
```

这只是 checkpoint 末端的局部方向，不是过去 50K step 所有更新的精确总和。因此报告把它作为独立的交叉检查，不会把它冒充真实累计更新量。

只统计 zero-based MoE block `1/3/5/7/9/11` 中的 12 个条件 routed experts。第 13 个 unconditional expert 和 shared expert 都不进入条件专家之间的比较。

## 提前锁定的门槛

Seed 0 和尚未训练完成的 Seed 1 使用完全相同的门槛：

- 5 个相邻区间乘 6 个 MoE block，共 30 个 block/阶段单元；
- 相对参数位移的中位 Gini 至少 `0.08`；
- 至少 `67%` 的单元达到 Gini `0.08`；
- AdamW 局部相对更新量的中位 Gini 至少 `0.08`；
- 至少 `90%` 的 block/阶段单元可以计算长期位移与局部更新量的 expert 排序相关；
- 长期位移与局部更新量在 expert 间的中位 Spearman 至少 `0.30`；
- 相邻阶段 expert 位移排序的中位 Spearman 至少 `0.30`；
- 每个 block 的 4 个相邻阶段排序都必须可以计算，少一个就把该 block 判为无效；
- 至少 `67%` 的 block 具有正的阶段排序稳定性。

这些门槛要求差异既有可见大小，又在多数 block/阶段出现，还要能从另一条 optimizer 证据得到支持。任何一项失败都停止这条假设，不能看到结果后改门槛。

协议文件：

- `analyses/expert_update_budget/manifests/expert_update_budget_seed0_v1.json`
- `analyses/expert_update_budget/manifests/expert_update_budget_seed1_v1.json`

## 与已有工作的边界

- GradNorm（ICML 2018）已经研究多任务损失的梯度幅度平衡；简单按 expert 梯度范数缩放不新。
- Loss-Free Balancing（arXiv:2408.15664）直接平衡 token 数；本分析必须证明参数更新差异不是把已有负载不均衡换个名字。
- SkewAdam（arXiv:2607.19058）研究 backbone、expert 和 router 使用不同 optimizer state；它没有解决同一 block 内具体 routed expert 的学习机会分配。
- MESH（arXiv:2608.04407）研究 routed expert 梯度的时间平滑和 Sinkhorn optimizer；通用 expert optimizer 也不足以成为这里的方法贡献。

因此，若后续设计方法，它至少必须同时满足：路由条件相关、同一 block 内按 token 数校正、总更新预算守恒，并用打乱 expert 对应关系证明“具体哪个 expert 欠了多少学习机会”具有因果作用。否则只是把梯度归一化、负载均衡和现有 optimizer 工作拼在一起。

## 运行 Seed 0

代码必须先检查、commit 并 push，且运行时满足 `HEAD == origin/repa == GitHub repa`。输出固定放在项目内已忽略的 `analyses/archvied_analyses/`。

程序会在分析前后各核对一次协议、配置和全部 checkpoint 的 SHA256，也会再次确认 Git 工作树没有变化。输出目录必须为空；使用 `--overwrite` 时也只允许覆盖本程序生成的四个文件，遇到其他文件会停止，避免把新旧结果混在一起。

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_expert_update_budget_audit.py \
  --manifest analyses/expert_update_budget/manifests/expert_update_budget_seed0_v1.json \
  --config configs/004_ProMoE_B_fresh_routing_audit_s0.yaml \
  --checkpoint-dir outputs/ProMoE_TC_B/004_ProMoE_B_fresh_routing_audit_s0/checkpoints \
  --output-dir analyses/archvied_analyses/2026-08-30/expert_update_budget_seed0_v1
```

生成：

- `audit.json`：每个区间、block、expert 的完整结果和 provenance；
- `summary.json`：门槛判断和跨阶段排序；
- `expert_metrics.csv`：方便画图和人工复核的逐 expert 表；
- `summary.md`：直白中文结论。

所有门槛都通过时命令返回 `0`。只要有一项未通过，四份报告仍会完整保存，但命令返回 `1`，提醒自动实验队列停止这条假设。

## Seed 1 复验

Seed 1 到 300K 后，用相同程序和提前提交的 Seed 1 协议运行。Seed 1 配置已经保存在 `configs/004_ProMoE_B_seed1_control.yaml`，输出目录改为对应的 Seed 1 路径。

## 解释限制

当前 Seed 0/1 训练启动于提交 `257d51a`，而 authoritative `repa` 后续已经前进。因此这两条轨迹只能作为提前锁定的诊断和独立种子复验，不能冒充最终论文的 canonical 500K 对照。

即使两条轨迹都通过，也还不能说明：

- 更新较少的 expert 一定学得差；
- 所有 expert 应该获得完全相同的更新；
- AdamW 局部向量等于 50K step 的累计更新；
- 任意梯度重加权都会改善 OpenAI 50K-sample FID。

下一步必须在相同冻结样本上联合测量 token count、learning credit 和参数更新量，并做 block 内 expert 对应关系打乱对照。只有该因果 gate 通过，才允许实现从 step 0 训练的新方法。
