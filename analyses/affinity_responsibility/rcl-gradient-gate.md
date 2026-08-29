# RCL 是否把专家责任推错方向

更新日期：2026-08-29

## 一句话说明

ProMoE 用同一个 cosine 分数做两件事：先选一位 routed expert，再把这位专家的输出乘以这个分数。前一件事是“让谁来做”，后一件事是“让它承担多少”。这两件事不一定应该共用一个数。

RCL 会根据 router 自己当前的分组，把每个 prototype 拉向已经分给它的 token 均值。因此要检查一个很具体的问题：

> RCL 为了让当前分组更整齐而移动 prototype 时，是否会同时把 routed expert 的输出强度推向更高的去噪误差？

如果答案是否定的，就停止这条解释。不能因为旧实验里换一个 scale 会更好，就反过来认定 RCL 是原因。

## 已知线索和不能隐瞒的地方

旧的 50K dirty 权重显示：固定 top-1 expert 不变，只换 routed output scale，很多 token 能降低 teacher-forced MSE。这个结果只能说明 Fresh 权重值得复查，也影响了“责任错配”门槛，所以 Fresh 检查不是完全盲测。

协议锁定时还没有看过 Fresh 300K 结果，也没有跑过 RCL 梯度冲突统计。样本、层、噪声位置、support batch、打乱次数、更新步长和门槛都在结果出现前固定。

## 为什么要把 support 和 query 分开

训练时，每张 GPU 用自己的 64 张图计算 RCL，然后 DDP 把 4 张卡的梯度取平均。若用一张 query 图既产生 RCL 梯度又评价这个梯度，得到的“分组更好”只是原地背答案。

本检查为 4 张 GPU 各固定一组 64 张 support 图。它们来自 256 个不同类别，而且与 36 张 query 的类别完全不重合。每组固定 6 张作为 unconditional，近似训练时的 10% class dropout。每张 support 图都按训练设置固定抽取自己的噪声时刻，不会让整批图共用一个时刻。程序最后把 4 张卡的梯度取平均，所有 query 都使用同一个平均方向。这与一次真实的四卡训练更新一致，也避免把共用一个局部方向的 6 张图误当成 6 次独立实验。

## 怎样把原因拆开

每张 query 检查 6 个 MoE block 和 3 个噪声位置，共 18 个格子。每个格子做下面几件事：

1. 固定 expert ID，只替换 routed output scale，复查原生 affinity 是否接近最优。
2. 单独计算去噪损失希望 prototype 怎样移动。top-1 选择在局部不求导，所以这个梯度只来自输出强度。
3. 使用独立的四卡 support batch 计算目标 block 的 RCL 直接 prototype 梯度，并像 DDP 一样平均四个 rank 的梯度。
4. 在每个 rank 内把 support token 与 expert 的对应关系随机打乱 16 次，但每位 expert 的 token 数完全不变；同一编号的四个打乱梯度也按 DDP 取平均。
5. 对正确 RCL、16 个打乱 RCL 和纯 diffusion 对照使用相同的 prototype 更新长度，更新后恢复每个 prototype 原范数。
6. 保持原 expert ID，只用更新后的 prototype 重算 cosine 输出强度，再完整前向得到精确 MSE。

正确 RCL 还会运行半步。若半步、整步和一阶预测互相不符，说明步长或程序不可靠，该格不能进入结论。

这里必须明确一个边界：实际 RCL 还会通过 token 均值更新前面的 hidden state。本检查只隔离“RCL 直接更新 prototype”这条路径，不能把结果写成整个 RCL 梯度的完整解释。

## 什么才算支持机制

确认集必须同时满足三类证据：

1. **责任确实错配**：原生 affinity 很少是固定 dispatch 下的最佳 scale，候选 scale 更好的比例有稳定下界，而且 affinity 与最佳 scale 的相关性不能很高。
2. **正确分组具有两面性**：相对保持负载的打乱分组，正确 support-RCL 必须更改善独立 query 的分组几何，同时造成更高的精确去噪 MSE。只有一阶梯度冲突而没有精确 MSE，不算通过。
3. **不是单点现象**：现象至少覆盖 4 个 MoE block 和 2 个噪声位置，并在独立图片上有稳定下界。

具体门槛写在 `manifests/rcl_responsibility_gate_v1.json`。程序会精确核对整个 manifest，多一个字段或改一句“尚未看过结果”都会拒绝运行。`plumbing` 只检查程序和数值对照；8 张 discovery 图片通过后，才能打开另外 24 张 confirmatory 图片。置信区间以图片为单位计算，不能把 token、格子或 shuffle 伪装成更多独立样本。它描述的是这一个预先固定的四卡 support 更新对独立 query 图片的效果，不会冒充对所有 support batch 都成立。

## 为什么使用在线权重

问题讨论的是训练时 RCL 梯度怎样与 diffusion 梯度共同更新参数。EMA 权重没有直接接受这些 optimizer 更新，不能代表训练机制。因此主检查强制读取 checkpoint 的 `model_state_dict`，并拒绝自动换成 `ema_model_state_dict`。

## 即使通过，也不能直接做什么

- 直接加一个独立 aggregation head；`Beyond Routing` 已经做过。
- 把 prototype 换成 expert 权重或在线均值；`Routers Learn the Geometry of Their Experts` 已覆盖很近的想法。
- 用 DINO 特征教 hidden state 对齐；这仍然是 REPA 式表征对齐，没有解释 MoE 问题。
- 只去掉 RCL，或只把 cosine 改成常数；这可以做消融，但不是完整方法。

若门控通过，下一步方法至少要让“选谁”的分组证据与“承担多少”的去噪证据各负其责，同时保留 ProMoE 的稀疏计算和 shared/routed 结构。正式训练必须从 step 0 开始，并包含正确依据、保持统计量不变的打乱依据、去掉新机制三条对照。
