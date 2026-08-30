# 独立语义路由检查的创新性边界

更新日期：2026-08-30

## 先说结论

这项检查本身不是足以投稿 TPAMI 的方法贡献。它是一道问题筛选门：先确认 ProMoE 的专家分组是否能被独立视觉语义解释，再决定是否继续研究“路由语义、专家能力和去噪效果为什么没有对齐”。

如果检查通过，能得到的新信息是：在保留专家负载和路由图空间形状后，历史 ProMoE 路由仍与外部视觉语义有关。不能把它写成“首次语义路由”，更不能把 DINOv2 当作新算法。

## 最接近的工作

| 工作 | 已经做了什么 | 本检查不能怎样宣传 |
| --- | --- | --- |
| [ProMoE](https://arxiv.org/abs/2510.24711) | 用 learnable prototype 和 routing contrastive loss 组织图像 token，并把语义专家分工作为核心动机 | 不能再次声称首次让视觉 MoE 按语义分专家；本检查只是在更独立的尺子下验证原主张 |
| [DINOv2](https://arxiv.org/abs/2304.07193) | 学到可迁移的视觉 patch 表征 | 不能把 DINO 特征或 kNN 当作方法创新；这里只把冻结 DINO 当测量仪器 |
| [REPA](https://arxiv.org/abs/2410.06940) | 把扩散模型内部表征对齐到外部视觉 encoder，以改善训练 | 不能把本检查改写成 DINO 表征对齐；DINO 不进入模型、router 或训练损失 |
| [Scaling Vision with Sparse Mixture of Experts](https://arxiv.org/abs/2106.05974) | 展示视觉 MoE 的专家使用和图像类别结构 | 不能说首次分析视觉专家分工或类别偏好 |
| [Routers Learn the Geometry of Their Experts](https://arxiv.org/abs/2605.12476) | 分析任务梯度怎样让 router 与 expert 形成共同几何，并比较 centroid router | 不能把 prototype 接近 token 均值或 router–expert 几何耦合作为新贡献 |
| [Polysemantic Experts, Monosemantic Paths](https://arxiv.org/abs/2604.17837) | 把 router 可见控制子空间与专家处理的内容路径分开分析 | 不能把 router 子空间或“一个专家含多种语义”本身包装成发现 |
| [Beyond Routing](https://arxiv.org/abs/2608.08853) | 把 dispatch 与多专家 aggregation 分开，并训练额外聚合头 | 即使发现语义分组与去噪能力不一致，也不能直接加一个 aggregation head 冒充新方法 |

## 与 REPA 的实质区别

REPA 的因果链是：外部 encoder 特征进入训练损失，改变 diffusion backbone 的表示，最后可能改善生成。

本检查的因果链只有：训练完成的 ProMoE 产生 route ID，冻结 DINO 对同一批图片做事后测量，程序比较真实对应和打乱对应。删除整项检查不会改变任何 checkpoint，也不会改变一张生成图片。

因此可以借 DINO 回答 MoE 问题，但不能把 DINO 直接接到 backbone 做通用表征对齐。若未来使用 DINO 训练 router，也必须证明它解决的是可量化的误路由、负载或专家解耦问题，并设置同成本的打乱教师信号对照；否则仍是 REPA 式拼接。

## 当前设计比普通可视化多了什么

普通热图只能说明专家编号在空间上成块。这里专门排除两种简单解释：

1. 整体平移路由图，保留每位专家的 token 数和空间形状，检查结果是否只来自相邻 patch 本来相似。
2. 把完整路由图换给错误图片，保留每张路由图本身，检查跨图结果是否只来自某些专家本来更常被使用。

图片而不是 patch 是统计单位，避免把一张图的 256 个高度相关 patch 当成 256 份独立证据。跨图指标同时重采样查询图片和候选图库，因为多个查询会共享图库，不能直接把它们当成独立样本。54 次 cell/指标检验统一做 Holm 校正。DINO 的权重、源码 commit 和源码树内容都被锁定，不能靠一个同名本地缓存目录冒充固定源码。通过还必须跨多个 block 和多个噪声位置，不能依赖一个漂亮案例。

这些设计提高了诊断可信度，但仍不构成训练算法创新。

## 即使通过也缺少的两条证据

1. RCL 因果：当前只有带 RCL 的 Base 路由。没有同种子、同训练设置、从零训练的无 RCL 对照，就不能说语义结构由 RCL 带来。
2. 专家能力：语义相似不等于专家适合。必须用保持计算量和专家负载的 exact counterfactual 检查，同语义 token 是否真的由同一专家获得更低去噪误差。

只有“独立语义存在”“RCL 确实改变它”“这种改变与真实专家能力一致”三条都成立，才形成值得设计方法的问题。否则继续加 loss 只是在已有模块上缝合。

## 停止条件

出现任一情况就停止这条语义叙事：

1. 真实路由没有稳定超过两种保结构对照。
2. 结果只出现在一两个 block 或一个噪声位置。
3. Fresh checkpoint 无法复现 dirty discovery。
4. 去掉 RCL 后语义指标不变，说明不能把现象归因于 RCL。
5. 语义指标与 exact expert utility 无关，说明“分得像”没有带来“分得对”。

即使全部通过，最直接的 DINO 蒸馏、prototype centroid、普通对比学习、额外 aggregation head、通用正交约束和表征对齐仍然不批准为 TPAMI 主方法。下一方法必须利用 ProMoE 的自分组、Top-1 shared+routed 结构和扩散去噪责任之间特有的矛盾，并配套正确信号与打乱信号的从零训练对照。
