# 第三组：Teacher-Conditioned Expert Geometry

## 1. Motivation

前两组分别约束了路由关系和 shared/routed 分支职责，但它们都没有直接回答一个更接近 ProMoE 的问题：被路由到不同专家的 token，经过专家变换后是否形成了有意义的专家间结构。

只把专家参数彼此推远并不够。参数不同不代表激活真的不同，激活不同也不代表这种差异对图像语义有用。因此本组不做统一的专家排斥，也不把 DINO 特征硬聚类成 12 个伪专家。我们让 ProMoE 自己的 top-1 路由决定每张图里的 token 分组，再用冻结教师回答“这些专家组之间应该呈现什么相对几何关系”。

核心假设是：如果 routed experts 学到了互补而有语义的局部变换，那么同一张图中各专家输出质心之间的关系，应当接近对应 DINO token 质心之间的关系。该目标监督的是专家间相对结构，不要求学生和教师特征维度相同，也不规定专家编号的固定语义。

## 2. 参考文献与边界

1. **ProMoE**：会议版本提供 conditional routing、prototypical routing、shared expert 和 routing contrastive guidance。本组保留这些机制，只读取现有 top-1 分组和 routed-expert 输出。
2. **Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think**（REPA，arXiv:2410.06940）：证明冻结的判别式视觉表征可以改善扩散 Transformer。本组把同一教师从逐 token 对齐扩展到专家组之间的关系监督。
3. **Advancing Expert Specialization for Better MoE**（arXiv:2505.22323）：说明专家重叠会妨碍 specialization，并用正交与方差目标改善语言 MoE。它启发我们检查专家分化，但 TCEG 不直接套用其统一正交目标，而是用图像教师定义每张图所需的非均匀几何。
4. **Geometric Regularization in Mixture-of-Experts: The Disconnect Between Weights and Activations**（arXiv:2601.00457）：报告参数正交并不能可靠降低激活重叠，也不稳定改善性能。这正是本组选择“任务相关的输出激活几何”而不是“专家参数彼此远离”的原因。
5. **DINOv3**（arXiv:2508.10104）：其 Gram anchoring 表明保持 dense feature 的关系结构是重要问题。TCEG 使用的是本项目现有 DINOv2 教师，但借鉴了“关系结构比单个向量值更稳定”的思路。
6. **SPARE: Structural Parameter-Free Affinity Regularization for Flow Matching**（arXiv:2608.01990）：说明匹配关系可以绕开特征维度和额外 projector。TCEG 与它的区别是：监督对象不是全部 token 对，而是由 ProMoE 实际路由形成的专家输出质心，目标是专家分工而不是一般表征正则化。

因此，TCEG 目前的创新主张是一个待实验验证的组合：用学生自己的稀疏路由形成组，用冻结视觉教师规定组间几何，并直接训练 routed-expert 激活。它不是已有正交损失、token affinity loss 或硬教师聚类的改名。

## 3. 方法

正向 TCEG 和 spatial-roll 对照使用同一模型与同一损失实现，只改变教师 token 的固定空间位移。

1. 完整保留 Multi-Align 的 `align_blocks: [2, 3, 4]`、三个 projector、动态 token 系数和 `0.5 * Multi-Align loss`。
2. 只在零起始编号第 3 个 DiT block 计算 TCEG。它是这三个对齐层中唯一的 MoE block。
3. 只使用 conditional 样本。classifier-free dropout 后的 unconditional 样本不参与该损失。
4. 从该 block 读取每个 token 的 detached top-1 assignment，以及专家 MLP 的原始输出。这里的输出还没有乘路由权重、没有加 shared expert，也没有乘 AdaLN gate。
5. 在每张图内部，分别按专家汇总学生输出和对应位置的 DINOv2 最后一层 token。一个专家至少收到 2 个 token 才形成有效质心；一张图至少有 3 个有效专家才计算损失。
6. 学生质心和教师质心分别减去各自的专家均值，再逐质心做 L2 归一化。随后构造两张专家余弦 Gram 矩阵，只取一次上三角非对角元素，用 Smooth-L1 匹配。
7. assignment 和教师特征都 detach。几何损失直接训练 routed experts 及其上游表示，但不通过离散分组直接训练 `cluster_centers`。
8. 外层系数固定为 `expert_geometry_coeff: 0.25`。正向实验使用 `expert_geometry_teacher_roll: [0, 0]`。
9. 负对照在 16x16 DINO token 网格上使用固定 roll `[7, 11]`。它保留教师 token 的整体分布、训练计算和损失形式，但打乱学生专家分组与教师空间位置的正确对应。

## 4. 公平性控制

| 项目 | Fresh Multi-Align | 正向 TCEG | Roll 对照 |
| --- | --- | --- | --- |
| 规模 | Base | Base | Base |
| 参数/state keys | 父模型 | 与父模型相同 | 与父模型相同 |
| 训练步数 | 501K | 501K | 501K |
| 全局 batch | 256 | 256 | 256 |
| 学习率 | 1e-4 | 1e-4 | 1e-4 |
| `global_seed` | Group 01 旧训练器未生效 | 0，显式生效 | 0，显式生效 |
| 对齐层 | `[2, 3, 4]` | 相同 | 相同 |
| `proj_coeff` | 0.5 | 0.5 | 0.5 |
| TCEG 系数 | 0 | 0.25 | 0.25 |
| 教师 roll | 不适用 | `[0, 0]` | `[7, 11]` |
| 路由/专家/激活数 | 原 Multi-Align | 相同 | 相同 |
| 新增推理参数/FLOPs | 0 | 0 | 0 |
| 采样 | 300K/500K，250 steps | 相同 | 相同 |
| 评估 | 50K ImageNet，CFG 1.0/1.5，OpenAI evaluator | 相同 | 相同 |

TCEG 增加训练期的 expert-output trace、质心汇总和小型 Gram 矩阵计算，不增加采样路径。正向和 roll 对照使用相同的 rank-separated seed 和相同 wrapper phase 边界，因此两臂的随机流配对。Group 01 的 Fresh Multi-Align 在 seed 修复前已经启动，只能用于首轮筛选，不能支持小差值的因果结论。任何达标结果都必须补跑同 seed 基线并做多种子确认。

## 5. 实验与判定规则

正式结果表：

| Step | CFG | Fresh Multi-Align FID / IS | 正向 TCEG FID / IS | Roll 对照 FID / IS | 结论 |
| --- | --- | --- | --- | --- | --- |
| 300K | 1.0 | pending | pending | pending | pending |
| 300K | 1.5 | pending | pending | pending | pending |
| 500K | 1.0 | pending | pending | pending | pending |
| 500K | 1.5 | pending | pending | pending | pending |

预先固定主指标为 500K、CFG 1.5 FID。正向 TCEG 进入确认阶段必须同时满足：

1. 相对 Fresh Multi-Align 的绝对 FID 至少降低 0.15。
2. 相对 roll 对照至少降低 0.10，证明收益依赖正确的教师空间对应，而不只是额外几何损失。
3. 300K/500K、CFG 1.0/1.5 的其他三个点没有清楚的系统性退化。
4. loss、梯度、expert usage、每图有效专家数和路由熵没有崩溃或异常集中。

低于 0.10 的基线差异先视为无明确收益。单次训练只做筛选；正式结论需要至少三个训练种子、均值和波动，并交换 GPU 分组或做等价硬件控制。

## 6. 当前结果

生成指标尚未产生，不能提前判断方法有效。Group 01 正在占用两组 GPU；本组只完成定义、验证和排队，不与运行中的作业重叠。

实现验证将在分配 runtime wrapper 前完成并记录：

| 检查 | 当前状态 |
| --- | --- |
| 父模型默认路径、训练预测与 Multi-Align loss | tiny model bitwise identical；父模型默认 `return_expert_trace=False` 路径不变 |
| 评估输出与父 Multi-Align | tiny model bitwise identical |
| 正向/roll loss 与 routed-expert/upstream 梯度 | 两个损失均 finite 且不同（0.03016808 / 0.04393238）；三个被选 routed experts 均有非零梯度，upstream 梯度范数 0.06912121 |
| `cluster_centers` 的 geometry-only 梯度 | 梯度范数 0；assignment 已 detach |
| 全 unconditional 与退化 teacher | 均返回可反传的 0；`index_copy_` trace 已验证保留 source gradient |
| 参数、state keys、配置、wrapper、output guard | Base 参数 338,002,976、state keys 499；registry、两份 YAML、semantic/runtime wrapper 四向一致，runtime slots 为 `2.1`/GPU 0-3 和 `2.2`/GPU 4-7；两份 output guard 均为 `RESULT: OK` |
| `$check 1` | 通过：round 1 的 3/3 reviewers 均为 `ALL_CLEAN`，没有 blocker/error，也没有修复项；parent `py_compile`、`bash -n`、YAML/batch、output guard 和 `git diff --check` 全部通过。reviewers 只保留了逐图 GPU 同步、bf16 Gram 精度和 DDP 本地有效图像均值三项 warning，均记录为启动前的 profiling/数值检查项 |

## 7. 反思与风险

1. 专家质心会丢掉专家内部 token 的多模态结构。两个专家可能质心关系正确，但内部表示仍然混乱。
2. 每张图只有 256 个 token，12 个专家的分配可能不均。`min_tokens=2` 和 `min_experts=3` 能过滤退化样本，但也可能让实际参与损失的图像比例过低，必须记录 coverage。
3. 分组来自当前学生路由。早期随机 assignment 可能给专家输出施加不稳定目标，`0.25` 只是保守的首轮系数，不是已经证明的最优值。
4. 质心 Gram 只约束专家间相对角度，不约束尺度，也不直接改善负载均衡。若少数专家长期占据大部分 token，几何损失可能无法修复路由塌缩。
5. Roll 对照会改变每个学生专家看到的教师 token 集合，但不会生成完全独立的随机目标。若正向和对照都改善，只能说明几何正则可能有用，不能证明 DINO 空间对应是关键。
6. TCEG 与 activation-space specialization、DINO Gram structure 和 SPARE 的关系必须在论文中准确说明。只有稳定的生成收益和专家分析才能支撑期刊贡献，不能只靠方法组合的新名称。
7. 当前逐图质心实现会产生 host/device 同步；外层 bf16 autocast 也会让 Gram 矩阵乘法降为 bf16，而各 rank 对本地有效图像独立取均值会在有效数不同时形成 rank 等权。它们不改变正向/roll 的配对定义，但正式启动前应先测量耗时和 fp32 差异，并确认有效图像覆盖率足够高；若需要改实现，必须重新跑静态、梯度和 `$check` 验证。

## 8. 下一组规划

先等待 Group 01 完成，再运行已排队的正向/反向 SRSR，之后运行正向/roll TCEG。GPU 0-3 与 4-7 始终成组使用，不跨日期或跨组重叠。

如果 TCEG 达标：

1. 补跑同 seed Fresh Multi-Align，并增加至少两个独立 seed；交换正向/对照的 GPU 半组。
2. 做 `expert_geometry_coeff` 的 0.1/0.25/0.5 单因素消融。
3. 做 block、`min_tokens` 和 `min_experts` 消融，并报告每步有效图像比例，避免只保留容易样本。
4. 增加专家质心 Gram 误差、expert usage、路由熵/间隔、类-专家互信息和 timestep 曲线，证明 FID 收益确实来自专家分工。
5. Base 结论稳定后再扩到 L，不用模型规模掩盖机制不确定性。

如果 TCEG 没达标：不围绕 roll 位移或阈值做事后搜索。下一组先用短的离线可行性分析检查“哪个专家能降低当前 denoising residual”是否提供稳定监督；只有伪标签覆盖率、跨 step 一致性和额外训练计算都合理时，才进入 denoising-regret routing 的完整 Base 实验。
