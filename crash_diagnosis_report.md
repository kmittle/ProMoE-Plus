# 5 个训练崩溃原因诊断报告

调查对象: `training_logs/` 下 plans 02, 03, 04, 06, 08 的训练崩溃
对照基线: plans 01, 05, 07 (健康运行)

---

## Plan 02 — `repa_cross_global_block` 后期数值发散

- **健康期**: 0 → 349k 步,MSE 稳定在 0.8–0.9,REPA ≈ −0.4
- **崩溃点**: step 371990 MSE=10 → 375990 MSE=298 → 持续停留在 100–500 区间
- **特征**: REPA loss 全程正常(−0.3 到 −0.5),**只有 MSE 爆炸**;没有 restart 迹象
- **根因**: 晚期 MSE 收敛到较低值使 loss landscape 变陡,`global_block` cross-alignment 引入的梯度在极小 MSE 附近不稳定。典型的长期训练发散,不是代码 bug

## Plan 03 — `repa_cross_expert_local` 训练逆转

- **初期**: MSE 从 1.68 下降到 **0.88** (step 8990)
- **逆转**: step 40k–90k 期间 MSE 从 0.9 **反向上升到 1.7**
- **终态**: step 90k–415k 完全卡死在 MSE ≈ 1.7(比什么都不输出还差)
- **对照**: 同配置的 plan 07 (MoS 版) MSE=0.77,正常训练
- **根因**: non-MoS `cross_expert_local` 的梯度耦合有 bug:REPA loss 仍在缓慢下降 (−0.45)但主任务梯度被破坏。很可能是 `ExpertLocalAttention` 模块在 encoder_depth 之后消耗了 `x` 的副本但未将结果参与主分支梯度流,或 `block.mlp._expert_indices` 的 detach 行为与 MoS 版不同。plan 07 的 MoS 结构顺带绕过了这个路径

## Plan 04 — `repa_cross_proto` REPA loss 间歇性爆炸

- **首次 spike**: step 560 (mos_repa_loss=1079,下一步恢复到 −0.39)
- **幅度**: 从百到 **−12103** 再到 1079 之间跳
- **MSE**: 始终有界 (~0.85),说明 grad-clip 在保护主任务
- **commit 4ad8a2f 的修复**: `proto_sim.clamp(min=0)` + `/W.sum()` 归一化已在 plan 04 启动前就位(提交于 04-11 23:48,训练启动于 04-12 23:29)
- **根因**: **修复不充分**。理论上 W∈[0,1]、cos_sim∈[−1,1]、loss∈[−1,1],但实测 loss 达到 10⁵ 量级。很可能是 bf16 autocast 下 `F.normalize` 精度损失使某些 token 的归一化向量模长 ≠ 1,或 `compute_router` 中缓存的 `_proto_sim` 是 fp32 而 `x` 被 autocast 转 bf16 后在 `_build_proto_cross_weights` 里的 dtype promotion 产生了反常行为。推荐加上 **`loss = loss.clamp(-1.0, 1.0)`** 强制约束 + 打印触发 spike 时的 W/cos_sim 统计

## Plan 06 — `mos_cross_global_block` 崩溃 + 坏断点续训

- **第一次运行** (04-11 20:35 启动): 正常训练到 step 226440 → step 226450 MSE 从 0.87 → 10 → 持续飙升到 >60(到 step 228100)
- **断点保存**: step 228000 保存的 ckpt 已经是发散权重
- **第二次运行** (04-12 23:27): 从坏断点 resume,MSE 起始就是 56,直接进入稳态 MSE ≈ 4000–5000
- **根因**: 复合故障。首次与 plan 02 类似的晚期不稳定(`global_block` 变体内在问题);resume 机制未检测到 checkpoint 已发散直接续训

## Plan 08 — `mos_cross_proto` 与 plan 04 同病

- **首次 spike**: step 400 (238),随后 997, 3063, 355703
- **MSE**: 始终有界 (~0.85–2.1)
- **根因**: 与 plan 04 完全相同。MoS 版还引入了额外的 `selected_block_weight` 乘法和 `num_cond.clamp(min=1)` 除法,这些 per-token 加权在早期 prototype 不稳定时会放大数值问题

---

## 建议修复优先级

| 优先级 | Plan | 修复方向 |
|-------|------|---------|
| P0 | 04, 08 | 在 `compute_cross_align_loss` 末尾加 `loss = torch.clamp(loss, -1.0, 1.0)` 作为硬保护;添加诊断日志(触发 spike 时 dump W.max/sum, cos_sim.max/min, proto_sim.max) |
| P0 | 03 | 审查 `ExpertLocalAttention` 前向里 `x` 是否被 in-place 修改或 detach,验证 encoder_depth 处梯度能否正确回传到主分支 |
| P1 | 06 | 对 global_block 加 resume 保护: 若 load 的 ckpt 训练日志显示发散(最近 N 步 MSE > 阈值)则拒绝 resume;同时排查 `global_block` 变体在 CE loss ≈ 0.8 附近的梯度稳定性 |
| P1 | 02 | 与 plan 06 同源问题,建议对所有 `cross_global_block` 运行启用更激进的 `max_grad_norm`(如 0.1)或 warmup 重启 |

Plan 01/05/07 (`cross_global_pre` / `mos_cross_global_pre` / `mos_cross_expert_local`) 为健康对照,可作为修复后回归基线。
