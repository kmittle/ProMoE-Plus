# 2026_09_14 启动命令表 / Launch commands

本目录 12 个 run-time wrapper，全部基于 `repa` 分支。EC 混合训练 ecmix 11 个实验按 **2 卡/实验**排布（资源受限，每机同时跑 4 个实验）：`X.1`→GPU `0-1`，`X.2`→GPU `2-3`，`X.3`→GPU `4-5`，`X.4`→GPU `6-7`（四个 2-GPU 作业填满一台 8-GPU 服务器）；全局类中心 global_center 1 个实验按 **4 卡**排布：`4.1`→GPU `0-3`。两大族：EC 混合训练 ecmix ×11（只改 loss ×6、整步换成 EC step ×5）、全局类中心 global_center ×1。

**Global batch size 未变**：`total_train_batch_size=256` 全部保持不动，`train.py` 按 `train_batch_size = total_train_batch_size // world_size` 推导，2 卡时 per-GPU batch 为 128，4 卡时为 64，global 都是 256，实验公平性不受影响。

12 个实验分布在 4 台机器：前 2 台各 4 个（`X.1`–`X.4`），第 3 台 3 个（`3.1`–`3.3`，GPU `6-7` 空闲），第 4 台 1 个 4 卡实验（`4.1`，GPU `4-7` 空闲）。

指令和输出路径均为纯文本（无引号 / 反引号），可直接复制粘贴。

## 命令表 / Command table

| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| Slot 1.1 · GPU 0-1 · EC 混合只改 loss，路由对比损失 = 95% TC + 5% EC | repa | bash scripts/_run_times/2026_09_14/1.1-B_ecmix_loss_w0p05.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_loss_w0p05/ |
| Slot 1.2 · GPU 2-3 · EC 混合只改 loss，路由对比损失 = 90% TC + 10% EC | repa | bash scripts/_run_times/2026_09_14/1.2-B_ecmix_loss_w0p10.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_loss_w0p10/ |
| Slot 1.3 · GPU 4-5 · EC 混合只改 loss，路由对比损失 = 75% TC + 25% EC | repa | bash scripts/_run_times/2026_09_14/1.3-B_ecmix_loss_w0p25.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_loss_w0p25/ |
| Slot 1.4 · GPU 6-7 · EC 混合只改 loss，路由对比损失 = 50% TC + 50% EC | repa | bash scripts/_run_times/2026_09_14/1.4-B_ecmix_loss_w0p50.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_loss_w0p50/ |
| Slot 2.1 · GPU 0-1 · EC 混合只改 loss，路由对比损失全部换成 EC loss（100%） | repa | bash scripts/_run_times/2026_09_14/2.1-B_ecmix_loss_w1p00.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_loss_w1p00/ |
| Slot 2.2 · GPU 2-3 · EC 混合只改 loss，EC loss 占比 50% 余弦退火到 0（500K） | repa | bash scripts/_run_times/2026_09_14/2.2-B_ecmix_loss_cos0p50.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_loss_cos0p50/ |
| Slot 2.3 · GPU 4-5 · EC 混合整步替换，5% 的 step 换成 EC step（EC 路由 + EC loss） | repa | bash scripts/_run_times/2026_09_14/2.3-B_ecmix_step_p0p05.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_step_p0p05/ |
| Slot 2.4 · GPU 6-7 · EC 混合整步替换，10% 的 step 换成 EC step（EC 路由 + EC loss） | repa | bash scripts/_run_times/2026_09_14/2.4-B_ecmix_step_p0p10.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_step_p0p10/ |
| Slot 3.1 · GPU 0-1 · EC 混合整步替换，25% 的 step 换成 EC step（EC 路由 + EC loss） | repa | bash scripts/_run_times/2026_09_14/3.1-B_ecmix_step_p0p25.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_step_p0p25/ |
| Slot 3.2 · GPU 2-3 · EC 混合整步替换，50% 的 step 换成 EC step（EC 路由 + EC loss） | repa | bash scripts/_run_times/2026_09_14/3.2-B_ecmix_step_p0p50.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_step_p0p50/ |
| Slot 3.3 · GPU 4-5 · EC 混合整步替换，EC step 占比 50% 余弦退火到 0（500K） | repa | bash scripts/_run_times/2026_09_14/3.3-B_ecmix_step_cos0p50.sh | outputs/ProMoE_TC_B_ecmix/004_ProMoE_B_ecmix_step_cos0p50/ |
| Slot 4.1 · GPU 0-3 · 全局类中心，路由对比损失的专家类中心在整个 global batch 上计算（all-reduce，4 卡） | repa | bash scripts/_run_times/2026_09_14/4.1-B_global_center.sh | outputs/ProMoE_TC_B_global_center/004_ProMoE_B_global_center/ |
