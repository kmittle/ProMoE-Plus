# 命令表 / Command table — 2026_07_08

合并自 2026_07_08(dagfuse shared-expert augmentation,slots 1.1–7.2,14 个) + 2026_07_07(lsreg 标签平滑 + diag 对角线干预,重编号为 slots 8.1–15.2,16 个)。共 30 个实验。

- **Slots 1–7**:dagfuse 补强 shared 专家家族(dense / densenet / sharedroute / region)。
- **Slots 8–15**:lsreg 路由对比损失标签平滑(fixed/dyn)+ diag 相似度矩阵对角线干预。

实验描述 | git分支 | 启动命令 | 输出位置
--- | --- | --- | ---
Slot 1.1 · GPU 0-3 · ProMoE-B dagfuse dense cond | repa | `bash scripts/_run_times/2026_07_08/1.1-B_dagfuse_dense_cond.sh` | `outputs/ProMoE_TC_B_dagfuse_dense/004_ProMoE_B_dagfuse_dense_cond/`
Slot 1.2 · GPU 4-7 · ProMoE-B dagfuse dense all | repa | `bash scripts/_run_times/2026_07_08/1.2-B_dagfuse_dense_all.sh` | `outputs/ProMoE_TC_B_dagfuse_dense/004_ProMoE_B_dagfuse_dense_all/`
Slot 2.1 · GPU 0-3 · ProMoE-B dagfuse densenet cond | repa | `bash scripts/_run_times/2026_07_08/2.1-B_dagfuse_densenet_cond.sh` | `outputs/ProMoE_TC_B_dagfuse_densenet/004_ProMoE_B_dagfuse_densenet_cond/`
Slot 2.2 · GPU 4-7 · ProMoE-B dagfuse densenet all | repa | `bash scripts/_run_times/2026_07_08/2.2-B_dagfuse_densenet_all.sh` | `outputs/ProMoE_TC_B_dagfuse_densenet/004_ProMoE_B_dagfuse_densenet_all/`
Slot 3.1 · GPU 0-3 · ProMoE-B dagfuse sharedroute cond top1 | repa | `bash scripts/_run_times/2026_07_08/3.1-B_dagfuse_sharedroute_cond_top1.sh` | `outputs/ProMoE_TC_B_dagfuse_sharedroute/004_ProMoE_B_dagfuse_sharedroute_cond_top1/`
Slot 3.2 · GPU 4-7 · ProMoE-B dagfuse sharedroute cond top2 | repa | `bash scripts/_run_times/2026_07_08/3.2-B_dagfuse_sharedroute_cond_top2.sh` | `outputs/ProMoE_TC_B_dagfuse_sharedroute/004_ProMoE_B_dagfuse_sharedroute_cond_top2/`
Slot 4.1 · GPU 0-3 · ProMoE-B dagfuse sharedroute all top1 | repa | `bash scripts/_run_times/2026_07_08/4.1-B_dagfuse_sharedroute_all_top1.sh` | `outputs/ProMoE_TC_B_dagfuse_sharedroute/004_ProMoE_B_dagfuse_sharedroute_all_top1/`
Slot 4.2 · GPU 4-7 · ProMoE-B dagfuse sharedroute all top2 | repa | `bash scripts/_run_times/2026_07_08/4.2-B_dagfuse_sharedroute_all_top2.sh` | `outputs/ProMoE_TC_B_dagfuse_sharedroute/004_ProMoE_B_dagfuse_sharedroute_all_top2/`
Slot 5.1 · GPU 0-3 · ProMoE-B dagfuse region shared cond dag | repa | `bash scripts/_run_times/2026_07_08/5.1-B_dagfuse_region_shared_cond_dag.sh` | `outputs/ProMoE_TC_B_dagfuse_region/004_ProMoE_B_dagfuse_region_shared_cond_dag/`
Slot 5.2 · GPU 4-7 · ProMoE-B dagfuse region shared all dag | repa | `bash scripts/_run_times/2026_07_08/5.2-B_dagfuse_region_shared_all_dag.sh` | `outputs/ProMoE_TC_B_dagfuse_region/004_ProMoE_B_dagfuse_region_shared_all_dag/`
Slot 6.1 · GPU 0-3 · ProMoE-B dagfuse region shared cond softmax | repa | `bash scripts/_run_times/2026_07_08/6.1-B_dagfuse_region_shared_cond_softmax.sh` | `outputs/ProMoE_TC_B_dagfuse_region/004_ProMoE_B_dagfuse_region_shared_cond_softmax/`
Slot 6.2 · GPU 4-7 · ProMoE-B dagfuse region shared all softmax | repa | `bash scripts/_run_times/2026_07_08/6.2-B_dagfuse_region_shared_all_softmax.sh` | `outputs/ProMoE_TC_B_dagfuse_region/004_ProMoE_B_dagfuse_region_shared_all_softmax/`
Slot 7.1 · GPU 0-3 · ProMoE-B dagfuse region resid dag | repa | `bash scripts/_run_times/2026_07_08/7.1-B_dagfuse_region_resid_dag.sh` | `outputs/ProMoE_TC_B_dagfuse_region/004_ProMoE_B_dagfuse_region_resid_dag/`
Slot 7.2 · GPU 4-7 · ProMoE-B dagfuse region resid softmax | repa | `bash scripts/_run_times/2026_07_08/7.2-B_dagfuse_region_resid_softmax.sh` | `outputs/ProMoE_TC_B_dagfuse_region/004_ProMoE_B_dagfuse_region_resid_softmax/`
Slot 8.1 · GPU 0-3 · ProMoE-B lsreg fixed0p05 | repa | `bash scripts/_run_times/2026_07_08/8.1-B_lsreg_fixed0p05.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_fixed0p05/`
Slot 8.2 · GPU 4-7 · ProMoE-B lsreg fixed0p10 | repa | `bash scripts/_run_times/2026_07_08/8.2-B_lsreg_fixed0p10.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_fixed0p10/`
Slot 9.1 · GPU 0-3 · ProMoE-B lsreg fixed0p20 | repa | `bash scripts/_run_times/2026_07_08/9.1-B_lsreg_fixed0p20.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_fixed0p20/`
Slot 9.2 · GPU 4-7 · ProMoE-B lsreg fixed0p30 | repa | `bash scripts/_run_times/2026_07_08/9.2-B_lsreg_fixed0p30.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_fixed0p30/`
Slot 10.1 · GPU 0-3 · ProMoE-B lsreg fixed0p40 | repa | `bash scripts/_run_times/2026_07_08/10.1-B_lsreg_fixed0p40.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_fixed0p40/`
Slot 10.2 · GPU 4-7 · ProMoE-B lsreg dynboth | repa | `bash scripts/_run_times/2026_07_08/10.2-B_lsreg_dynboth.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_dynboth/`
Slot 11.1 · GPU 0-3 · ProMoE-B lsreg dynunder | repa | `bash scripts/_run_times/2026_07_08/11.1-B_lsreg_dynunder.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_dynunder/`
Slot 11.2 · GPU 4-7 · ProMoE-B lsreg dynover | repa | `bash scripts/_run_times/2026_07_08/11.2-B_lsreg_dynover.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_dynover/`
Slot 12.1 · GPU 0-3 · ProMoE-B lsreg diag idea1 s0p05 | repa | `bash scripts/_run_times/2026_07_08/12.1-B_lsreg_diag_idea1_s0p05.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p05/`
Slot 12.2 · GPU 4-7 · ProMoE-B lsreg diag idea1 s0p10 | repa | `bash scripts/_run_times/2026_07_08/12.2-B_lsreg_diag_idea1_s0p10.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p10/`
Slot 13.1 · GPU 0-3 · ProMoE-B lsreg diag idea1 s0p15 | repa | `bash scripts/_run_times/2026_07_08/13.1-B_lsreg_diag_idea1_s0p15.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p15/`
Slot 13.2 · GPU 4-7 · ProMoE-B lsreg diag idea1 s0p20 | repa | `bash scripts/_run_times/2026_07_08/13.2-B_lsreg_diag_idea1_s0p20.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p20/`
Slot 14.1 · GPU 0-3 · ProMoE-B lsreg diag inv s0p05 | repa | `bash scripts/_run_times/2026_07_08/14.1-B_lsreg_diag_inv_s0p05.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p05/`
Slot 14.2 · GPU 4-7 · ProMoE-B lsreg diag inv s0p10 | repa | `bash scripts/_run_times/2026_07_08/14.2-B_lsreg_diag_inv_s0p10.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p10/`
Slot 15.1 · GPU 0-3 · ProMoE-B lsreg diag inv s0p15 | repa | `bash scripts/_run_times/2026_07_08/15.1-B_lsreg_diag_inv_s0p15.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p15/`
Slot 15.2 · GPU 4-7 · ProMoE-B lsreg diag inv s0p20 | repa | `bash scripts/_run_times/2026_07_08/15.2-B_lsreg_diag_inv_s0p20.sh` | `outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p20/`
