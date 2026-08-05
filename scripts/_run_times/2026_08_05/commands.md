# 2026_08_05 启动命令表 / Launch commands

本目录 23 个 run-time wrapper，全部基于 `repa` 分支。物理映射：`X.1` → GPU 0-3，`X.2` → GPU 4-7（两个 4-GPU 作业填满一台 8-GPU 服务器）。三大族：EC-BC 异构专家 ×1、LS-Reg 路由对比标签平滑 ×16、专家参数正则 expert_contra ×6。

指令和输出路径均为纯文本（无引号 / 反引号），可直接复制粘贴。

## 命令表 / Command table

| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| Slot 1.1 · GPU 0-3 · EC-BC 批展平 Expert-Choice + 异构宽度专家 | repa | bash scripts/_run_times/2026_08_05/1.1-B_ec_bc_hetero.sh | outputs/ProMoE_EC_BC_hetero_B/004_ProMoE_B_EC_BC_hetero/ |
| Slot 1.2 · GPU 4-7 · LS-Reg 对角修正 idea-1，强度 0.30 | repa | bash scripts/_run_times/2026_08_05/1.2-B_lsreg_diag_idea1_s0p30.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p30/ |
| Slot 2.1 · GPU 0-3 · LS-Reg 对角修正 idea-1，强度 0.40 | repa | bash scripts/_run_times/2026_08_05/2.1-B_lsreg_diag_idea1_s0p40.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p40/ |
| Slot 2.2 · GPU 4-7 · LS-Reg 对角修正 inverse，强度 0.30 | repa | bash scripts/_run_times/2026_08_05/2.2-B_lsreg_diag_inv_s0p30.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p30/ |
| Slot 3.1 · GPU 0-3 · LS-Reg 对角修正 inverse，强度 0.40 | repa | bash scripts/_run_times/2026_08_05/3.1-B_lsreg_diag_inv_s0p40.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p40/ |
| Slot 3.2 · GPU 4-7 · LS-Reg 对角修正 idea-1，强度 0.05 | repa | bash scripts/_run_times/2026_08_05/3.2-B_lsreg_diag_idea1_s0p05.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p05/ |
| Slot 4.1 · GPU 0-3 · LS-Reg 对角修正 idea-1，强度 0.10 | repa | bash scripts/_run_times/2026_08_05/4.1-B_lsreg_diag_idea1_s0p10.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p10/ |
| Slot 4.2 · GPU 4-7 · LS-Reg 对角修正 idea-1，强度 0.15 | repa | bash scripts/_run_times/2026_08_05/4.2-B_lsreg_diag_idea1_s0p15.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p15/ |
| Slot 5.1 · GPU 0-3 · LS-Reg 对角修正 idea-1，强度 0.20 | repa | bash scripts/_run_times/2026_08_05/5.1-B_lsreg_diag_idea1_s0p20.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_idea1_s0p20/ |
| Slot 5.2 · GPU 4-7 · LS-Reg 对角修正 inverse，强度 0.05 | repa | bash scripts/_run_times/2026_08_05/5.2-B_lsreg_diag_inv_s0p05.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p05/ |
| Slot 6.1 · GPU 0-3 · LS-Reg 对角修正 inverse，强度 0.10 | repa | bash scripts/_run_times/2026_08_05/6.1-B_lsreg_diag_inv_s0p10.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p10/ |
| Slot 6.2 · GPU 4-7 · LS-Reg 对角修正 inverse，强度 0.15 | repa | bash scripts/_run_times/2026_08_05/6.2-B_lsreg_diag_inv_s0p15.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p15/ |
| Slot 7.1 · GPU 0-3 · LS-Reg 对角修正 inverse，强度 0.20 | repa | bash scripts/_run_times/2026_08_05/7.1-B_lsreg_diag_inv_s0p20.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_diag_inv_s0p20/ |
| Slot 7.2 · GPU 4-7 · LS-Reg 固定标签平滑 ε=0.05 | repa | bash scripts/_run_times/2026_08_05/7.2-B_lsreg_fixed0p05.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_fixed0p05/ |
| Slot 8.1 · GPU 0-3 · LS-Reg 固定标签平滑 ε=0.20 | repa | bash scripts/_run_times/2026_08_05/8.1-B_lsreg_fixed0p20.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_fixed0p20/ |
| Slot 8.2 · GPU 4-7 · LS-Reg 固定标签平滑 ε=0.40 | repa | bash scripts/_run_times/2026_08_05/8.2-B_lsreg_fixed0p40.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_fixed0p40/ |
| Slot 9.1 · GPU 0-3 · LS-Reg 动态标签平滑 dyn_over | repa | bash scripts/_run_times/2026_08_05/9.1-B_lsreg_dynover.sh | outputs/ProMoE_TC_B_lsreg/004_ProMoE_B_lsreg_dynover/ |
| Slot 9.2 · GPU 4-7 · 专家参数正则 param，不拼 bias（include_bias=False） | repa | bash scripts/_run_times/2026_08_05/9.2-B_expert_contra_param_nobias.sh | outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_nobias/ |
| Slot 10.1 · GPU 0-3 · 专家参数正则 param_cos（余弦 relu 排斥，无 τ） | repa | bash scripts/_run_times/2026_08_05/10.1-B_expert_contra_param_cos.sh | outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_cos/ |
| Slot 10.2 · GPU 4-7 · 专家参数正则 param，纳入 shared expert | repa | bash scripts/_run_times/2026_08_05/10.2-B_expert_contra_param_shared.sh | outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_shared/ |
| Slot 11.1 · GPU 0-3 · 专家参数正则 param，纳入 shared + uncond expert | repa | bash scripts/_run_times/2026_08_05/11.1-B_expert_contra_param_shared_uncond.sh | outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_shared_uncond/ |
| Slot 11.2 · GPU 4-7 · 专家参数正则 param，RBF 带宽 τ=0.07（÷10 死区端） | repa | bash scripts/_run_times/2026_08_05/11.2-B_expert_contra_param_tau0p07.sh | outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_tau0p07/ |
| Slot 12.1 · GPU 0-3 · 专家参数正则 param，RBF 带宽 τ=7（×10 激活端） | repa | bash scripts/_run_times/2026_08_05/12.1-B_expert_contra_param_tau7.sh | outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_tau7/ |
