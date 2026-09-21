| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| Slot 1.1 · GPU 0-1 · 专家 pooling 表征余弦正则 lam=0.1，叠加 LS-Reg（ls_diag_strength=0.05） | repa | bash scripts/_run_times/2026_09_21/1.1-B_dualreg_ls0p05_lam0p1.sh | outputs/ProMoE_TC_B_dualreg/004_ProMoE_B_dualreg_ls0p05_lam0p1/ |
| Slot 1.2 · GPU 2-3 · 专家 pooling 表征余弦正则 lam=0.3，叠加 LS-Reg（ls_diag_strength=0.05） | repa | bash scripts/_run_times/2026_09_21/1.2-B_dualreg_ls0p05_lam0p3.sh | outputs/ProMoE_TC_B_dualreg/004_ProMoE_B_dualreg_ls0p05_lam0p3/ |
| Slot 1.3 · GPU 4-5 · 专家 pooling 表征余弦正则 lam=1.0，叠加 LS-Reg（ls_diag_strength=0.05） | repa | bash scripts/_run_times/2026_09_21/1.3-B_dualreg_ls0p05_lam1p0.sh | outputs/ProMoE_TC_B_dualreg/004_ProMoE_B_dualreg_ls0p05_lam1p0/ |
| Slot 1.4 · GPU 6-7 · 专家 pooling 表征余弦正则 lam=3.0，叠加 LS-Reg（ls_diag_strength=0.05） | repa | bash scripts/_run_times/2026_09_21/1.4-B_dualreg_ls0p05_lam3p0.sh | outputs/ProMoE_TC_B_dualreg/004_ProMoE_B_dualreg_ls0p05_lam3p0/ |
| Slot 2.1 · GPU 0-1 · 专家 pooling 表征余弦正则 lam=0.1，仅表征正则（ls_diag_strength=0（关闭）） | repa | bash scripts/_run_times/2026_09_21/2.1-B_dualreg_lsoff_lam0p1.sh | outputs/ProMoE_TC_B_dualreg/004_ProMoE_B_dualreg_lsoff_lam0p1/ |
| Slot 2.2 · GPU 2-3 · 专家 pooling 表征余弦正则 lam=0.3，仅表征正则（ls_diag_strength=0（关闭）） | repa | bash scripts/_run_times/2026_09_21/2.2-B_dualreg_lsoff_lam0p3.sh | outputs/ProMoE_TC_B_dualreg/004_ProMoE_B_dualreg_lsoff_lam0p3/ |
| Slot 2.3 · GPU 4-5 · 专家 pooling 表征余弦正则 lam=1.0，仅表征正则（ls_diag_strength=0（关闭）） | repa | bash scripts/_run_times/2026_09_21/2.3-B_dualreg_lsoff_lam1p0.sh | outputs/ProMoE_TC_B_dualreg/004_ProMoE_B_dualreg_lsoff_lam1p0/ |
| Slot 2.4 · GPU 6-7 · 专家 pooling 表征余弦正则 lam=3.0，仅表征正则（ls_diag_strength=0（关闭）） | repa | bash scripts/_run_times/2026_09_21/2.4-B_dualreg_lsoff_lam3p0.sh | outputs/ProMoE_TC_B_dualreg/004_ProMoE_B_dualreg_lsoff_lam3p0/ |
