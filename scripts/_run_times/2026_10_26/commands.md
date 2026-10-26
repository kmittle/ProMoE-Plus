| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| ProMoE-B adaptive-depth q=0.3 图内排序 p_only，从自身 300K checkpoint 续训至 500K 并评估，使用 data2 latent 路径（4 卡） | repa | bash scripts/_run_times/2026_10_26/B_adepth_q0p3_p_only_resume_500k.sh | outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p3_p_only/ |
| ProMoE-B clean-route student_ratio0 加矫正 router，从自身 300K checkpoint 续训至 500K 并评估，用户指定探索性续训、使用 data2 latent 路径（4 卡） | repa | bash scripts/_run_times/2026_10_26/B_clean_route_student_ratio0_corr_router_resume_500k.sh | outputs/ProMoE_TC_B_clean_route_corr_router/004_ProMoE_B_clean_route_student_ratio0_corr_router/ |
| ProMoE-B 矫正 router，从自身 300K checkpoint 续训至 500K 并评估，用户指定探索性续训、使用 data2 latent 路径（4 卡） | repa | bash scripts/_run_times/2026_10_26/B_corr_router_resume_500k.sh | outputs/ProMoE_TC_B_corr_router/004_ProMoE_B_corr_router/ |
