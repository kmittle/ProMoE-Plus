| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| 续训：历史参考辅助当前步估计，复核 300K 双 FID 后补 500K（4 卡） | repa | bash scripts/_run_times/2026_11_23/B_corr_router_current_history_resume.sh | outputs/ProMoE_TC_B_corr_router/004_ProMoE_B_corr_router_current_history/ |
| 续训：上一参考固定间隔 4，复核 300K 双 FID 后补 500K（4 卡） | repa | bash scripts/_run_times/2026_11_23/B_corr_router_previous_gap4_resume.sh | outputs/ProMoE_TC_B_corr_router/004_ProMoE_B_corr_router_previous_gap4/ |
| 续训：估计 teacher＋KL detach-base，取消随机参考丢弃，复核 300K 双 FID 后补 500K（4 卡） | repa | bash scripts/_run_times/2026_11_23/B_estimated_route_student_corr_router_kl_detach_base_no_ref_drop_resume.sh | outputs/ProMoE_TC_B_clean_route_corr_router/004_ProMoE_B_estimated_route_student_corr_router_kl_detach_base_no_ref_drop/ |
| 续训：估计 teacher＋KL detach-base，复核 300K 双 FID 后补 500K（4 卡） | repa | bash scripts/_run_times/2026_11_23/B_estimated_route_student_corr_router_kl_detach_base_resume.sh | outputs/ProMoE_TC_B_clean_route_corr_router/004_ProMoE_B_estimated_route_student_corr_router_kl_detach_base/ |
| 续训：完整组合 L，补齐 500K，无 300K FID 门槛（4 卡） | repa | bash scripts/_run_times/2026_11_23/L_clean_route_student_ratio0_corr_router_resume.sh | outputs/ProMoE_TC_L_clean_route_corr_router/004_ProMoE_L_clean_route_student_ratio0_corr_router/ |
| 续训：完整组合 XL，补齐 500K，无 300K FID 门槛（4 卡） | repa | bash scripts/_run_times/2026_11_23/XL_clean_route_student_ratio0_corr_router_resume.sh | outputs/ProMoE_TC_XL_clean_route_corr_router/004_ProMoE_XL_clean_route_student_ratio0_corr_router/ |
