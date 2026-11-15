| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| 矫正器 only：带历史参考预测当前速度，再用当前估计矫正正式前向（4 卡） | repa | bash scripts/_run_times/2026_11_15/B_corr_router_current_history.sh | outputs/ProMoE_TC_B_corr_router/004_ProMoE_B_corr_router_current_history/ |
| 矫正器 only：当前点两轮参考估计，再进行正式前向（4 卡） | repa | bash scripts/_run_times/2026_11_15/B_corr_router_current_refined.sh | outputs/ProMoE_TC_B_corr_router/004_ProMoE_B_corr_router_current_refined/ |
| 矫正器 only：上一参考固定训练间隔为 4，对照多间隔训练（4 卡） | repa | bash scripts/_run_times/2026_11_15/B_corr_router_previous_gap4.sh | outputs/ProMoE_TC_B_corr_router/004_ProMoE_B_corr_router_previous_gap4/ |
| 估计干净 teacher＋路由矫正，KL 对原始路由分数停止直接梯度（4 卡） | repa | bash scripts/_run_times/2026_11_15/B_estimated_route_student_corr_router_kl_detach_base.sh | outputs/ProMoE_TC_B_clean_route_corr_router/004_ProMoE_B_estimated_route_student_corr_router_kl_detach_base/ |
| 估计 teacher＋KL detach-base，取消随机参考丢弃，仅当前 t=1000 的学生参考置零（4 卡） | repa | bash scripts/_run_times/2026_11_15/B_estimated_route_student_corr_router_kl_detach_base_no_ref_drop.sh | outputs/ProMoE_TC_B_clean_route_corr_router/004_ProMoE_B_estimated_route_student_corr_router_kl_detach_base_no_ref_drop/ |
