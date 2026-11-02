| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| 实验 1：双 teacher KL：真实与估计干净图，各 2.5e-4，无矫正器；300K 双 CFG FID 达标后续训（4 卡） | repa | bash scripts/_run_times/2026_11_02/B_clean_route_dual_teacher.sh | outputs/ProMoE_TC_B_clean_route_dual_teacher/004_ProMoE_B_clean_route_dual_teacher/ |
| 实验 2：clean teacher：仅 KL 使用 MLP + Sigmoid 投影头；300K 双 CFG FID 达标后续训（4 卡） | repa | bash scripts/_run_times/2026_11_02/B_clean_route_student_ratio0_align_projector.sh | outputs/ProMoE_TC_B_clean_route_projector/004_ProMoE_B_clean_route_student_ratio0_align_projector/ |
| 实验 6：clean teacher＋路由矫正：KL 对本层原分数停止直接梯度；300K 双 CFG FID 达标后续训（4 卡） | repa | bash scripts/_run_times/2026_11_02/B_clean_route_student_ratio0_corr_router_kl_detach_base.sh | outputs/ProMoE_TC_B_clean_route_corr_router/004_ProMoE_B_clean_route_student_ratio0_corr_router_kl_detach_base/ |
| 实验 3：clean teacher：当前 token 矫正器，KL 监督 s+d；300K 双 CFG FID 达标后续训（4 卡） | repa | bash scripts/_run_times/2026_11_02/B_clean_route_student_ratio0_token_corrector.sh | outputs/ProMoE_TC_B_clean_route_token_corrector/004_ProMoE_B_clean_route_student_ratio0_token_corrector/ |
| 实验 4：矫正器 only：当前步速度估计参考，推理增加前向；300K 双 CFG FID 达标后续训（4 卡） | repa | bash scripts/_run_times/2026_11_02/B_corr_router_current_ref.sh | outputs/ProMoE_TC_B_corr_router/004_ProMoE_B_corr_router_current_ref/ |
| 实验 5：估计干净 teacher＋路由矫正：仅学生参考保留 50% 置零；300K 双 CFG FID 达标后续训（4 卡） | repa | bash scripts/_run_times/2026_11_02/B_estimated_route_student_corr_router.sh | outputs/ProMoE_TC_B_clean_route_corr_router/004_ProMoE_B_estimated_route_student_corr_router/ |
