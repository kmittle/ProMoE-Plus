| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| Clean-Route student_ratio0：100K 后开启路由蒸馏，最终系数 5e-4；训至 300K → 双 CFG 评估 → 续训至 500K → 双 CFG 评估，各 50K 张（4 卡） | repa | bash scripts/_run_times/2026_10_30/B_clean_route_student_ratio0_align_delay100k.sh | outputs/ProMoE_TC_B_clean_route/004_ProMoE_B_clean_route_student_ratio0_align_delay100k/ |
| Clean-Route student_ratio0：路由蒸馏系数在 200K 内线性 warmup，最终系数 5e-4；训至 300K → 双 CFG 评估 → 续训至 500K → 双 CFG 评估，各 50K 张（4 卡） | repa | bash scripts/_run_times/2026_10_30/B_clean_route_student_ratio0_align_warmup200k.sh | outputs/ProMoE_TC_B_clean_route/004_ProMoE_B_clean_route_student_ratio0_align_warmup200k/ |
| Clean-Route＋路由矫正统一重评：300K/500K × CFG 1.0/1.5，全部使用 CFG fusion 后的速度，每组重新采样 50K（4 卡） | repa | bash scripts/_run_times/2026_10_30/B_clean_route_student_ratio0_corr_router_cfg_ref.sh | outputs/ProMoE_TC_B_clean_route_corr_router/004_ProMoE_B_clean_route_student_ratio0_corr_router_cfg_ref/ |
| 路由矫正统一重评：300K/500K × CFG 1.0/1.5，全部使用 CFG fusion 后的速度，每组重新采样 50K（4 卡） | repa | bash scripts/_run_times/2026_10_30/B_corr_router_cfg_ref.sh | outputs/ProMoE_TC_B_corr_router/004_ProMoE_B_corr_router_cfg_ref/ |
| ProMoE-B 基础 token Top-3＋shared，专家宽度 1536，cond/uncond FFN 宽度和 6144/3072；训至 300K → 双 CFG 评估 → 续训至 500K → 双 CFG 评估，各 50K 张（4 卡） | repa | bash scripts/_run_times/2026_10_30/B_top3.sh | outputs/ProMoE_TC_B/004_ProMoE_B_top3/ |
