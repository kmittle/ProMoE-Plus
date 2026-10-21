| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| ProMoE-B adaptive-depth q=0.3，每张图内排序、仅 looped token 的 p 训练打分器（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_adepth_q0p3_p_only.sh | outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p3_p_only/ |
| ProMoE-B adaptive-depth q=0.3，保留原 STE、每张图内独立排序（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_adepth_q0p3_per_image_ste.sh | outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p3_per_image_ste/ |
| ProMoE-B clean-route 路由蒸馏，student 侧对比学习，teacher 为干净 latent（t=0），align_lam 5e-4、tau 0.07（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_clean_route_student_ratio0.sh | outputs/ProMoE_TC_B_clean_route/004_ProMoE_B_clean_route_student_ratio0/ |
| ProMoE-B clean-route 路由蒸馏，student 侧对比学习，teacher 噪声为 student 的一半（同一噪声），align_lam 5e-4、tau 0.07（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_clean_route_student_ratio0p5.sh | outputs/ProMoE_TC_B_clean_route/004_ProMoE_B_clean_route_student_ratio0p5/ |
| ProMoE-B clean-route 路由蒸馏，teacher 侧对比学习（只更新原型），teacher 为干净 latent（t=0），align_lam 5e-4、tau 0.07（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_clean_route_teacher_ratio0.sh | outputs/ProMoE_TC_B_clean_route/004_ProMoE_B_clean_route_teacher_ratio0/ |
| ProMoE-B clean-route 路由蒸馏，teacher 侧对比学习（只更新原型），teacher 噪声为 student 的一半（同一噪声），align_lam 5e-4、tau 0.07（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_clean_route_teacher_ratio0p5.sh | outputs/ProMoE_TC_B_clean_route/004_ProMoE_B_clean_route_teacher_ratio0p5/ |
| ProMoE-B 共享分支改为 4 个 image-wise 专家，cross-attn pooling 和图级对比路由（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_image_shared_e4.sh | outputs/ProMoE_TC_B_image_shared/004_ProMoE_B_image_shared_e4/ |
| ProMoE-B 保留共享专家并新增 4 个仅 cond 使用的图级专家，FFN 宽度按 1/2+1/4+1/4 分配（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_shared_plus_cond_image_e4.sh | outputs/ProMoE_TC_B_shared_plus_cond_image/004_ProMoE_B_shared_plus_cond_image_e4/ |
