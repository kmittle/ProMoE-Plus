| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| ProMoE-B adaptive-depth q=0.3，每张图内排序、仅 looped token 的 p 训练打分器（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_adepth_q0p3_p_only.sh | outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p3_p_only/ |
| ProMoE-B adaptive-depth q=0.3，保留原 STE、每张图内独立排序（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_adepth_q0p3_per_image_ste.sh | outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p3_per_image_ste/ |
| ProMoE-B 共享分支改为 4 个 image-wise 专家，cross-attn pooling 和图级对比路由（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_image_shared_e4.sh | outputs/ProMoE_TC_B_image_shared/004_ProMoE_B_image_shared_e4/ |
| ProMoE-B 保留共享专家并新增 4 个仅 cond 使用的图级专家，FFN 宽度按 1/2+1/4+1/4 分配（4 卡） | repa | bash scripts/_run_times/2026_10_21/B_shared_plus_cond_image_e4.sh | outputs/ProMoE_TC_B_shared_plus_cond_image/004_ProMoE_B_shared_plus_cond_image_e4/ |
