| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| Slot 5.4 · GPU 6-7 · 在 12 个 routed experts 的参数正则中加入 shared expert | `repa` | `bash scripts/_run_times/2026_08_05/5.4-B_expert_contra_param_shared.sh` | `outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_shared/` |
| Slot 6.1 · GPU 0-1 · 在参数正则中同时加入 shared 和 unconditional expert | `repa` | `bash scripts/_run_times/2026_08_05/6.1-B_expert_contra_param_shared_uncond.sh` | `outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_shared_uncond/` |
| Slot 6.2 · GPU 2-3 · 将参数正则温度从 0.7 改为 0.07 | `repa` | `bash scripts/_run_times/2026_08_05/6.2-B_expert_contra_param_tau0p07.sh` | `outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_tau0p07/` |
| Slot 6.3 · GPU 4-5 · 将参数正则温度从 0.7 改为 7 | `repa` | `bash scripts/_run_times/2026_08_05/6.3-B_expert_contra_param_tau7.sh` | `outputs/ProMoE_TC_B_expert_contra/004_ProMoE_B_expert_contra_param_tau7/` |
