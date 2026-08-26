# Paired-CFG Route Inversion Probe

这个入口只回答一个问题：同一个 conditional token 保持计算量不变、改走另一个专家后，conditional 去噪效用与 CFG 实际使用的 cond/uncond 差分效用是否给出相反结论。

它不是训练方法，也不直接证明 FID 会改善。输入是带真实目标的 ImageNet VAE latent，因此结果属于 teacher-forced oracle 诊断。

## 指标

- `conditional_exact_mse_change`：替换专家后的 conditional 去噪 MSE 变化，负数更好。
- `guided_exact_mse_change["1.5"]`：使用相同 unconditional 预测、按 CFG 1.5 合成后的 MSE 变化，负数更好。
- `route_inversion_rate`：上述两个变化符号相反的比例。
- `guidance_alignment_change`：`cond - uncond` 与 oracle 修正方向 `target - uncond` 的余弦变化，正数更好。
- `guidance_projection_mse_change`：允许每个样本使用最优 guidance scale 后，差分方向仍无法解释的目标误差变化，负数更好。
- `scale_one_exact_equivalence_max_abs`：CFG 1.0 必须与 conditional 结果一致，是强制数值控制。
- `noop_*`：把 token 强制回原专家时必须为零。

固定 scale 的 guided MSE 只是诊断量。CFG 本身会有意外推，因此必须与 alignment、projection MSE 和最终生成评估一起解释，不能单独包装成方法动机。

## 示例

```bash
python analyses/run_cfg_route_inversion_probe.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_20000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_20000.pth \
  --latent /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz/n01440764/n01440764_10027.latent.npz \
  --label 0 \
  --seed 11 \
  --block-index 3 \
  --sigmas 0.2,0.5,0.8 \
  --guidance-scales 1.0,1.5 \
  --analysis-scale 1.5 \
  --num-token-probes 32 \
  --candidate-mode mixed \
  --exact-batch-size 4 \
  --device cpu \
  --output /home/dev/promoe-probes/base20k-cfg-block3-class000.json
```

`--ckpt` 用于解析原始 YAML 和 checkpoint step；`--weights-ckpt` 可以指向本地 NVMe 副本，避免重复读取 NAS。默认使用 CPU，不会隐式占用训练 GPU。
