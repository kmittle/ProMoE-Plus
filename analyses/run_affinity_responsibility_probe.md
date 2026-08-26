# Affinity-Responsibility Probe

这个入口检查 ProMoE 的 prototype affinity 是否真的适合作为 routed expert 的输出强度。

Base ProMoE 使用 `top_k=1`，同一个 cosine score 同时决定两件事：选哪个 routed expert，以及该 expert 输出乘多大。前者是 dispatch，后者实际控制 shared expert 与 routed expert 的责任比例。两者没有必然相同的最优解。

## 干预方式

诊断固定以下内容不变：

- top-1 expert ID；
- expert 输入与 expert 参数；
- 被执行的 expert 数量和 FLOPs；
- shared expert 输出；
- 其余 block 和网络参数。

它只把目标 MoE block 的 routed output scale 替换为一组绝对值。默认测试 `0,0.25,0.5,0.75,1.0`，并始终以 checkpoint 原生 cosine affinity 为 baseline。

单 token 干预用于定位局部责任错配；整层固定 scale 干预用于判断一个简单、可部署的统一校准是否已经优于原生 affinity。输入包含真实 latent 和 flow-matching target，因此结论属于 teacher-forced 因果诊断，不直接等价于 FID 改善。

## 关键指标

- `native_best_rate`：原生 affinity 不劣于所有候选 scale 的 token 比例。
- `candidate_oracle_better_rate`：至少一个固定 scale 能降低精确去噪 MSE 的比例。
- `affinity_best_candidate_scale_spearman`：原生 affinity 与最佳候选责任强度的秩相关。如果 affinity 同时是合理的责任估计，该相关性应为正且稳定。
- `responsibility_slope`：在原生 scale 处，增加 routed expert 权重对去噪损失的一阶导数。
- `global_summary`：保持整层 dispatch 不变、把所有 conditional token 设为同一 scale 的精确结果。
- `noop_*`：把权重写回原值时必须为零。

该问题与替换 expert ID 的 counterfactual routing 不同，也与 Top-k 集合内部重新分配固定总 mass 不同。这里研究的是 top-1 routed path 的总责任强度。

## 示例

```bash
python analyses/run_affinity_responsibility_probe.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_20000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_20000.pth \
  --latent /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz/n01440764/n01440764_10027.latent.npz \
  --label 0 \
  --seed 11 \
  --block-index 3 \
  --sigmas 0.2,0.5,0.8 \
  --candidate-scales 0,0.25,0.5,0.75,1.0 \
  --num-token-probes 32 \
  --exact-batch-size 4 \
  --device cpu \
  --output /home/dev/promoe-probes/base20k-responsibility/block3-class000.json
```

`--ckpt` 用于解析原始 YAML 和 checkpoint step；`--weights-ckpt` 可指向本地 NVMe 副本，避免反复读取 NAS。默认使用 CPU，不会占用训练 GPU。
