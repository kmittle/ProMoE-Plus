# Routing Translation Stratified Probe

这个入口只回答一个机制问题：translation 后没有跟随内容的路由，是否只是低 margin 的 top-1/top-2 边界抖动。

对每个 noise level 和 shift，先找到 native route 与 transported content route 不同的 token，然后用两种互补方式分组：

- `low_margin` / `high_margin`：按 shifted router 的 top-1/top-2 margin 稳定排序，较低的 `ceil(n/2)` 与较高的 `floor(n/2)` 分开干预；
- `content_top2` / `content_rank3plus`：按 transported expert 在 shifted router 全部专家中的 rank 分开干预。

每组都只改该组 token 的 expert ID，并放入两种错误对应：

- `spatial`：改动位置和替换专家数量完全相同，至少一半专家 ID 与正确内容对应不同。256 个候选都从正确内容图开始，各提出 256 次随机交换，只接受不会破坏改动位置和专家数量的提议；达到一半错配后，还要求交换不能增大四邻域 TV，最后选择最接近正确图的一张；
- `random`：旧版的完全随机错误对应，仍然匹配改动位置和替换专家数量。

所有执行保留 shifted input 自己的 router weight，每个 token 仍只计算一个同宽专家。新版主要看 `*_content - *_spatial`；`*_content - *_random` 只用于连接旧结果。空间对照只有在四邻域专家配对分布的总变差不超过 0.10、且不差于旧随机图时才算有效。做不到时，该单元记为不可识别，原始 JSON 中对应的 `mse`、`mse_change` 和 `relative_mse_change` 都写成 `null`，不能退回旧随机对照。

每个单元会先为每个分组的 `random` 和 `spatial` 对照分别固定随机种子，然后才生成任何候选。因此，增加空间搜索步数不会顺带改变旧随机对照；JSON 也会保存这些种子。

为了解这次 256 次有限搜索为什么失败，每个单元还会记录：实际评估及去重后的候选数、达到 50% 错配的候选数、这些候选中最低的四邻域 TV，以及分别因错配不足、TV 超过 0.10、比随机图更差而被淘汰的数量。这些数字只说明本次搜索覆盖了什么，不能证明约束本身无解；只有穷举、精确求解或严格下界才能作出这种判断。它们也不会改变任何接受门槛。

如果 `high_margin` 和 `content_rank3plus` 仍稳定优于各自 spatial control，才可以排除“只是连续路由图比打散路由图更好”这一解释。随后仍要检查 RCL 梯度是否强化这些错误，才能把问题归因到 ProMoE 的自分配原型学习。该探针本身不是训练方法，也不能替代 FID/IS 评估。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python analyses/run_routing_translation_stratified_probe.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_50000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_50000.pth \
  --latent /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz/n01440764/n01440764_10027.latent.npz \
  --label 0 \
  --seed 11 \
  --block-index 3 \
  --sigmas 0.276,0.5,0.724 \
  --shifts 0:2,0:-2,2:0,-2:0 \
  --device cpu \
  --num-threads 8 \
  --output /home/dev/promoe-probes/base50k-routing-translation-stratified/block3-class000.json
```
