# Routing Translation Stratified Probe

这个入口只回答一个机制问题：translation 后没有跟随内容的路由，是否只是低 margin 的 top-1/top-2 边界抖动。

对每个 noise level 和 shift，先找到 native route 与 transported content route 不同的 token，然后用两种互补方式分组：

- `low_margin` / `high_margin`：按 shifted router 的 top-1/top-2 margin 稳定排序，较低的 `ceil(n/2)` 与较高的 `floor(n/2)` 分开干预；
- `content_top2` / `content_rank3plus`：按 transported expert 在 shifted router 全部专家中的 rank 分开干预。

每组都只改该组 token 的 expert ID，并与完全相同 changed-token support 和 replacement-expert histogram 的 random route 比较。所有执行保留 shifted input 自己的 router weight，每个 token 仍只计算一个同宽专家。因此 `*_content - *_random` 是固定计算量的主要因果量。

如果 `high_margin` 和 `content_rank3plus` 仍稳定优于各自 random control，就不能把问题解释成普通边界平滑；这更接近 prototype score 与 expert utility 的非局部错配。该探针本身不是训练方法，也不能替代 FID/IS 评估。

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
