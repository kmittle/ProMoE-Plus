# Routing Flip Probe

这个入口检查 ProMoE 的 top-1 expert assignment 是否随水平翻转后的内容一起移动。训练时 LatentFolder 以相同概率读取 latent 和 latent_flip，所以水平翻转不是额外引入的测试分布。

诊断会对 sampled clean latent 和 paired noise 做完全相同的水平翻转，并只在一个 MoE block 强制 expert ID。所有模式保留翻转输入自己的 router weight，每个 token 仍只执行一个同宽专家：

- native：翻转输入自己的路由；
- noop_native：强制回 native expert 的严格数值控制；
- content_follow：把原图 route map 水平翻转后用于翻转输入；
- position_follow：原图 route map 留在原绝对位置；
- random_matched：与 content_follow 具有完全相同的改动 token support 和 replacement-expert histogram。

主因果量是 content_follow - random_matched 的相对 MSE 差。只有它在独立图像、noise regions、blocks 和成熟 checkpoints 上稳定为负，才能说明 transported content route 优于任意等规模换专家。

route_margin 同时报告发生变化的 token 上：

- native top-1 score 比 transported content expert 高多少；
- transported expert 在全部 experts 中的平均 rank；
- changed 与 unchanged token 的 top-1/top-2 margin。

这些量用于区分低 margin 决策边界抖动与高置信路由错误。它们本身不能证明生成指标会改善。

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    python analyses/run_routing_flip_probe.py \
      --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_50000.pth \
      --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_50000.pth \
      --latent /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz/n01440764/n01440764_10027.latent.npz \
      --label 0 \
      --seed 11 \
      --block-index 3 \
      --sigmas 0.276,0.5,0.724 \
      --device cpu \
      --num-threads 8 \
      --output /home/dev/promoe-probes/base50k-routing-flip/block3-class000.json

默认使用 CPU。--ckpt 只负责定位原 YAML 和 canonical step；--weights-ckpt 可指向本地 checkpoint 副本，避免反复读取 NAS。
