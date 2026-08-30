# 路由分组是否真的有语义

这项检查回答一个小问题：ProMoE 把两个图像 patch 分给同一位专家时，这两个 patch 是否真的更相似，而不是只在 router 自己的特征空间里看起来相似。

它不是训练方法，也不会改模型。DINOv2 只在训练结束后充当一把独立的尺子：DINO 特征不会送进 ProMoE，不会影响 router，也不会进入损失。因此这不是 REPA 的表征对齐路线。

## 当前数据只能做什么

输入是一个历史 Base ProMoE step 50K checkpoint 抓取的 32 张图片路由，包含 6 个 MoE block 和 3 个噪声位置，共 18 个检查格。这个旧实验缺少现在要求的完整训练封存，所以结果只能决定这个问题值不值得在 Fresh checkpoint 上复验，不能作为论文证据。

图片来自本地 VAE posterior 参数。服务器没有对应的原始 JPEG，因此程序先用同一个 SD-VAE 的 posterior mode 重建干净图片，再提取 DINOv2-B patch 特征。这会损失一部分图像细节，也是结果的明确限制。

## 三项检查

1. 图内最近邻：一张图片里，DINO 认为最相似的两个 patch 是否更常进入同一专家。
2. 跨图最近邻：不同图片里，DINO 认为最相似的两个 patch 是否更常进入同一专家。
3. 分离度：同一专家的 patch 对，其 DINO cosine 是否高于不同专家的 patch 对。

仅看原始同专家比例是不够的。空间上相邻的 patch 本来就相似，某些专家也可能接收大多数 token。因此程序放入两个保持结构的对照：

- 图内对照整体循环平移每张路由图。它保留每位专家的 token 数，也保留路由图本身的形状，只打断路由图与图片内容的位置对应。
- 跨图对照把整张路由图换给另一张图片。它保留每张路由图和总体专家负载，只打断路由图与原图内容的对应。

每个 block 和噪声位置必须同时满足：

- 图内最近邻差值至少 `0.02`；
- 跨图最近邻差值至少 `0.02`；
- DINO 分离度差值至少 `0.01`；
- 三个差值的图片级 bootstrap 95% 下界都大于 0；
- 三个随机对照检验经过全体 54 次检验的 Holm 校正后，单侧 `p` 都不大于 `0.05`。

最终至少 9/18 个检查格通过，并且覆盖至少 4/6 个 block 和 2/3 个噪声位置。图内检查和分离度按图片重采样。跨图检查会同时重采样查询图片和候选图库，再重新选择最近邻，避免把共享同一个图库的 32 个分数误当成互相独立。门槛、4999 次随机对照、10000 次图片级 bootstrap，以及 4096 个固定 patch 对都在查看结果前写入 manifest。

manifest 还逐项锁定了 32 个 latent、DINO 和 VAE 权重、DINOv2 源码 commit 及源码树内容、CPU、batch size 和软件版本。程序强制关闭 xFormers，让 DINO 始终走同一条普通 PyTorch attention 路径；随后先检查源码树哈希，再从不含旧字节码的临时副本加载 DINO。任何一项变化都会直接停止，而不是悄悄生成另一套结果。

## 运行

程序拒绝未提交改动、`skip-worktree`/`assume-unchanged` 文件，也会直接查询真实远端，拒绝尚未 push 的提交。先完成代码审查、commit 和 push，再等当前 CPU OpenAI 评估结束，避免两个任务争抢 CPU。

```bash
cd /home/dev/promoe-worktrees/long-horizon-routing

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_routing_semantic_audit.py \
  --route-ids /mnt/cubefs/caoboyuan/ProMoE-Plus/analyses/archvied_analyses/2026-08-28/fresh-base-vs-proto-t-50k-routing/route_ids.npz \
  --capture-summary /mnt/cubefs/caoboyuan/ProMoE-Plus/analyses/archvied_analyses/2026-08-28/fresh-base-vs-proto-t-50k-routing/summary.json \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --dino-path /mnt/cubefs/caoboyuan/ProMoE-Plus/pretrained_ckpt/encoder/dinov2_vitb14/state_dict.pth \
  --dino-source-path /home/dev/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
  --vae-path /mnt/cubefs/caoboyuan/ProMoE-Plus/pretrained_ckpt/vae/stabilityai--sd-vae-ft-mse \
  --output-dir /mnt/cubefs/caoboyuan/ProMoE-Plus/analyses/archvied_analyses/2026-08-30/dirty-route-semantic-audit-v2 \
  --device cpu \
  --batch-size 4
```

默认输出：

- `dino_features.npz`：与潜变量、DINO、VAE、DINO 源码、软件环境和 manifest 绑定的特征缓存；它还保存特征内容哈希和首次生成环境，复用缓存时不会把当前环境误写成生成环境；
- `summary.json`：完整输入身份、图片级汇总统计、逐格结果和最终门槛；
- `summary.md`：不需要读 JSON 的中文结论。

## 怎样决定下一步

若未通过，就停止“现有路由已经形成独立语义分组”这一说法，不为它安排正式训练。

若通过，只允许在正在从零训练的 Fresh Base checkpoint 上做一次预注册确认。它仍然不能说明语义联系由 RCL 造成，也不能说明 router 选中的专家更擅长去噪。前者需要同设置的无 RCL 从零对照；后者由独立的 exact utility 审计回答。只有这三条证据能连成一个清楚的问题后，才值得设计训练方法。
