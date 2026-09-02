# DINO 语义邻域与专家真实作用诊断

这个脚本只回答一个问题：两块在 DINO 看来相似的图像区域，是否也需要相似的
ProMoE 专家。它不会训练模型，也不能用来代替 FID 评估。

## 输入是什么

输入是 `run_timestep_utility_probe.py` 已经生成的逐图 JSON。每个 JSON 必须记录相同
checkpoint、相同 block 和噪声位置下，固定 token 分别换成全部 routed expert 后的真实
去噪误差变化。

脚本会按 JSON 里的随机数种子重新取得同一个 VAE latent 样本，再按 `sample.py` 的正式
方式解码：先把供扩散模型使用的 latent 除以 `0.18215`，再交给 VAE。随后用冻结的
DINOv2-B/14 提取 `16 x 16` patch feature，它与 Base ProMoE 的 `16 x 16` token 网格
一一对应。

## 五种比较

1. `dino_correct`：token 使用自己位置上的 DINO patch。
2. `dino_wrong_image`：每张图固定换成另一张图的整张 DINO feature map，但保留位置。
3. `dino_spatial_shift`：仍用本图，却把 feature map 循环平移 5 行、7 列。
4. `router_scores`：不用 DINO，按原 router 的 12 维分数寻找近邻。
5. `random`：用固定随机 feature 寻找近邻。

每个 token 只能从其他图片寻找近邻，不能偷看同一张图。近邻还必须处在同一个 MoE
block 和同一个噪声位置。默认取 8 个近邻，用这些近邻的专家效用排序预测当前 token
的 12 位专家排序。

## 主要指标

- `utility_spearman`：预测的 12 位专家排序和真实排序有多一致。
- `oracle_top1_rate`：预测最好的专家是否真的是事后最好的专家。
- `capacity_additive_gain_relative`：严格保留当前 8 个 token 的逐专家数量，只按预测
  排序交换专家后，单 token 反事实误差变化之和是否改善。它是加性诊断，不是假装成
  真正联合运行后的整模型 MSE。

所有总平均先在每张图片内部聚合，再对图片求平均。置信区间也以图片为重采样单位，
避免把同一张图里的许多 token 当成互相独立的样本。

## 运行方式

应先锁定输入结果、脚本提交和协议，再运行。下面路径使用 Fresh Base v2 300K 的 24 张
探索图片：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_dino_utility_neighborhood.py \
  --result-dir analyses/archvied_analyses/2026-09-02/router-utility-discovery/base-step300k-independent-confirm-v1 \
  --output analyses/archvied_analyses/2026-09-03/dino-utility-neighborhood/exploratory-result.json \
  --dino-path pretrained_ckpt/encoder/dinov2_vitb14/state_dict.pth \
  --vae-path pretrained_ckpt/vae/stabilityai--sd-vae-ft-mse \
  --expected-checkpoint-sha256 60fcc3f80dd354846d8be2505ef1882adc3e035bb06735466d6588cb06bb73e2 \
  --expected-config-sha256 97fe9376303cc390eada34e2bc82fa903b998b78c82d181486630a25187c0ab6 \
  --expected-source-aggregate-sha256 b75bc0ee7d7df5e37263ae4ea62c848081f3dc58cd2b44194b6b27ed69bfc47a \
  --expected-source-results-sha256 782e65c423df42a90e59a3655a36eee3fc4d8162aa0ab2308c28d97712075718 \
  --expected-dino-sha256 8e2abd4e3e90a1d3bbb9b6f0869157d9af08027f984d7e7cbea967ccd9396c69 \
  --expected-vae-config-sha256 44ced7d11f21b60cc66ca65c8be1a44c271a6299aab96914cfdf50240b8389b7 \
  --expected-vae-weights-sha256 2aa1f43011b553a4cba7f37456465cdbd48aab7b54b9348b890e8058ea7683ec \
  --device cuda:0 \
  --expected-cases 24 \
  --k 8
```

## 继续研究的门槛

探索结果必须同时满足：正确 DINO 的平均排序相关至少 `0.10`，且自身置信区间下界
大于 0；相对错图和空间错位对照都至少高 `0.05` 且配对置信区间下界大于 0；至少
16/24 张图同时胜过两个语义
对照；9 个 block/噪声组合中至少 7 个分别胜过每个语义对照；固定专家数量的加性
改善至少 `1e-5` 且置信区间下界大于 0。

脚本还要求所有 JSON 记录相同的模型、配置、canonical checkpoint、实际加载权重、
checkpoint 状态、探针版本和源设备；canonical 与实际加载的权重都必须匹配命令行给出的
SHA-256。DINO 权重、VAE 配置和 VAE 权重也必须匹配命令行预先给出的 SHA-256。
每个 latent 必须匹配源 aggregate 指向的锁定 manifest；源探针版本只接受当前明确支持
的版本 1。24 份逐图结果另有一个整体摘要：按 case ID 排序，把每个 ID、一个零字节、
对应 JSON 的 SHA-256 和换行依次送入 SHA-256。命令行同时锁定 aggregate 和这个整体
摘要，所以修改任何一份专家效用 JSON 都会被拒绝。目前只接受源探针在 CPU 上生成的结果，避免同一个随机数种子跨 CPU/CUDA
重采样出不同 latent。输出路径必须不存在，脚本不会覆盖已经锁定的结果。

任何一项失败，都不能据此启动 DINO teacher-router 训练。全部通过也只允许在一批新图
上确认，仍不等于方法有效。直接用 teacher route 做 KL 监督已有相邻工作，本诊断通过后
还必须另做创新性审查。
