# Fresh Base 路由审计

这个审计回答一个很具体的问题：ProMoE 的 router 选中的专家，是否真的更适合当前 token 的去噪任务。

它不训练模型，也不生成图片。即使审计通过，也只能说明“问题确实存在”，不能说明某个新方法有效，更不能代替 500K 训练和 50K 图片评估。

## 为什么要重新做

旧权重上曾经看到过两个现象：router 分数与真实去噪收益几乎不相关；在保持每个专家 token 数不变时，重新分配 token 仍能降低误差。那些实验已经归档为 dirty 结果，只能提供线索。

旧的 `phase_metric_base_s0` 虽然确实从 step 0 开始，但它启动时还没有自动保存源码和 checkpoint 哈希，所以不能作为这次严格审计的正式证据。这里改用独立的 `fresh_routing_audit_s0`：它必须在干净且已经 push 的提交上重新从 step 0 训练，并同时检查 50K、100K、150K、200K 四个 checkpoint。四个 checkpoint 使用完全相同的图像、噪声、block 和 sigma，因此可以判断现象是否只是某个训练时刻的偶然结果。

## 公平性

- 只检查 zero-based block 1、5、11 和 sigma 0.2、0.5、0.8。
- 替换专家时保留 token 原来的 route weight，避免把“选谁”和“乘多少”混在一起。
- `native_capacity_oracle` 必须逐专家保持原 token 数不变，所以负载和 routed-expert FLOPs 不变。
- discovery 使用 8 张图，confirmatory 使用另外 24 张图。
- manifest 排除了在锁定时已经出现在项目历史 probe 结果中的全部类别（共 394 类，包含 smoke 用过的类别 1）。Fresh case 只能从这个冻结集合之外选择。
- 统计门槛沿用旧 probe 在结果揭晓前锁定的门槛，不根据 Fresh Base 结果重新调节。

## 运行顺序

所有审计输出必须写入项目内的 `analyses/archvied_analyses/`。训练输出命令行路径仍写成 `outputs/...`；本服务器允许单个实验目录通过登记的软链接落到 `/home/dev/promoe-runs/`，审计会同时记录这两个路径并拒绝未登记的目标。即使整个 `outputs/` 被替换成软链接，它的目标也不会自动获得信任。manifest 只能使用仓库中预注册的 canonical 文件，不能用命令行替换。下面命令依次执行，前一阶段失败时程序会拒绝后续阶段。每个阶段都会再次检查 Git 提交、Python/Torch/CUDA 版本和物理 GPU；因此四个命令必须在同一台机器、同一个干净且已推送的提交上执行。

进入 discovery 或 confirmatory 前，程序不会只相信上一阶段的 summary。它会重新读取该阶段每个 checkpoint、每个 case 的结果，再算一次 summary；重新计算的内容必须与已经发布的 summary 完全相同。这样，即使 summary 和它的校验文件被一起改写，也不能凭空把失败改成通过。

每个 `.seal.json` 只用于发现文件写到一半、文件错配或普通误改，它不是带私钥的数字签名。拥有归档目录写权限的人仍可同时改写结果和校验文件。因此，这套审计依靠逐 case 重算、重新检查 checkpoint、配置、数据和 Git 状态来形成证据，不能把 seal 单独描述成防篡改证明。

```bash
# 必须使用训练时实际调用的同一个 Python。直接运行仓库 wrapper 的标准
# 实验服务器使用第一行；只有训练启动命令也把解释器替换到当前服务器时，
# 才使用第二行。两条路径不能混用。
PYTHON=/mnt/workspace/yujie/.conda/envs/promoe/bin/python
# PYTHON=/home/dev/miniforge3/envs/promoe/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_DIR=outputs/ProMoE_TC_B/004_ProMoE_B_fresh_routing_audit_s0
LATENT_ROOT=/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz
OUTPUT_DIR=analyses/archvied_analyses/2026-08-29/fresh_base_routing_audit_v1

$PYTHON analyses/run_fresh_base_routing_audit.py \
  --run-dir "$RUN_DIR" \
  --latent-root "$LATENT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --stage prepare

$PYTHON analyses/run_fresh_base_routing_audit.py \
  --run-dir "$RUN_DIR" \
  --latent-root "$LATENT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --stage plumbing

$PYTHON analyses/run_fresh_base_routing_audit.py \
  --run-dir "$RUN_DIR" \
  --latent-root "$LATENT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --stage discovery

$PYTHON analyses/run_fresh_base_routing_audit.py \
  --run-dir "$RUN_DIR" \
  --latent-root "$LATENT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --stage confirmatory
```

## 如何判断

prepare 会检查四个 checkpoint 都带有完整的 trainer state：版本、seed、4 卡 world size、sampler 位置、rank RNG state、LatentFolder 顺序哈希和同一个唯一 `run_id` 必须有效，四个点还必须共享同一个 sampler contract。

训练启动时还会记录当时的 Git 提交、工作区是否干净、关键源码哈希、实际加载配置的哈希、Python/PyTorch/CUDA 版本和四张 GPU 的 UUID。程序会直接查询预先写死的 GitHub `repa` 分支，不能只拿本地 `origin/repa` 缓存冒充“已经 push”。第一次写训练日志前，整个实验输出目录也必须为空；只有 `checkpoints/` 为空还不够，因为旧图片会被采样程序跳过并混进 FID。这份记录必须同时出现在日志和每个 checkpoint 中，而且与运行审计时的环境完全一致。只在训练结束后查看“现在的源码”不算证据。

日志必须有 `fresh=True` 的唯一 run marker，并在 marker 之后按 50K、100K、150K、200K 顺序记录每个 checkpoint 的文件大小、SHA256 和训练启动记录的哈希；marker 之前的旧日志内容不会被当作 Fresh 证据。这避免把别的权重拼到一份 fresh 日志上。

每个 discovery/confirmatory checkpoint 的“路由准确度”门槛是：平均 native regret 至少 `5e-5`，native route 恰好是 oracle 的比例不超过 `0.15`，router 与真实 utility 的 Spearman 不超过 `0.10`，保持专家计数不变的平均改善至少 `1e-5` 且 bootstrap 95% CI 下界大于 0；至少 5 张（confirmatory 至少 16 张）图改善，并且每个 block、每个 sigma 都改善。所有 no-op、forced-native 和联合 native 控制的数值误差必须严格为零，native-capacity oracle 的专家计数必须完全相同。

相位结构另有四个门槛：平均“router 排序稳定度减去 utility 排序稳定度”至少 `0.10` 且 CI 下界大于零，utility 成对翻转率至少 `0.30`，oracle 相对 native 的翻转优势至少 `0.10`。

主 checkpoint 是 200K。只有 200K 通过，而且 50K、100K、150K 中至少两个也通过，才认为“路由准确度问题”在这一条从零训练轨迹上稳定存在；相位结构也要满足同样的二过三规则。

相位结构单独判断：只有 200K 通过，而且更早三个 checkpoint 中至少两个也通过，才认为专家的真实能力排序会随噪声阶段系统变化。

通过以后仍不能马上开 500K 新方法训练。下一步应先设计一个不偷看 target、推理时不增加 FLOPs 的可部署 router，并在第二个随机种子上复验问题。
