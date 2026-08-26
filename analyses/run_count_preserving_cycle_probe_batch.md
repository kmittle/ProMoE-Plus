# Count-Preserving Cycle Probe

这个 probe 只回答一个问题：在完全保持每个 routed expert 的 token 数量时，Base ProMoE 的一阶 denoising utility 能否选出真正降低 exact denoising MSE 的联合路由变更。

它不训练模型，不生成图片，也不能单独证明 FID 会提高。

## 候选和对照

每个 image/block/sigma cell 固定生成五组候选，每组 64 个。候选只依赖 native route IDs 和提交前锁定的随机种子，不能查看 VJP、exact loss 或 teacher feature。

- `four_cycle`：两个 native expert 不同的 token 交换 expert。这是 token-expert 二部图中的 4-cycle，只改变两个 token。
- `six_cycle`：三个 native expert 两两不同的 token 做一次有向轮换。这是二部图中的 6-cycle，只改变三个 token。
- `mixed_cycle`：32 个独立 4-cycle 和 32 个独立 6-cycle。
- `single_token`：从独立 cycle component 中取一条边，作为不保持 expert count 的 EPO 类对照。
- `random_joint`：从完整合法的八-token joint signature 空间无放回抽样。每个 source expert 最多提供四个 token，再对 native expert multiset 做无固定点排列，因此八个 token 全部改变且完整 expert-count vector 不变。该 arm 是更一般的联合扰动对照，不固定为若干 pair swap。

所有 count-preserving 候选都逐项验证完整 12-expert count vector。每个候选保留各 token 的 native top-1 route weight，只改变 expert identity，因此激活 token 数、expert histogram 和 routed-expert FLOPs 不变。

一阶标签来自目标 MoE block 输出处的 suffix gradient。候选的所有 token-expert changes 一起求和，再与一次 paired forced-route forward 得到的 exact MSE change 比较。不能把 token 当独立统计样本。Exact forward 固定 batch size 为 8，使所有 368 个候选以四对一组整除执行，并避免 batch 24 相对 batch 1 的数值漂移超过预注册安全门限。

## 锁定顺序

协议固定 Base seed-0 step-200K EMA、sigmas `0.2/0.5/0.8` 和以下三阶段：

1. `plumbing`：8 张历史已观察图，只检查 route override、数值误差、显存和恢复逻辑。输出不会包含 efficacy 聚合。
2. `discovery`：24 张全新图，只测 zero-based block 5。通过预注册 gate 才能解锁确认集。
3. `confirmatory`：另外 48 张全新图，测试 zero-based blocks `1/5/11`。统计先在每张图内平均，再对 48 张图做 200,000 次 paired image bootstrap。

确认 gate 同时要求 selected exact gain、pair concordance、selected-positive rate、相对 random-joint 的 per-flip gain、相对 single-token 的保留比例和 block/sigma 分层结果通过。三个 block 和三个 sigma 的点估计必须全部为正，而且 block 与 sigma 两组都必须各有至少 `2/3` 的单侧 LCB 大于零，不能把两组的通过数量合并。6-cycle 还必须存在无法由相同三个 token 的任何直接 pair swap 获得的独立收益，并使 mixed arm 显著胜过 four-only。

如果 4/6/mixed 都失败，路线停止。如果只有 4-cycle 通过，则删除 6-cycle 叙事。如果确认集通过，也只允许做小规模 router-fitting prototype；在第二个 checkpoint 或 seed 复现前，不允许启动 ImageNet 长训。

## 准备协议

代码必须先经过检查、commit 并 push，且工作树干净。runner 会验证 `HEAD == origin/repa`，否则拒绝写 protocol。

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_count_preserving_cycle_probe_batch.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_200000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/count-preserving-cycle-gate-base200k-v3 \
  --prepare-only
```

`protocol.json` 锁定 checkpoint/config/manifest/source/Git/GPU 环境及全部 gate；`protocol.sha256` 绑定其内容。manifest 排除了 11 份历史 probe manifest 中出现过的 110 个类别，并锁定 24+48 张新图的 latent 和 SHA256。

## 运行

长任务必须在当前 attached tmux session 的新窗口中启动。runner 内部把 case round-robin 分配到 `cuda:4,5,6,7`，每个 worker 只加载一次模型。

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_count_preserving_cycle_probe_batch.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_200000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --output-dir /home/dev/promoe-probes/count-preserving-cycle-gate-base200k-v3 \
  --split plumbing
```

只有前一阶段 summary 的 sealed `gate.passed=true` 时，才能依次把 `--split` 改为 `discovery` 和 `confirmatory`。每个 case 和 summary 都有绑定 protocol hash 的 seal；兼容结果可恢复，不兼容或未密封结果会 fail closed。

## 解释限制

Alternating assignment cycle、固定 row/column marginals 和 2x2 move 都是经典结果，不能宣称 cycle 数学本身新颖。若后续方法成立，能讨论的贡献边界是：

`downstream-utility structured preference learning over fixed expert-count MoE routing cycles`

即便 gate 通过，也必须继续与 single-token downstream preference、一般 joint perturbation 和 matched-compute load-balancing 方法做训练及生成对照。
