# Base/Loss-Free Cross-Checkpoint Credit Gate

这个分析回答一个限定很窄的问题：当每个 routed block 的 token 数量已经明显更均衡时，不同 expert 每个 token 收到的去噪梯度是否仍然不均衡。

它只分析冻结的 Base 与 Loss-Free step-200K EMA checkpoint，不训练模型，也不生成图片。Loss-Free 是诊断对照，不是本文的新方法；通过本 gate 也不能说明 FID 会提高。

## 为什么需要新 probe

Base probe 把无偏原型相似度的 argmax 当作原生路由。Loss-Free 会在“选择 expert”时加入不参与梯度的 `expert_bias`，但仍用无偏相似度作为 route weight。因此，无偏 argmax 与实际 route 不同是预期行为，不能记为错误。

新 probe 分别检查：

- route ID 等于“无偏相似度 + Loss-Free bias”的 argmax；
- route weight 等于所选 expert 的无偏相似度；
- 同一冻结输入重复计算得到完全相同的 route；
- 无偏 argmax 分歧只作为 Loss-Free 实际重路由量报告。

## 两种信用

设 token `t` 的 routed-expert 输入是 `x_t`，MoE 输出的 suffix gradient 是 `g_t`，原生 route weight 是 `w_t`。输出侧信用为：

```text
v_t = w_t * g_t
c_t = ||v_t||^2
```

对项目中的两层 `MoeMLP`，参数侧信用不做随机近似，而是精确计算逐 token 经验 Fisher trace：

```text
z_t = W_up x_t + b_up
a_t = GELU_tanh(z_t)
u_t = (W_down^T v_t) * GELU_tanh'(z_t)
p_t = ||v_t||^2 (||a_t||^2 + 1) + ||u_t||^2 (||x_t||^2 + 1)
```

两个 `+1` 分别是 `b_down` 和 `b_up` 的梯度贡献。runner 在读取 checkpoint 前用确定性 toy case 将闭式结果与逐 token autograd 对照，最大相对误差必须不超过 `1e-5`。

参数侧主指标是每个 image/block/sigma cell 内，active experts 的 `sum(c_t)` 与 `sum(p_t)` Spearman；先对同一图的 9 个 cell 求均值，再以 image 为单位 bootstrap。Base 和 Loss-Free 都必须满足 mean `>= 0.50`、单侧 95% LCB `>= 0.30`，每个 cell 至少有 3 个 active experts。

## 负载统计

全局负载不能用“每张图的 CV/Gini 再平均”判断。runner 对 block `1/5/11` 分别累加所有图和 sigma 的完整 12-expert count vector，再各自计算：

- CV `<= 0.20`；
- Gini `<= 0.12`；
- max/min `<= 2.0`；
- 相比同 case 的 Base，CV 和 Gini 每个 block 都至少下降 50%；
- 所有 expert 都被激活。

不同 block 的 expert ID 没有共同语义，任何统计都禁止跨 block 相加后再计算不均衡。

## 封存依据与顺序

有效预注册由两个不可变文件共同定义：

```text
v1 SHA256 59ce95f39220511c510b589b78e69b0139c961aaa1d3e4e3f013c16312565a43
v2 SHA256 04ced5b1cebf371153c33c4f7b9cf703b58d430ee504d8d52c083a186f254b57
```

case manifest 复用 Base protocol：

```text
SHA256 9c25bd0144228e921be1a5491dafa32299356f5af00e0a5cc15d857a1eeef096
```

runner 会先逐字段核对两个预注册文件中的 seed、batch、阈值和模型设置，重放训练端 cache-first 的 latent 枚举与哈希规则，同时要求训练采用的缓存序列与实际磁盘完整排序清单完全一致，再把所得 dataset identity 与 checkpoint 中的 sampler provenance 逐字段比较。随后才把两个预注册 hash、Base protocol hash、Base/Loss-Free checkpoint hash、Loss-Free trainer/sampler/RNG provenance、配置、latent manifest、项目源文件 hash、Git commit、运行环境以及本次给定的全部输入位置写入新 protocol。worker 在实际加载 checkpoint 前后及读取每个 latent 前后都会复核 hash，阶段汇总后还会再次复核全部输入。每个 case 与 summary 都有独立 seal，内容被修改或只剩单边文件时会直接失败。

阶段必须按以下顺序运行：

1. `plumbing`：8 张图，只发布数值安全字段，隐藏 MSE 和信用结果。
2. `discovery`：32 张图，只判断 Loss-Free 的逐 block 全局 count gate，暂不汇总信用效果。
3. `parameter`：对 discovery 顺序中的前 16 张图，在 Base 和 Loss-Free 上验证精确参数信用；通过后才汇总 discovery 的 count-adjusted output credit。
4. `confirmatory`：64 张类不重叠的新图；只有此前所有 gate 通过才解锁，先过 count gate，再查看信用结果。

任何门槛失败都停止，不能改阈值或跳过阶段。step-10K 等中间 checkpoint 只能放在另一个明确标为 trajectory diagnostic 的协议中，不能替代这里固定的 step-200K 结论。

## 输入按内容认，不按位置认

所有被锁定的输入都没有默认路径，必须用命令行参数显式给出。runner 只按内容认它们：文件可以挪到任何位置，但字节必须和封存时完全一致。

| 参数 | 内容 | 怎样确认是封存时那一份 |
| --- | --- | --- |
| `--preregistration-v1` / `--preregistration-v2` | 两份 Loss-Free 预注册 JSON | SHA256 等于上面钉住的 v1 / v2 值 |
| `--base-protocol` | Base step-200K credit-balance gate 的 `protocol.json` | 规范化 JSON 的 SHA256 等于 `9c25bd01…`，同目录的 `.sha256` 文件一致 |
| `--base-preregistration` | Base gate 的预注册 JSON | SHA256 等于 Base protocol 中 `preregister.sha256` 的记录 |
| `--base-results-dir` | Base gate 的逐 case 结果目录 | 每个结果都必须通过自己的 seal，并属于上述 Base protocol |
| `--base-weights-ckpt`、`--base-config` | Base seed-0 step-200000 checkpoint 及其训练配置 | checkpoint 大小与 SHA256 钉在代码中；配置 SHA256 等于 Base protocol 的记录 |
| `--lossfree-ckpt` | Loss-Free step-200000 checkpoint | checkpoint 内的训练记录（seed、world size、step、sampler 位置、数据集指纹）必须与预注册的训练设置吻合，各 rank 的 RNG 状态必须完整；路径最后三级必须仍是 `004_ProMoE_B_lossfree_u1e2_credit_control_s0_200k/checkpoints/ckpt_step_200000.pth` |
| `--lossfree-config` | Loss-Free 训练配置 | SHA256 等于预注册中的 `config_sha256`（`ce7ce84a…`） |
| `--latent-root` | ImageNet latent 目录，默认 `/home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz` | 重算的有序数据集指纹等于 checkpoint 的 sampler 记录，每个 case 的 latent SHA256 等于 manifest |

因此它只接受封存时那一对 checkpoint；换成别的 checkpoint（例如另一个 `bias_update_rate` 的 Loss-Free 训练）会被拒绝。这是封存 gate 的设计，不是路径问题。

Loss-Free checkpoint 还要保留最后三级路径，是因为预注册时它还没有训练出来，当时只能记下“那次训练自己输出目录里的 step-200000 文件”。换存放的根目录没有关系，改目录名或文件名不行。

`--output-dir` 同样没有默认值，而且必须是**仓库内、被 Git 忽略**的目录（例如 `outputs/` 或 `analyses/archvied_analyses/` 下），否则参数解析阶段就会拒绝：结果不允许写到仓库外；目录若不被忽略，写出的 `protocol.json` 会让工作区变脏，下一阶段的 clean-tree 检查必然失败。

## 运行

代码必须先通过检查、commit 并 push，且 `HEAD == origin/repa`。先把锁定输入写成一个参数数组，`/path/to/...` 换成实际位置：

```bash
PY=/home/dev/miniforge3/envs/promoe/bin/python
LOCKED_INPUTS=(
  --preregistration-v1 /path/to/credit-balance-lossfree-s0-200k-v1-preregister.json
  --preregistration-v2 /path/to/credit-balance-lossfree-s0-200k-v2-preregister.json
  --base-protocol /path/to/credit-balance-gate-base200k-v1/protocol.json
  --base-preregistration /path/to/base-credit-balance-preregister.json
  --base-results-dir /path/to/credit-balance-gate-base200k-v1
  --base-weights-ckpt /path/to/base-seed0-ckpt_step_200000.pth
  --base-config configs/004_ProMoE_B_seed0_control.yaml
  --lossfree-ckpt /path/to/004_ProMoE_B_lossfree_u1e2_credit_control_s0_200k/checkpoints/ckpt_step_200000.pth
  --lossfree-config /path/to/004_ProMoE_B_lossfree_u1e2_credit_control_s0_200k.yaml
  --output-dir analyses/archvied_analyses/YYYY-MM-DD/credit-balance-lossfree-s0-200k-v2
)
```

Loss-Free step-200K checkpoint 存在后，先锁 protocol：

```bash
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${LOCKED_INPUTS[@]}" --prepare-only
```

再依次运行：

```bash
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${LOCKED_INPUTS[@]}" --stage plumbing
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${LOCKED_INPUTS[@]}" --stage discovery
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${LOCKED_INPUTS[@]}" --stage parameter
"$PY" analyses/run_learning_credit_balance_cross_checkpoint.py "${LOCKED_INPUTS[@]}" --stage confirmatory
```

每个阶段都要传同一组锁定输入：protocol 记录了这些文件的实际位置，位置变了会与已锁的 protocol 不一致而直接失败。GPU 固定为 4-7（`--devices` 只接受 `cuda:4,cuda:5,cuda:6,cuda:7`）。这是长任务，必须在当前 attached tmux session 的新 window 中运行。

## 当前输入状态（2026-09-13 在实验服务器的仓库副本中核对）

`/home/dev/promoe-probes` 和 `/home/dev/promoe-runs` 这两个仓库外目录已经不存在，保留下来的内容迁到了仓库内的 `outputs/`、`outputs/archived_outputs/` 和 `analyses/archvied_analyses/`。在这些目录中逐项核对：

| 输入 | 状态 |
| --- | --- |
| Base step-200K checkpoint | 在 `analyses/archvied_analyses/2026-08-28/dirty_probes/promoe-probes/base-seed0-ckpt_step_200000.pth`，大小和 SHA256 都与代码钉住的值一致 |
| Loss-Free 训练配置 | 已随实验从 `configs/` 删除（`717d172`）。钉住的字节只在 Git 历史里：`git show b75164f:configs/004_ProMoE_B_lossfree_u1e2_credit_control_s0_200k.yaml`；`0fca45a` 改过其中的 `output_dir`，之后的版本哈希对不上。取回后放在被 Git 忽略的位置，不要放回 `configs/` |
| 两份 Loss-Free 预注册 JSON | 未找到 |
| Base credit-balance gate 的 protocol、预注册与结果目录 | 未找到 |
| Loss-Free step-200K checkpoint | 未找到（`outputs/archived_outputs/2026-08-28/ProMoE_TC_B_lossfree/` 是空目录） |

后三项找回之前，这个 gate 无法重跑，也不能拿其他文件顶替。

## 结论边界

完整通过只支持：在这一对 checkpoint 和锁定输入上，token-count balance 没有消除稳定的 routed-expert credit mismatch，而且 suffix-gradient energy 能预测真实参数侧经验 Fisher。

它不能支持：

- Loss-Free 本身是论文贡献；
- 梯度能量等于 expert 的语义知识或样本价值；
- frozen-checkpoint 相关性证明了因果优化收益；
- 通用梯度归一化足以构成 TPAMI 方法；
- 内部指标改善等于 OpenAI 50K-sample FID 改善。
