# 有限步专家分配批量确认

这个入口把单样本检查分成三道门：

1. `plumbing` 只检查程序、哈希和数值对照，不公布汇总效果。
2. `discovery` 使用事先固定的 8 张图。只有全部门槛通过，程序才允许打开确认集。
3. `confirmatory` 使用另外 24 张图。它不能在 discovery 失败时强行运行。

三组图直接复用 Fresh Base 路由审计在看到 Fresh 结果前固定的 manifest。新协议再次记录该 manifest、36 个 latent、Fresh 300K checkpoint、配置、代码和 Git 提交的 SHA256。分析代码放在独立工作树，但 Fresh 训练输出仍从 Git 主工作树的 `outputs/` 读取；协议会同时记住这两个工作树及运行目录的真实路径，防止把同名空目录误当成训练结果。它还会核对训练日志中的 run_id、从 step 0 开始的记录、300K 保存记录、训练时实际读取的数据清单，以及 GPU UUID 和 Torch/CUDA 环境。每次分析还必须使用同一 NVIDIA driver 和 cuDNN，换版本后不能把新结果混进旧协议。

需要如实说明一个边界：已经启动的 Fresh 训练使用 version-1 训练记录，它保存了 Python、Torch、CUDA runtime 和 GPU UUID，但当时没有保存 NVIDIA driver 与 cuDNN。程序会把这两个缺项明确写进协议，不能事后假装训练开始时已经记录过。Fresh 300K 权重仍由完整 SHA256 固定；driver 与 cuDNN 的强制一致只适用于之后运行的分析。

## 为什么门槛较严格

这条方法与已有的 rollout credit、MoE 路由强化学习和严格均衡指派都很接近。只有下面四种证据同时存在，才值得继续：

- 即时和 8 步排序的相关性上置信界仍低；
- “现在有益、未来有害”或反方向的交换占比有稳定下界；
- 按即时效果挑出的最佳交换，在 8 步以后有稳定 regret；
- 现象至少覆盖 4 个专家层和 2 个噪声位置，而不是单点异常。

任一条件不满足，就停止这条训练方法。置换检验只回答候选标签关联是否超过随机，不替代按图像聚类的置信区间，也不计入 `passed`。

## 执行顺序

必须在独立分析分支已经 `$check 1`、提交并推送之后执行。Fresh 训练在 300K 门禁暂停、GPU 0-3 空闲后，先生成协议：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_finite_horizon_routing_probe_batch.py prepare \
  --ckpt /path/to/fresh/checkpoints/ckpt_step_300000.pth \
  --latent-root /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

程序会打印并保存 `output_dir`。后续三步都使用这个目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_finite_horizon_routing_probe_batch.py run-split \
  --output-dir <output_dir> --split plumbing

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_finite_horizon_routing_probe_batch.py run-split \
  --output-dir <output_dir> --split discovery

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_finite_horizon_routing_probe_batch.py run-split \
  --output-dir <output_dir> --split confirmatory
```

协议先在带随机名字的临时目录中完整写好，再在锁内一次发布；两个 `prepare` 进程不能互相覆盖，也不会共用一个 `.tmp` 文件。每张图的结果单独原子写入，并带有内容哈希，进程中断后可以继续。checkpoint 的训练状态、大小和哈希都从同一个打开文件读取，并与训练日志中的 300K 保存记录逐项核对；每个 latent 也从同一个打开文件读取并在前后各验一次哈希。读取期间替换或改写文件会直接失败。已有结果只有在 case 信息、权重身份、设备、完整协议和内容哈希都一致时才会复用。进入下一组以前，程序会从上一组每张图的原始结果重新计算汇总，而不是只相信一个可以手改的 `passed: true`。同一时间只允许一个进程占用 GPU 0-3 和这个结果目录；如果协调进程被强制杀死，Linux 会同时杀死它启动的 GPU worker，避免旧 worker 与新任务重叠。H1 对照复用采样器的 float32 算术；`1e-5` 相对误差门槛只容纳候选与原生 MSE 相减产生的舍入误差，错误步长或错误符号仍会失败。

## 结果边界

即使确认集通过，也只说明“即时专家分配效果和有限步效果存在系统性错位”。它不能直接批准 500K 训练，更不能声称已有方法空白。下一步仍需单独完成训练方法的创新性检查、正确依据与打乱依据对照、`$check 1`、提交和推送。
