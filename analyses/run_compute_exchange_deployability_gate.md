# 同专家计算交换的可部署性门槛

`run_compute_exchange_deployability_gate.py` 检验一个比 target-gradient oracle 更严格的问题：只看生成时已经存在的 MoE 输入和路由状态，能否判断同一 expert 内哪些 token 应该放弃一次 routed FFN pass，哪些 token 应该递归执行第二次 pass。

它不修改 Base checkpoint，不训练 diffusion backbone，也不使用 REPA、DINO 或通用表征对齐。校准结束后，scorer 的 forward 不接收 target、noise、gradient 或 routed-expert output。

## 方法和同算力约束

scorer 有两个独立 head：

```text
donor head:   预测删除 w E_e(h) 的代价
receiver head: 预测增加 w E_e(h + w E_e(h)) 的价值
```

每个 `(image, expert)` 独立求解三态分配。原来有 `n` 个 routed pass 时，固定选择 `k` 个 0-pass token、`k` 个 2-pass token，其余为 1-pass：

```text
0*k + 1*(n-2k) + 2*k = n
```

因此每个 expert 的逻辑 FFN pass 数逐项不变。route ID、route weight、shared expert 和 unconditional expert 也保持不变。这个等式不代表 scorer、排序和额外 kernel 没有开销；进入正式模型后仍要单独报告总 FLOPs 和 latency，并加入 Base + dummy scorer/solver 对照。

## 数据隔离

- `calibration` 只使用原 compute-exchange discovery 的 24 张图。目标梯度只在这里生成 donor/receiver 一阶监督。
- 18 张图用于拟合，6 张图用于固定 early stopping。划分由 case ID 的哈希确定。
- `retrospective` 使用原 confirmatory 的 48 张图，但只提取 forward-only 特征。公开协议不保存 confirmatory source-result path；feature payload 和 metadata 中出现 donor/receiver target、source-result path 或 target-derived native MSE 都会直接报错。
- primary、没有 hidden state 的 router-context 对照、同 expert 内打乱监督对应的负对照采用相同优化流程。
- 旧 confirmatory 的聚合结果已经公开，所以这一阶段只能筛选 scorer，不能称为新的盲确认。通过后仍需在未见类别和第二 checkpoint 或 seed 上重新封存 exact gate。

## 执行顺序

入口要求代码已经 commit 并 push、工作树干净、`origin/repa...HEAD` 为 `0 0`。先锁定协议：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_compute_exchange_deployability_gate.py \
  --prepare-only
```

随后必须按顺序执行：

```bash
/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_compute_exchange_deployability_gate.py \
  --extract-split calibration

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_compute_exchange_deployability_gate.py \
  --fit --fit-device cuda:4

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_compute_exchange_deployability_gate.py \
  --extract-split retrospective

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_compute_exchange_deployability_gate.py \
  --select --fit-device cuda:4

/home/dev/miniforge3/envs/promoe/bin/python \
  analyses/run_compute_exchange_deployability_gate.py \
  --evaluate --fit-device cuda:4
```

长任务应在当前 tmux session 的新 window 中运行。特征提取和 exact reveal 固定使用 GPU 4-7，每张图都写独立结果和 seal，可按已封存 case 恢复。`--fit-device` 被锁定为 `cuda:4`，不能改到正在运行 Base 流水线的 GPU 0-3。

`--select` 的 action-generation 路径只接收已封存的 forward-only feature 和 scorer；公共协议不携带 confirmatory source-result path。它生成每个 `(image, block, sigma)` 的 exact 0/1/2 action 后，立即写入 `retrospective-actions.json` 并封存。`--evaluate` 是下一次独立调用：它先复核 action seal，再重跑并封存这些具体 action 的真实 suffix counterfactual，最后才解析旧 source result，计算 exact gain 和候选排序诊断。source 与新 reveal 的 native MSE 不一致也会触发 safety failure。旧 64-candidate bank 不再替代 learned exact action 的收益。

## 停止条件

fit gate 先要求 primary 在 held-out calibration 上恢复候选排序，并明显优于 router-context 和打乱对应。失败时禁止提取 retrospective 特征。

retrospective gate 要求图像级 learned-action 精确收益、95% 单侧下界、正收益图数、候选排序一致性、旧候选库 oracle 参考比例、三个 block、三个 sigma、五个对照、数值安全检查和 Holm 校正同时通过。任何核心项失败都会在封存结果后以非零状态退出并停止该 scorer 设计；不能放宽阈值后重跑，也不能因此启动 501K 长训。
