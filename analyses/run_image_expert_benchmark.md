# Image expert backend benchmark

Compare `bmm` and grouped-expert `loop` using the complete B model with DDP.
Both start from the same random weights and consume the same preloaded synthetic
latent batches, noise levels, labels, and condition-dropout RNG sequence. Each
timed step includes model forward, MSE and routing-contrastive backward, DDP gradient
communication, gradient clipping, fused AdamW, zero-grad, and the trainer's EMA update.
Data loading, VAE encoding, checkpointing, and logging are excluded. This measures
compute throughput, not real-data quality or complete dataloader throughput.

Use an attached tmux session, a new window, and the repository GPU-claim helper.
Run the following inside that window after claiming four idle GPUs and exporting
their IDs as `CUDA_VISIBLE_DEVICES` (use the configured promoe interpreter):

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  analyses/run_image_expert_benchmark.py \
  --config configs/004_ProMoE_B_shared_plus_cond_image_e4.yaml \
  --global-batch-size 256 --warmup 10 --steps 30 --repeats 2 \
  --output outputs/benchmarks/image_experts/<unique-run>/results.json
```

Two repeats use order `bmm, loop, loop, bmm`; every arm reconstructs the model and
optimizer, warms up separately, and clears allocations afterward. The report records
the slowest-rank wall time, global images/sec, per-rank CUDA event timings, maximum
allocated/reserved GiB, final finite loss/gradient checks, and final expert counts.
It also records source hashes, GPU IDs, precision, and software versions. Output
paths must be new and under `outputs/`. No checkpoint is loaded or saved.

To isolate scaling with expert count, add `--expert-counts 3 6 12 24 --routing balanced
--freeze-weights`. These overrides apply only inside the benchmark; experiment YAMLs
and model routing defaults are unchanged. Balanced dispatch assigns conditional
images round-robin and gathers each assigned prototype's cosine score, retaining
the pooling/router/contrastive gradient paths. Every expert is exercised when there
are at least as many conditional images as experts. Frozen weights use AdamW at
learning rate zero while still running its state updates, plus the usual backward,
clipping, DDP communication, and EMA. This measures execution with fixed parameters
and equal dispatch, rather than allowing BF16 numerical differences to accumulate
into different learned routes. It is a controlled performance test, not a quality
experiment. JSON results include all per-rank measurements; a companion summary CSV
reports averaged per-step time, throughput, peak allocated memory, and bmm speedup.

The replacement-shared `004_ProMoE_B_image_shared_e4.yaml` is also supported; it replaces the
shared expert for all images, while `shared_plus_cond_image` retains the shared
expert and adds an image branch only for conditional images. Keep the config fixed
when comparing execution backends. The current pooling design is unchanged.

Earlier benchmark reports reference the former `e3` YAML filenames; the two launch
configs now use `e4`. Reproduce the earlier three-expert setting with
`--expert-counts 3`. Historical reports retain their original paths and source hashes.
