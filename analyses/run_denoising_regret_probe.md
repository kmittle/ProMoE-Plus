# `run_denoising_regret_probe.py`

## Purpose

This probe tests whether the denoising gradient can cheaply identify a better
expert for one token. It freezes a trained ProMoE checkpoint, replaces the
selected expert with an equal-compute challenger at one token, and compares:

- the first-order MSE change predicted at the chosen MoE block;
- the exact MSE change after rerunning the unchanged suffix of the model.

The result is a feasibility check for denoising-regret routing. It does not
train or modify the checkpoint.

## Method

For each requested noise level, the script samples token positions and uses
the router runner-up, a random non-selected expert, or an even mixture of the
two as challengers. Exact comparisons use matched batches: one normal forward
and one forward with a single forced route per image. This controls numerical
differences caused only by changing CPU/GPU batch shape.

The counterfactual changes only the selected expert index. It keeps the
original top-1 route weight, shared-expert contribution, and AdaLN gate fixed,
so the measured change isolates expert identity at equal inference compute.
Each noise level also reruns a small no-op control that forces the already
selected expert; its maximum absolute MSE change records the numerical floor.

The default device is CPU. Pass an explicit CUDA device only when that GPU is
available and does not overlap a training, sampling, or evaluation job.
On high-latency network filesystems, `--weights-ckpt` can point to a staged
local copy while `--ckpt` remains the canonical path used to resolve config
and output metadata. Probe v4 reads the integer `step` stored in the loaded
checkpoint and rejects it unless it matches the step encoded by the canonical
`--ckpt` filename. Both paths and both step values are retained in the result.

## Output

By default the JSON result is written under:

```text
outputs/<model_name>/<config_name>/sample/step<step>/denoising_regret_probe/
```

It contains aggregate and per-sigma Pearson/Spearman correlation, sign
agreement, the fraction of challengers that truly reduce MSE, precision and
recall for predicted improvements, no-op numerical controls, route-weight
semantics, timings, effective PyTorch thread counts, and every token-level
comparison. Probe v4 records whether each challenger came from the
runner-up or random arm; mixed mode alternates those arms by sampled probe
slot, giving exactly 16 of each for the default 32 probes. The JSON also records
the requested token-probe count, exact-counterfactual batch size, checkpoint
state (`ema_model_state_dict` when available), canonical checkpoint step, and
loaded-weights checkpoint step.

For the multi-image FDRR launch gate, use
`run_denoising_regret_probe_batch.py` and its fixed manifest rather than
manually combining favorable single-image results.

## Example

```bash
python \
  analyses/run_denoising_regret_probe.py \
  --ckpt outputs/ProMoE_TC_REPA_Multi_Align_B/004_ProMoE_B_repa_multi_align_g1_baseline/checkpoints/ckpt_step_10000.pth \
  --latent /path/to/example.latent.npz \
  --label 0 \
  --sigmas 0.2,0.5,0.8 \
  --block-index 3 \
  --num-token-probes 32 \
  --candidate-mode runner-up \
  --device cpu \
  --num-threads 8
```

Use at least several images, seeds, and a checkpoint beyond initial warm-up
before treating the statistics as evidence for or against a full experiment.
