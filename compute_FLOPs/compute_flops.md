# `compute_flops.py`

## Purpose

`compute_flops.py` is the runnable entry script under `compute_FLOPs/`.
It resolves the corresponding YAML configuration from a checkpoint path, uses the sampling-step settings defined in that YAML, and reports FLOPs, activated-parameter statistics, and expert activation frequencies.

## What It Reports

- Conditional-forward FLOPs during sampling
- Activated-parameter statistics for the model
- Overall expert activation frequency for each tracked MoE block
- Average expert activation frequency across all tracked blocks
- Per-step expert activation frequency snapshots saved every `N` denoising steps

## Output Directory

Results are written to:

```text
outputs/<model_name>/<config_name>/sample/step<ckpt_step>/flops_eval/
```

Typical outputs include:

- `flops_result.txt`
- `expert_freq_block_<block_idx>.png`
- `expert_freq_average.png`
- per-step subdirectories such as `step-050/` and `step-100/`

Each `step-xxx/` subdirectory contains:

- expert-frequency bar charts for each block at that step
- `expert_freq_average.png`
- `expert_frequencies.txt`

## Main Arguments

- `ckpt`: Required checkpoint path
- `--num_samples_per_class`: Number of samples generated for each ImageNet class. Default: `5`
- `--seed`: Random seed. Default: `0`
- `--guide_scale`: CFG scale. Default: `1.0`
- `--save_every_steps`: Save per-step expert-frequency reports every `N` denoising steps. Default: `50`

## Example

```bash
python compute_FLOPs/compute_flops.py \
  outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --num_samples_per_class 5 \
  --guide_scale 1.0 \
  --save_every_steps 50
```

## Directory Layout

The directly runnable entry script remains at the root of `compute_FLOPs/`:

- `compute_flops.py`

Reusable helper modules are grouped by function:

- `config/`: checkpoint resolution, YAML loading, and model construction helpers
- `tracking/`: expert-activation and activated-parameter trackers
- `profiling/`: FLOPs counting utilities
- `visualization/`: expert-frequency visualization helpers
