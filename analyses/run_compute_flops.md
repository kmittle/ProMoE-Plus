# `run_compute_flops.py`

## Purpose

`run_compute_flops.py` is the runnable FLOPs/statistics entry script under `analyses/`.
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

- `--ckpt`: Required checkpoint path
- `--num-samples-per-class`: Number of samples generated for each ImageNet class. Default: `5`
- `--seed`: Random seed. Default: `0`
- `--guide-scale`: CFG scale. Default: `1.0`
- `--save-every-steps`: Save per-step expert-frequency reports every `N` denoising steps. Default: `50`

Backward compatibility:

- The legacy positional checkpoint argument is still accepted.
- Underscore-style flags such as `--num_samples_per_class`, `--guide_scale`, and `--save_every_steps` are also still accepted.

## Example

```bash
python analyses/run_compute_flops.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --num-samples-per-class 5 \
  --guide-scale 1.0 \
  --save-every-steps 50
```

## Directory Layout

The directly runnable entry script lives at the root of `analyses/`:

- `run_compute_flops.py`

Reusable helper modules live under `analyses/flops/`:

- `checkpoint_utils.py`: checkpoint resolution, YAML loading, and model construction helpers
- `activated_params_tracker.py`: activated-parameter statistics
- `expert_tracker.py`: expert-activation counting and normalization
- `flops_counter.py`: FLOPs counting utilities
- `plotting.py`: expert-frequency visualization helpers
- `tracking_utils.py`: MoE block discovery helpers
