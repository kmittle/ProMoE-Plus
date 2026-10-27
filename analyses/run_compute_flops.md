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

For `ProMoE_TC_B_image_shared`, the FLOPs counter temporarily switches the image-wise
shared experts from the default `bmm` backend to `loop`, then restores the previous
backend when stopped. Both backends perform the same selected FFNs. The loop calls
each selected expert's `nn.Linear` modules, allowing the existing THOP hooks to count
them without materializing per-image weight copies. Raw functional `torch.bmm` calls
are not counted by those Linear hooks; changing backend is instrumentation, not a
change in the model's theoretical FFN work.

The image pool's K/V and output Linear projections, single-query attention products,
and image-prototype dot products are included. Counts follow the existing THOP MAC
convention (one multiply-accumulate is one reported operation); memory copies and
normalization/softmax elementwise work are not additional GEMM FLOPs. Unselected image
experts do not perform dummy Linear forwards during evaluation. Activated-parameter
statistics count only the image experts selected during that forward, while pooling
and routing parameters remain always-on. Expert-frequency plots still describe the
original conditional token-wise expert pool.

For `ProMoE_TC_B_shared_plus_cond_image`, the fixed shared expert remains always-on,
while the additional image pool receives only conditional images after label dropout.
The same automatic loop instrumentation counts only those images. On all-unconditional
input, no image-pooling, image-routing, or image-FFN work runs, and none of that pool's
parameters are counted as activated. The original `e4` B experiment uses widths 1536 for shared
and unconditional FFNs, and 768 for conditional token and image FFNs: the selected
FFN widths sum to the dense width of 3072 in both cases. Pooling/router work is extra,
so matching those FFN widths is not an assertion of equal complete-model FLOPs.

The `e4_token_top2` B config instead keeps all expert FFN widths at 1536 and
selects two token experts plus one image expert for each conditional token,
in addition to shared. Its conditional FFN width sum is 6144, the same as base
Top-3 routing plus shared at that width. Unconditional FFN work stays at 3072.
Image pooling and routing remain part of the measured total, so complete-model
FLOPs and total parameters are not claimed to equal the base Top-3 model.

The `e4_image_top2` counterpart selects one token expert and two image experts
(`top_k: 1`, `image_top_k: 2`). Expert counts and widths are unchanged, so its
trainable parameter count matches `e4_token_top2` exactly (367,743,008), and
conditional/unconditional FFN width sums remain 6144/3072. The image pool routes
once per image, executes each selected image/expert pair, and sums cosine-weighted
outputs. Loop instrumentation counts both selected FFNs; activated-parameter
tracking uses the union of selected image expert IDs, without counting unused experts.

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
