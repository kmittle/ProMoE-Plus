# `run_token_choice_expert_heatmap.py`

## Purpose

This script visualizes token-choice expert routing as heatmaps over the final sampled RGB images.

For each selected ImageNet class, it generates one SVG figure whose grid layout is:

- rows: every MoE block, plus one extra row that averages all MoE blocks
- columns: every selected denoising step, plus one extra column that averages all denoising steps

The heatmap value is the routed expert index for each token:

- smaller expert index: bluer
- larger expert index: redder
- if `top_k > 1`, the script uses the arithmetic mean of the selected expert indices

## What It Does

- Accepts a generated image directory through `--image-dir`.
- Optionally accepts `--ckpt`; otherwise the checkpoint and YAML are inferred from the image directory layout.
- Uses `seed=42` by default.
- Randomly samples `20` distinct ImageNet classes by default unless `--class-ids` is provided.
- Uses the sampling-step count defined in the inferred YAML.
- Captures routed expert indices every `50` denoising steps by default.
- Re-runs deterministic sampling with one sample per selected class so routing can be captured reliably.
- Overlays the routing heatmap on the final RGB image background.
- Saves one SVG per class under:
  `outputs/<model_name>/<config_stem>/sample/step<step>/heatmap/expert_act/`

## Important Note

`--image-dir` is used to locate the experiment run, infer the checkpoint/config, and determine the output directory.
The script does **not** analyze the existing PNG files directly.
Instead, it re-samples a deterministic analysis subset so the per-step routing decisions can be captured from the model.

## Output Layout

- one SVG per class
- title: `class_name-class_index`
- row labels: `MoE Block <idx>` and `Mean All MoE Blocks`
- column labels: `Step <idx>` and `Mean All Steps`
- colorbar: mean routed expert index

## Main Arguments

- `--image-dir`: Required path to the generated image folder, or its parent sample folder containing `images/`.
- `--ckpt`: Optional checkpoint path. If omitted, it is inferred from `--image-dir`.
- `--vae-path`: Optional local VAE path.
- `--seed`: Random seed for class selection and latent sampling. Default: `42`.
- `--num-classes`: Number of randomly selected ImageNet classes. Default: `20`.
- `--class-ids`: Optional comma-separated class IDs. Overrides random class sampling.
- `--analysis-every`: Capture interval in denoising steps. Default: `50`.
- `--sample-batch-size`: Number of classes processed together per worker. Default: `4`.
- `--overwrite`: Recreate existing outputs instead of skipping them.

## Example

```bash
python analyses/run_token_choice_expert_heatmap.py \
  --image-dir outputs/ProMoE_TC_S/004_ProMoE_S/sample/step500000/img256_cfg1.0_seed0_FID50K_bs128_ema/images
```

```bash
python analyses/run_token_choice_expert_heatmap.py \
  --image-dir outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/sample/step500000/img256_cfg1.0_seed0_FID50K_bs128_ema/images \
  --class-ids 1,7,12,281 \
  --analysis-every 25 \
  --overwrite
```

## Notes

- GPUs are resolved from `sample_gpu_ids` or `gpu_ids` in the YAML.
- CUDA is required.
- Only MoE blocks are visualized; dense blocks are skipped automatically.
- A metadata YAML is written next to the SVGs, and temporary per-worker partial files are used during execution.
