# `run_repa_dyna_heatmap.py`

## Purpose

This script visualizes raw `MLP + Sigmoid` token weights from dynamic REPA models as heatmaps overlaid on sampled RGB images.

It is designed for models that expose:

- `encoder_depth`
- `repa_token_weighter`
- a sigmoid-based token-weight prediction head

## What It Does

- Accepts a generated image directory through `--image-dir`.
- Optionally accepts `--ckpt`; otherwise the checkpoint and YAML are inferred from the image directory layout.
- Uses `seed=42` by default.
- Uses `CFG=1.0`.
- Randomly samples `20` ImageNet classes by default unless `--class-ids` is provided.
- Generates `5` samples per class by default.
- Re-runs sampling with the inferred checkpoint and YAML settings.
- Captures token weights every `50` denoising steps by default.
- Adds one extra final heatmap column that uses the per-token average weight over the full denoising trajectory.
- Uses the final RGB image as the background and overlays the token-weight heatmap:
  blue = low weight, red = high weight.
- Writes one SVG per class under:
  `outputs/<model_name>/<config_stem>/sample/step<step>/heatmap/repa_dyna/`

## Important Note

`--image-dir` is used to locate the run, infer the checkpoint/config, and decide where outputs are saved.
The script does **not** read the existing PNGs as analysis inputs.
Instead, it generates a controlled analysis subset with deterministic seeds so that per-step token weights can be captured reliably.

## Output Layout

- one SVG per class
- rows: sampled images for that class
- columns: selected denoising steps, plus a final mean column over all denoising steps
- title: `class_name-class_index`
- colorbar: raw sigmoid token weight in `[0, 1]`

## Main Arguments

- `--image-dir`: Required path to the generated image folder, or its parent sample folder containing `images/`.
- `--ckpt`: Optional checkpoint path. If omitted, it is inferred from `--image-dir`.
- `--vae-path`: Optional local VAE path.
- `--seed`: Random seed for class selection and latent sampling. Default: `42`.
- `--num-classes`: Number of randomly selected ImageNet classes. Default: `20`.
- `--class-ids`: Optional comma-separated class IDs. Overrides random class sampling.
- `--samples-per-class`: Number of generated samples per selected class. Default: `5`.
- `--analysis-every`: Capture interval in denoising steps. Default: `50`.
- `--sample-batch-size`: Number of samples processed together per worker. Default: `4`.
- `--overwrite`: Recreate existing outputs instead of skipping them.

## Example

```bash
python analyses/run_repa_dyna_heatmap.py \
  --image-dir outputs/ProMoE_TC_REPA_DYNA_B/004_ProMoE_B_repa_dyna/sample/step500000/img256_cfg1.0_seed0_FID50K_bs128_ema/images
```

```bash
python analyses/run_repa_dyna_heatmap.py \
  --image-dir outputs/ProMoE_TC_B_hierar_expert_repa_dyna/004_ProMoE_B_hierar_expert_repa_dyna/sample/step500000/img256_cfg1.0_seed0_FID50K_bs128_ema/images \
  --class-ids 1,7,12,281 \
  --samples-per-class 3 \
  --analysis-every 25 \
  --overwrite
```

## Notes

- GPUs are resolved from `sample_gpu_ids` or `gpu_ids` in the YAML.
- CUDA is required.
- The script visualizes the raw `MLP + Sigmoid` prediction before optional extra scaling, masking, or coefficient multiplication.
- A metadata YAML is written next to the SVGs, and temporary per-worker partial files are used during execution.
