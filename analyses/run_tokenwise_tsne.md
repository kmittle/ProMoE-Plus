# `run_tokenwise_tsne.py`

## Purpose

This script runs token-wise t-SNE analysis for routed MoE blocks during diffusion sampling.
It samples ImageNet classes, captures token routing decisions at fixed denoising intervals, and draws one SVG per class.

## What It Does

- Accepts a checkpoint path through `--ckpt`.
- Infers the matching YAML config from the checkpoint path.
- Uses `seed=42` by default.
- Samples `20` ImageNet classes by default unless `--class-ids` is provided.
- Generates one image per selected class with `CFG=1.0`.
- Captures routed token information every `50` denoising steps by default.
- Uses the top-1 conditional expert index as the t-SNE label for each token.
- Saves one SVG per class under:
  `outputs/<model_name>/<config_stem>/sample/step<step>/t-sne/token-wise/`

## Output Layout

Each output figure is arranged as:

- rows: routed blocks
- columns: analyzed denoising steps

The figure title uses the `class_name-class_index` format.
Token colors represent expert labels rather than ImageNet classes.

## Main Arguments

- `--ckpt`: Required checkpoint path.
- `--seed`: Random seed for class sampling and latent sampling. Default: `42`.
- `--num-classes`: Number of randomly selected ImageNet classes. Default: `20`.
- `--class-ids`: Optional comma-separated class IDs. Overrides random class sampling.
- `--analysis-every`: Capture interval in denoising steps. Default: `50`.
- `--analysis-batch-size`: Number of classes processed together by each worker. Default: `4`.
- `--perplexity`: Optional manual t-SNE perplexity.
- `--overwrite`: Recreate existing SVGs instead of skipping them.

## Example

```bash
python analyses/run_tokenwise_tsne.py \
  --ckpt outputs/ProMoE_TC_S/004_ProMoE_S/checkpoints/ckpt_step_500000.pth
```

```bash
python analyses/run_tokenwise_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --class-ids 7,12,56,207 \
  --analysis-batch-size 2 \
  --overwrite
```

## Notes

- GPUs are resolved from `sample_gpu_ids` or `gpu_ids` in the YAML.
- CUDA is required.
- The script writes `analysis_metadata.yaml` into the output directory.
