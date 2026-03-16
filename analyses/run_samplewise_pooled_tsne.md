# `run_samplewise_pooled_tsne.py`

## Purpose

This script runs sample-wise t-SNE analysis using pooled block token features.
For each sampled image, it pools the output tokens of every block into a single vector and uses the ImageNet class as the label.

## What It Does

- Accepts a checkpoint path through `--ckpt`.
- Infers the matching YAML config from the checkpoint path.
- Uses `seed=42` by default.
- Samples `5` ImageNet classes by default unless `--class-ids` is provided.
- Generates `50` samples per class by default with `CFG=1.0`.
- Captures block outputs every `50` denoising steps by default.
- Pools each block's token sequence into one sample vector.
- Draws a single `m x n` SVG where:
  `m = number of analyzed blocks`
  `n = number of analyzed denoising steps`
- Saves the SVG under:
  `outputs/<model_name>/<config_stem>/sample/step<step>/t-sne/sample-wise/`

## Output Layout

- rows: blocks
- columns: analyzed denoising steps
- points: generated samples
- colors and legend: ImageNet classes in `class_name-class_index` format

## Main Arguments

- `--ckpt`: Required checkpoint path.
- `--seed`: Random seed for class sampling and latent sampling. Default: `42`.
- `--num-classes`: Number of randomly selected ImageNet classes. Default: `5`.
- `--class-ids`: Optional comma-separated class IDs. Overrides random class sampling.
- `--samples-per-class`: Number of generated samples for each selected class. Default: `50`.
- `--analysis-every`: Capture interval in denoising steps. Default: `50`.
- `--sample-batch-size`: Number of samples processed together by each worker. Default: `8`.
- `--pool-type`: Token pooling type. Default: `mean`. Supported: `mean`, `max`.
- `--perplexity`: Optional manual t-SNE perplexity.
- `--overwrite`: Recreate existing outputs instead of skipping them.

## Example

```bash
python analyses/run_samplewise_pooled_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth
```

```bash
python analyses/run_samplewise_pooled_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --class-ids 7,12,56,207,999 \
  --samples-per-class 30 \
  --pool-type max \
  --overwrite
```

## Notes

- GPUs are resolved from `sample_gpu_ids` or `gpu_ids` in the YAML.
- CUDA is required.
- The script writes a metadata YAML next to the SVG and uses temporary per-worker partial files during execution.
