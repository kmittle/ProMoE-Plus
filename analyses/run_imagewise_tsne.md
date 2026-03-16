# `run_imagewise_tsne.py`

## Purpose

This script runs image-wise t-SNE analysis on already generated images.
It encodes generated PNGs with a pretrained image classifier backbone and uses the feature vector before the classifier head as the image representation.

## What It Does

- Accepts a generated image directory through `--image-dir`.
- Optionally accepts `--ckpt`; otherwise the checkpoint and YAML are inferred from the image directory layout.
- Uses `seed=42` by default.
- Uses pretrained `resnet152` by default.
- Uses the vector before the classification linear head as the image feature.
- Samples `20` ImageNet classes by default unless `--class-ids` is provided.
- Encodes `50` images per class by default.
- Draws one SVG for the selected images and saves it under:
  `outputs/<model_name>/<config_stem>/sample/step<step>/t-sne/image-wise/`

## Output Layout

- points: generated images
- colors and legend: ImageNet classes in `class_name-class_index` format
- feature source: pretrained image encoder before the final classification head

## Main Arguments

- `--image-dir`: Required path to the generated image folder, or its parent sample folder containing `images/`.
- `--ckpt`: Optional checkpoint path. If omitted, it is inferred from `--image-dir`.
- `--seed`: Random seed for class and image selection. Default: `42`.
- `--encoder-name`: Encoder name. Default: `resnet152`.
- `--num-classes`: Number of randomly selected ImageNet classes. Default: `20`.
- `--class-ids`: Optional comma-separated class IDs. Overrides random class sampling.
- `--images-per-class`: Number of encoded images per selected class. Default: `50`.
- `--image-batch-size`: Number of images processed together by each worker. Default: `64`.
- `--perplexity`: Optional manual t-SNE perplexity.
- `--overwrite`: Recreate existing outputs instead of skipping them.

## Example

```bash
python analyses/run_imagewise_tsne.py \
  --image-dir outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/sample/step500000/img256_cfg1.0_seed0_FID50K_bs128_ema/images
```

```bash
python analyses/run_imagewise_tsne.py \
  --image-dir outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/sample/step500000/img256_cfg1.0_seed0_FID50K_bs128_ema/images \
  --class-ids 7,12,56,207 \
  --images-per-class 30 \
  --overwrite
```

## Notes

- GPUs are resolved from `sample_gpu_ids` or `gpu_ids` in the YAML.
- CUDA is required.
- The script writes a metadata YAML next to the SVG and uses temporary per-worker partial files during execution.
- When `--class-ids` is not provided, random sampling is restricted to classes that already have enough generated images.
