# Analyses Usage

`analyses/` contains runnable entry scripts only. Shared logic and reusable helpers are placed in subdirectories such as `analyses/t_SNE/`.

## Current Entry Scripts

### `run_samplewise_pooled_tsne.py`

Purpose:

- Runs sample-wise t-SNE analysis for a specified checkpoint.
- Infers the YAML config automatically from the checkpoint path.
- Uses a default random seed of `42`.
- Randomly selects 5 ImageNet classes by default, or uses `--class-ids` when provided.
- Generates 50 samples per class by default with `CFG=1.0`; sampling hyperparameters are read from the YAML.
- Captures block output tokens every fixed number of denoising steps, pools them into sample vectors, and uses the ImageNet class as the label for t-SNE visualization.

Default output location:

- `outputs/<model_name>/<config_stem>/sample/step<step>/t-sne/sample-wise/`

Basic usage:

```bash
python analyses/run_samplewise_pooled_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth
```

Common arguments:

- `--ckpt`: Required checkpoint path.
- `--seed`: Random seed, default `42`.
- `--num-classes`: Number of randomly selected classes, default `5`.
- `--class-ids`: Manually specified class IDs, such as `7,12,56,207,999`.
- `--samples-per-class`: Number of generated samples per class, default `50`.
- `--analysis-every`: Denoising interval for analysis, default `50`.
- `--sample-batch-size`: Number of samples processed together by each worker, default `8`.
- `--pool-type`: Token pooling method, default `mean`; `max` is also supported.
- `--perplexity`: Optional manual t-SNE perplexity override.
- `--overwrite`: Re-generate the SVG if it already exists.

Examples:

```bash
python analyses/run_samplewise_pooled_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --num-classes 5 \
  --samples-per-class 50 \
  --analysis-every 50
```

```bash
python analyses/run_samplewise_pooled_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --class-ids 7,12,56,207,999 \
  --samples-per-class 30 \
  --pool-type max \
  --overwrite
```

### `run_tokenwise_tsne.py`

Purpose:

- Runs token-wise routing t-SNE analysis for a specified checkpoint.
- Infers the YAML config automatically from the checkpoint path.
- Uses a default random seed of `42`.
- Randomly selects 20 ImageNet classes by default.
- Uses `CFG=1.0`, with sampling hyperparameters read from the YAML.
- Captures token representations from routed MoE blocks every fixed number of denoising steps and uses the top-1 conditional expert index as the t-SNE label.

Default output location:

- `outputs/<model_name>/<config_stem>/sample/step<step>/t-sne/token-wise/`

Basic usage:

```bash
python analyses/run_tokenwise_tsne.py \
  --ckpt outputs/ProMoE_TC_S/004_ProMoE_S/checkpoints/ckpt_step_500000.pth
```

Common arguments:

- `--ckpt`: Required checkpoint path.
- `--seed`: Random seed, default `42`.
- `--num-classes`: Number of randomly selected ImageNet classes, default `20`.
- `--class-ids`: Manually specified class IDs, such as `1,5,23`; this overrides random class selection.
- `--analysis-every`: Denoising interval for analysis, default `50`.
- `--analysis-batch-size`: Number of classes processed together by each worker, default `4`.
- `--perplexity`: Optional manual t-SNE perplexity override.
- `--overwrite`: Re-generate the SVG if it already exists.

Examples:

```bash
python analyses/run_tokenwise_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --seed 42 \
  --num-classes 20 \
  --analysis-every 50
```

```bash
python analyses/run_tokenwise_tsne.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --class-ids 7,12,56,207 \
  --analysis-batch-size 2 \
  --overwrite
```

Requirements:

- CUDA GPUs are required.
- Visible GPUs are read from `sample_gpu_ids` or `gpu_ids` in the YAML when available.
- `matplotlib` and `scikit-learn` are required in addition to the standard project dependencies.

## Subdirectories

### `t_SNE/`

`t_SNE/` stores the shared modules used by `run_tokenwise_tsne.py` and `run_samplewise_pooled_tsne.py`. These files are not intended to be used as direct task entry points.

- `block_capture.py`: Generic block-output capture helpers.
- `checkpoint_utils.py`: Checkpoint, YAML, output directory, and GPU resolution utilities.
- `imagenet_utils.py`: ImageNet class-name lookup and class-sampling helpers.
- `model_registry.py`: Unified model registry used by analysis scripts.
- `pooling.py`: Token-to-sample pooling utilities.
- `routing_capture.py`: Routing and routed-block capture for token-wise analysis.
- `samplewise.py`: Sample specification, partial-result storage, and merge helpers for sample-wise analysis.
- `sampling.py`: Sampling utilities and denoising-step scheduling.
- `plotting.py`: SVG plotting utilities for token-wise and sample-wise t-SNE.
