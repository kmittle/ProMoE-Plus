# run_mos_routing_analysis.py

Analyze MoS router teacher block selection patterns during denoising sampling.

## What It Does

- Accepts a checkpoint path through `--ckpt`.
- Infers the matching YAML config from the checkpoint path.
- Runs denoising sampling for selected ImageNet classes, capturing MoS routing weights at specified denoising steps via forward hooks.
- Supports all MoS model variants: MoS (AdaLNRouter), MoS Naive/Naive Choice (BlockRouter), MoS Choice PerBlock (PerBlockRouter), Blockwise.
- Auto-detects model type from model attributes; no manual configuration needed.
- Does NOT require a teacher encoder — routing weights depend only on (x, c); a dummy tensor triggers the routing code path.

## Output

Results are saved to `outputs/{model}/{config}/sample/step{N}/mos_routing/`:

- `per_block_hist.svg` — Per-block teacher block top-1 selection frequency histograms
- `all_blocks_hist.svg` — All aligned blocks aggregated histogram
- `per_block_hist_by_timestep.svg` — Small multiples: block x timestep histograms
- `token_variance.svg` — Token-wise routing weight variance per block
- `routing_entropy.svg` — Routing entropy per block
- `metadata.yaml` — Run parameters

## Main Arguments

- `--ckpt`: Required checkpoint path.
- `--seed`: Random seed. Default: `42`.
- `--num-classes`: Number of randomly selected ImageNet classes. Default: `20`.
- `--class-ids`: Optional comma-separated class IDs (overrides `--num-classes`).
- `--samples-per-class`: Samples per class. Default: `5`.
- `--analysis-every`: Capture every N denoising steps. Default: `50`.
- `--plots`: Comma-separated plot types or `all`. Default: `all`.
- `--vae-path`: Optional local VAE path.
- `--overwrite`: Re-generate even if outputs exist.

## Examples

```bash
python analyses/run_mos_routing_analysis.py \
  --ckpt outputs/ProMoE_TC_REPA_MoS_Naive_Choice_B/004_ProMoE_B_repa_MoS_naive_choice_b3_5/checkpoints/ckpt_step_500000.pth
```

```bash
python analyses/run_mos_routing_analysis.py \
  --ckpt outputs/ProMoE_TC_REPA_MoS_Naive_Choice_B/004_ProMoE_B_repa_MoS_naive_choice_b3_5/checkpoints/ckpt_step_500000.pth \
  --class-ids 0,207,971 \
  --samples-per-class 10 \
  --plots per_block_hist,all_blocks_hist,timestep
```

## Statistics

With default settings (20 classes x 5 samples, analysis_every=50, sample_steps=250):
- 100 images x 256 tokens = 25,600 tokens per timestep point
- 5 timestep points (steps 50, 100, 150, 200, 250)
- Sufficient for 12-teacher-block frequency histograms
