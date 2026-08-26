# Routing Translation Probe

This checkpoint-backed diagnostic asks whether a ProMoE router follows moved
content or the original absolute token coordinate. It is a causal routing
audit, not an image-quality evaluation and not a training method.

For each noise level, the probe reflect-pads and translates both the sampled
clean latent and its paired noise. The denoising target therefore undergoes the
same translation. At one MoE block it compares five equal-compute executions:

- `native`: the shifted input's original top-1 route;
- `noop_native`: an identical forced route used as a numerical control;
- `content_follow`: the unshifted route map transported with the content;
- `position_follow`: the unshifted route map kept at its old coordinates;
- `random_matched`: randomized replacement IDs with exactly the same changed
  token support and replacement-expert histogram as `content_follow`.

Only expert identity changes. All modes retain the shifted input's native
router weight, execute one expert of the same width per token, and leave every
other block untouched. The valid route region excludes newly reflected border
tokens. Exact denoising MSE still covers the full shifted latent because every
intervention sees the same input and target.

## Example

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python analyses/run_routing_translation_probe.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_50000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_50000.pth \
  --latent /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz/n01440764/n01440764_10027.latent.npz \
  --label 0 \
  --seed 11 \
  --block-index 3 \
  --sigmas 0.276,0.5,0.724 \
  --shifts 0:2,0:-2,2:0,-2:0 \
  --device cpu \
  --num-threads 8 \
  --output /home/dev/promoe-probes/base50k-routing-translation/block3-class000.json
```

The default noise levels are the approximate centers of three equal-probability
regions under the training logit-normal schedule. Latent shifts must be exact
multiples of the model patch size so the content route map has an unambiguous
token-grid translation.

## Interpretation

The primary causal quantity is `content_follow_relative_mse_change`. A route
equivariance gap alone is not evidence of harmful position bias because fixed
DiT positional embeddings intentionally encode spatial priors. Continue only
if content-follow routes reduce exact MSE across independent images, shifts,
noise regions, blocks, and mature checkpoints while position-follow and the
matched random control do not reproduce the gain.

Treat `noop_native` and `forced_unforced` differences as hard numerical
controls. A non-negligible value invalidates the corresponding result. Outputs
load EMA weights when the checkpoint contains them and record the exact state,
canonical checkpoint step, local weights path, thread settings, and every
cell-level result.

Probe version 2 additionally records route_margin for tokens where native and
transported content routes differ. It reports the router-score deficit and rank
of the transported expert plus top-1/top-2 margins for changed and unchanged
tokens. This separates fragile decision-boundary crossings from confident
misrouting; it does not change any intervention or MSE definition.
