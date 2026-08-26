# `run_denoising_regret_probe_batch.py`

## Purpose

This is the offline regret-evidence gate for FDRR. It runs probe v4 over the
fixed ImageNet latent manifest, saves each case atomically, and resumes from
compatible case results. The aggregate gate passes only when every image and
sigma cell passes; a strong global average cannot hide a failed subgroup.

This command is CPU-only by default. It does not train, sample images, evaluate
FID, or authorize an FDRR launch by itself. The separate Base-shape GPU profile
must still show at most 10% training overhead and valid four-GPU memory use.

## Fixed Gate

The checked-in `fdrr_gate_v1.json` manifest uses six classes spread across the
sorted ImageNet label range: 0, 100, 250, 500, 750, and 999. Each case fixes the
synset, latent basename, and probe seed without embedding a machine-specific
dataset root.

Default execution checks 256 tokens for each of sigma 0.2, 0.5, and 0.8. All
18 case/sigma cells must satisfy:

- Spearman correlation at least 0.25;
- first-order/exact sign agreement at least 60%;
- exact beneficial-challenger rate at least 15%;
- no-op absolute MSE drift at most `1e-12`;
- exactly half runner-up and half random challenger sources.

The gate also requires probe version 4, the Base Multi-Align model, a canonical
checkpoint at step 10K or later, fixed selected-route-weight semantics, exactly
the six checked-in cases in their canonical order, 256 comparisons per cell,
and the exact sigma list `(0.2, 0.5, 0.8)`. The manifest payload is locked, not
only its case count.

Each cell summary is recomputed from its raw token records. A case is rejected
if its cached `per_sigma` summary differs from that recomputation. The loaded
weights must contain the same integer checkpoint step as the canonical
`--ckpt` filename; a renamed older checkpoint is rejected.

## Example

Keep `--ckpt` pointed at the original output tree so config and output paths
resolve correctly. `--weights-ckpt` may point to a local staged copy to avoid
reading a checkpoint while training continues.

```bash
python analyses/run_denoising_regret_probe_batch.py \
  --ckpt outputs/ProMoE_TC_REPA_Multi_Align_B/004_ProMoE_B_repa_multi_align_g1_baseline/checkpoints/ckpt_step_10000.pth \
  --weights-ckpt /tmp/promoe-regret-ckpt_step_10000.pth \
  --manifest analyses/denoising_regret/manifests/fdrr_gate_v1.json \
  --latent-root /path/to/sd-vae-ft-mse_Latents_256img_npz \
  --num-threads 8
```

The gate-defining block, device, sigmas, token count, exact batch size, case
count, and thresholds are intentionally not CLI options. Use the single-image
probe entrypoint for diagnostics that need a different specification; its
results cannot be mislabeled as this launch gate.

Compatible per-case JSON files are reused automatically, but path equality is not
enough. `protocol.json` locks the checkpoint, weights, config, manifest, every
latent, the model import closure, probe sources, run settings, and their SHA256
values before a case can be reused. Legacy case files without this provenance are
rejected. Use a fresh output directory, or pass `--overwrite-cases` to start or
resume an intentional recomputation. A compatible sealed pending result from the
same locked protocol is reused; incompatible or unsealed pending work is discarded
before that case is recomputed.

## Output

Results are written under:

```text
outputs/<model_name>/<config_name>/sample/step<step>/
  denoising_regret_probe_batch/fdrr_gate_v1/
```

`protocol.json` and `protocol.sha256` retain the locked inputs and source closure.
Each `cases/*.json` result has a sibling `.seal` that binds its content hash to
the protocol and latent. `summary.json` records both hashes for every case and
contains aggregate metrics, all 18 cell checks, contract failures, threshold
values, provenance, and the final `gate.passed` decision. A completed gate with
any failed check still writes this summary and exits with status 1.
