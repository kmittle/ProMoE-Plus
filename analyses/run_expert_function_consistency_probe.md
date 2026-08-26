# Expert Function Consistency Probe

This checkpoint-backed diagnostic asks whether an expert's function follows
image content under translation and whether that signal predicts the expert's
actual denoising responsibility.

For routed expert `e`, the probe defines its specialized residual relative to
the always-active shared expert:

```text
r_e(h) = E_e(h) - E_shared(h)
```

For a translated token, the primary score is the cosine similarity to the
original content-corresponding residual minus the similarity to the original
same-position residual. This is an MoE function diagnostic, not student/teacher
representation alignment.

The frozen model then routes each sampled token to every routed expert at one
block while preserving the native top-1 gate weight. Each native/candidate pair
runs in the same forward pass, so expert-batch composition cannot turn a no-op
into a numerical difference. All alternatives activate the same number and width
of experts. The probe compares each score with exact full-model denoising-MSE
changes and reports per-token rank correlations, top-expert outcomes, oracle
gaps, and no-op numerical controls.

Example:

```bash
python analyses/run_expert_function_consistency_probe.py \
  --ckpt outputs/ProMoE_TC_B/004_ProMoE_B_seed0_control/checkpoints/ckpt_step_50000.pth \
  --weights-ckpt /home/dev/promoe-probes/base-seed0-ckpt_step_50000.pth \
  --latent /home/dev/imagenet-1k/sd-vae-ft-mse_Latents_256img_npz/n01440764/example.latent.npz \
  --label 0 \
  --block-index 3 \
  --num-token-probes 8 \
  --exact-batch-size 24 \
  --device cuda:4 \
  --output /home/dev/promoe-probes/function-consistency/example.json
```

The canonical checkpoint path resolves the experiment config and step. The
optional local weight copy avoids loading a multi-gigabyte checkpoint from NAS;
its internal step must match the canonical checkpoint.

The output is diagnostic evidence only. It does not establish sample quality,
and no generation claim is valid without the repository's OpenAI 50K-sample
evaluation protocol.
