# Repository Guidelines

## Project Overview
ProMoE-Plus implements ProMoE, a Mixture-of-Experts framework for scaling Diffusion Transformers on ImageNet class-conditional generation. The core routing design combines conditional routing for cond/uncond token separation with prototypical routing through learnable cluster centers, plus routing contrastive guidance for expert specialization.

## Project Structure & Module Organization
Core entrypoints are at repository root:

- `train.py`: baseline and non-REPA training (DiT, TCDiT, ECDiT, DiffMoE, ProMoE, hierarchical, expert-choice/batch-choice, structured-batch, proto-t, anchor, proto-choice, load-balance-contrastive, DAG-fuse, shared-expert DAG-fuse, adaptive-depth, loss-free, first-order denoising-regret routing, label-smoothing regularization, noise-expert, and expert-contrastive, and regularizer-combination (`regcombo`) variants).
- `train_with_repa.py`: REPA-enabled training (REPA / REPA-Shared / REPA-Cond / REPA-DYNA / REPA-DYNA-SELECT / REPA-DYNA-SCALE / REPA-DYNA-ONLY / REPA-Router / REPA-Router-Contra / REPA-Routed / REPA-Double-Share / heterogeneous-expert REPA-DYNA), including teacher-feature alignment loss.
- `train_with_MoS_repa.py`: MoS-REPA, MoS-Naive / Naive-Choice, separate-projector / per-block / blockwise / fused / multi-align, Teacher-Affinity Routing, shared-routed spectral responsibility, teacher-conditioned expert geometry, first-order denoising-regret routing, and both standard-REPA + MoS cross-alignment training with teacher-block routing and per-block REPA projectors.
- `train_with_mae.py`: MAE/group-alignment training for `group_align` and `group_align_proj` variants.
- `sample.py`: sampling/inference entrypoint. It merges model registries from `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, and `train_with_mae.py`, so one script can sample all registered families.

Shared defaults and helpers:

- `config.py`: global defaults (`cfg`) and model templates (`DiT_*_config`, `DiffMoE_*`, etc.).
- `utils.py`: `deep_update`, `find_free_port`, VAE caching/loading (`load_vae`), `TrainingMonitor`, CLI list parsers, and Inception utilities.
- `scripts/check_output_dir.py`: output-directory collision guard for new or rerun experiments.
- `scripts/template.sh`: required base pattern for new train + sample + eval experiment wrappers.
- `doc/ProMoE-REPA.md`: repo-level guide for current REPA / MoS-REPA workflows and variants.

Main code layout:

- `models/`: architecture implementations; keep the `models_*.py` naming convention.
- `model.py`: root-level reference file that is not imported by the main training/sampling entrypoints; project model implementations live under `models/`.
- `repa/`: REPA helper package used by `train_with_repa.py` and `train_with_MoS_repa.py` (`encoder.py`, `loss.py`).
- `preprocess/`: VAE latent preprocessing, parquet-direct latent encoding, ImageNet preparation helpers, and shared cache files such as `preprocess/image_paths_cache.txt` and `preprocess/latent_paths_cache.txt`.
- `evaluation/`: OpenAI-style evaluation pipeline (`run_eval.py`, `evaluator.py`, `download_ref_batches.py`).
- `analyses/`: analysis entry scripts live directly under this directory. Current entrypoints include the t-SNE, REPA-DYNA heatmap, FLOPs, MoS-routing, denoising-regret, routing-equivariance, timestep-utility, compute-exchange, phase-default, expert-update-budget, learning-credit-balance, DINO-neighborhood, affinity-responsibility, CFG-route-inversion, and expert-function-consistency probes listed in `analyses/README.md`. Reusable helpers live under `analyses/t_SNE/`, `analyses/heatmap/`, `analyses/flops/`, `analyses/mos_routing/`, `analyses/denoising_regret/`, `analyses/routing_translation/`, `analyses/timestep_utility/`, `analyses/routing_metric/`, `analyses/phase_default/`, `analyses/expert_update_budget/`, and `analyses/dino_utility_neighborhood/`; `analyses/README.md` should remain a brief directory-level overview only, while detailed usage belongs in per-entry Markdown files that share the same basename as each entry `.py`.
- `scripts/repa/`: REPA-B / REPA-Shared-B / REPA-Cond-B helpers plus router / routed / double-share, cross-alignment, and L/XL scale-up train + sample + eval wrappers.
- `scripts/MoS_repa/`: MoS-REPA, naive / naive-choice, per-block / blockwise / fused, multi-align, Teacher-Affinity Routing, shared-routed spectral responsibility, teacher-conditioned expert geometry, first-order denoising-regret routing, cross-alignment, and B/L/XL block-range sweep wrappers following `scripts/template.sh`.
- `scripts/hierar/`: B-scale hierarchical + heterogeneous-expert train + infer/eval wrappers.
- `scripts/dynamic_repa/`: REPA-DYNA-B, REPA-DYNA-SELECT-B, REPA-DYNA-SCALE-B, and REPA-DYNA-ONLY-B train + sample + eval pipelines (including select-ratio variants r25/r75).
- `scripts/mae_align/`: group-align and group-align-proj train + sample + eval wrappers.
- `scripts/noise_expert/`: noise-expert, proj, EMA-on-shared, and EMA-on-noise train + sample + eval wrappers.
- `scripts/expert_contra/`: expert-contrastive output/param train + sample + eval wrappers, including B4 ablations.
- `scripts/expert_choice/`: expert-choice batch-choice (`EC_BC`) train + sample + eval wrappers.
- `scripts/structured_batch/`: token-choice and expert-choice structured-batch train + sample + eval wrappers.
- `scripts/proto_t/`: token-choice and expert-choice proto-t direct/residual train + sample + eval wrappers.
- `scripts/anchor/`: anchor-routing and anchor-replace train + sample + eval wrappers.
- `scripts/proto_choice/`: contrastive proto-choice ratio sweep train + sample + eval wrappers.
- `scripts/lbcontra/`: load-balance-aware routing-contrastive reweight/logit-adjust/balance-term/soft-only train + sample + eval wrappers.
- `scripts/dagfuse/`: DAG-MoE shared/conditional fusion train + sample + eval wrappers.
- `scripts/dagfuse_shared/`: shared-expert augmentation wrappers (`dense`, `densenet`, `sharedroute`, and region attach/mechanism sweeps).
- `scripts/adepth/`: adaptive routed-FFN depth fixed-quota train + sample + eval wrappers.
- `scripts/image_shared/`: independent base-TC image-expert ablations: replace the shared branch for all images, or retain the shared branch and add conditional-only image experts. Both use cross-attention pooling and image routing contrastive guidance.
- `scripts/lossfree/`: loss-free balancing-bias train + sample + eval wrappers.
- `scripts/lsreg/`: routing-contrastive label-smoothing and diagonal-correction sweeps.
- `scripts/capacity_combo/`: capacity-aware H/R/O/P expert-responsibility study. Only the
  `HROP` arm was launched; it failed the 300K gate. The nine never-launched arms (`H`,
  `HO`, `HP`, `HR`, `HOP`, `HRO`, `HRP`, `HO_norm`, `HROP_norm`) and the old queue were
  deleted on 2026-09-14 pending a redesign. The retained directory contains only the
  `HROP` wrapper, its evaluation helper, and the `hrop_gate_then_q0p4.sh` supervisor.
- `scripts/dino_route/`: DINO-assisted load-aware routing experiments and shuffled controls.
- `scripts/phase_metric/`: phase-conditioned routing-metric experiments and controls.
- `scripts/fdrr/`: teacher-free first-order denoising-regret routing and its control.
- `scripts/regcombo/`: the 2026-10-03 regularizer-combination study (`ProMoE_TC_B_regcombo`): global center + LS-Reg + expert parameter or output repulsion arms, LS-Reg with global counts, and output repulsion with global pooling.
- `scripts/_run_times/`: dated launch wrappers; each date directory carries a maintained `commands.md` launch table and adjacent `*-describe.txt` experiment notes. `commands.csv` is a separate optional export.
- `command-tables/`: CSV template assets for run-time command tables.
- `collapse_smoking_test/`, `collapse_smoking_test_10k/`: crash-diagnosis smoke configs, logs, summaries, and rerun helpers for cross-alignment stability work.
- `tb_smoke_200/`, `tb_smoke_500/`: TensorBoard/`TrainingMonitor` smoke harnesses for selected cross-alignment configs.
- `research_on_expert_learning_signal_balance/`: retained research note and the shared `git_provenance.py` utility. The former credit-rate continuation implementation, configs, launchers, and tests were removed; do not treat archived continuation outputs as experiments or paper evidence.
- `REPA/` (uppercase): separate upstream-style subproject with its own docs and `AGENTS.md`.

Top-level wrappers:

- `scripts/run_all_infer_eval_500K.sh`: batch sample+eval for multiple configs at 500K.
- `scripts/eval_B_hetero_expert.sh`, `scripts/eval_B_hetero_expert_NoPenalty.sh`: eval-only wrappers.

Companion documentation:

- `doc/ProMoE-REPA.md`: detailed REPA / MoS-REPA workflow, configuration reference, and FAQ.
- `analyses/README.md`: analysis entrypoint overview; keep per-script usage in matching `analyses/<basename>.md` files.
- `plans/`: implementation plans for standard REPA and MoS cross-alignment variants.
- `doc/contrastive-label-smoothing.md`: LS-Reg design, experiment matrix, and evaluation discipline.
- `doc/load-balance-design.md`: explicit load-balance intervention design notes (reference only).
- `.claude/skills/`: project-local workflow descriptions for inspect/check/new-experiment/rerun-experiment/command-table/describe-experiment helpers; useful as procedural reference even when not running Claude slash commands.
- `.agents/skills/`: active Codex workflow definitions. For Codex experiment work, use `command-table`, `describe-experiment`, `new-experiment`, and `rerun-experiment` from this directory rather than translating Claude slash-command mechanics literally.
- `.codex/skills/`: project-local Codex helper skills.
- `doc/implementation-plan.md`: Chinese draft plan for a future attention-weighted same-expert same-image alignment family; reference only, not current code.
- `_previous_results/`: archived experiment result tables and renders. They are reference data only; no current queue or training script reads them.

Outputs follow:

`outputs/<model_name>/<config_stem>/`

with:

- `checkpoints/ckpt_step_*.pth`
- `training.log`, `sample.log`
- `tensorboard/`
- `sample/step<step>/img<...>/images`
- `sample/step<step>/<analysis_name>/` for analysis artifacts such as `flops_eval/` and t-SNE outputs

## Current Research Scope and Storage Policy

The active research line is the factorial combination of the four validated
MoE mechanisms:

- `H`: heterogeneous routed experts;
- `R`: LS-Reg routing-load balancing;
- `O`: expert-output representation regularization;
- `P`: expert-parameter regularization.

The validated combination design matrix is `H`, `HO`, `HP`, `HOP`, `HR`, `HRO`,
`HRP`, and `HROP`. New training, sampling, or evaluation work should stay
within this matrix unless the user explicitly reopens another direction. REPA
and DINO are not the current main line; DINO may only return as a router/load/
specialization signal, never as direct feature alignment.

**Redesigned combination study (2026-10-03, user decision).** The user reopened
the combination work with a new design that replaces the H/R/O/P matrix:
global routing-contrastive class centers are on in every combination arm, the
heterogeneous experts (`H`) are deferred to the end, and the remaining modules
are combined in pairs before stacking more, in the priority LS-Reg > expert
parameter regularization > expert output regularization. The first batch
(`scripts/_run_times/2026_10_03/`, model `ProMoE_TC_B_regcombo`) has two
four-arm groups on 2 GPUs each -- LS-Reg + parameter repulsion and LS-Reg +
output repulsion -- where each group holds one anchor with its single-module
coefficients and three arms sweeping the L2 temperature only; plus two separate
studies of the global mechanism, LS-Reg with global counts (2 GPUs, like
`lsreg_diag_idea1_s0p05`) and output repulsion with global pooling (4 GPUs, like
`expert_contra_output`). By the user's decision every run of this batch is
evaluated at 300K and 500K without a 300K stop, the 2-GPU arms are compared with
each other rather than with the 4-GPU baseline, and no multi-seed repeat is run.

As of 2026-09-14, only the `HROP` combination was actually launched. Its 300K
CFG 1.0/1.5 FIDs were `31.19`/`10.07`; it was abandoned before 500K. These
also fail the canonical Base gate (`30.86`/`9.73`). The other nine planned
combination arms were never launched and their configs, wrappers, descriptions,
and queue supervisor were deleted pending a redesigned study. Do not recreate
or schedule those arms from the historical design notes without an explicit
redesign decision.

Adaptive-depth (`scripts/adepth/`) is a separate historical exploration, not
part of the H/R/O/P factorial study. Its repeated application of the same
MLP is not evidence of meaningful additional reasoning. New adaptive-depth
variants require the user to reopen this direction explicitly. An already-running
adaptive-depth job must still reach its declared 300K evaluation boundary before the 300K FID gate is
applied; do not stop it early from an intermediate loss or checkpoint.

The canonical ProMoE-TC B baseline is fixed (user rule, 2026-09-21): at 300K,
FID/IS are `30.86 / 48.35` for CFG 1.0 and `9.73 / 121.25` for CFG 1.5;
at 500K, they are `24.44 / 60.38` and `6.39 / 154.21`, respectively. The
separately trained "fresh routing baseline" is void and must not be used for
comparisons or gates.

Every active combination experiment must be training-from-scratch with
`global_seed: 0`, global batch `256`, learning rate `1e-4`, and
`img_num_workers: 16`, using four GPUs (the 2026-10-03 `regcombo` batch above is
the user-approved exception: 2 GPUs for its combination arms and the LS-Reg
global-count run, and 300K + 500K evaluation without a gate). It must evaluate 300K at CFG 1.0 and
1.5 with 50K samples. Only a candidate whose two 300K FIDs are both strictly
better than the canonical Base gate (`30.86`/`9.73`) may continue to 500K.
Do not infer success or failure from 50K/200K checkpoints, training loss, or
an auxiliary loss alone.

On 2026-10-26 the user explicitly selected `B_adepth_q0p3_p_only`,
`B_clean_route_student_ratio0_corr_router`, and `B_corr_router` for continuation
from their own 300K checkpoints to 500K, on four GPUs each. The two correction-router
arms are approved exploratory exceptions to the 300K dual-CFG gate. Their configs
under `configs/resume_500k/` intentionally retain the original YAML basenames and
model names, so checkpoints and 500K evaluations stay in the original output buckets.
These are same-run extensions, not fresh experiments or `_vN` reruns. The launchers
require an existing checkpoint at or beyond 300K, select the greatest saved step up
to 500K, and pin that step via `resume_checkpoint_step`; they must not start from scratch.

On 2026-10-30 the user set today's three new training experiments
(`B_clean_route_student_ratio0_align_delay100k`,
`B_clean_route_student_ratio0_align_warmup200k`, and `B_top3`) to train from
scratch to 300K, evaluate, resume their own weights to 500K, then evaluate again.
Both checkpoints use CFG 1.0/1.5 with 50K samples per CFG on four GPUs. These
three arms continue after a successful 300K evaluation without a FID threshold gate.

The output bucket must be a real directory under `outputs/`; do not use
soft-links for experiment results. `/home/dev` must not contain any
`promoe*` directory. Research papers and source packages belong in the
repository-level `literature/` directory. Keep only the checkpoints and
analysis artifacts needed for reproducibility; regenerable PNG samples may be
discarded after evaluator artifacts are verified.

Never write experiment results outside this repository. Checkpoints, logs,
samples, evaluator outputs, TensorBoard data, metrics, manifests, and analysis
artifacts must use paths under the repository's `outputs/` or other
repository-tracked experiment directories. External dataset, VAE, and teacher
checkpoint caches are input dependencies and do not authorize placing outputs
under `/home/dev`, `/tmp`, or any other path outside the repository.

Do not revive deleted continuation studies by loading a checkpoint from another
experiment. Any future credit-redistribution or similar optimization study
must be implemented as a fresh, step-0-controlled experiment with matched
seeds, data order, optimizer settings, and the same 300K dual-CFG gate as the
active H/R/O/P matrix.

## Build, Test, and Development Commands
Create training env:

```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```

Run standard training:

```bash
python train.py --config configs/004_ProMoE_L.yaml
```

Run REPA training (optional local teacher checkpoint):

```bash
python train_with_repa.py \
  --config configs/004_ProMoE_B_repa.yaml \
  --repa-enc-path /path/to/state_dict.pth
```

Run MoS-REPA training (optional local teacher checkpoint):

```bash
python train_with_MoS_repa.py \
  --config configs/004_ProMoE_B_repa_MoS.yaml \
  --repa-enc-path /path/to/state_dict.pth
```

Run MAE/group-alignment training:

```bash
python train_with_mae.py --config configs/004_ProMoE_B_mae_align.yaml
```

Optional local VAE path (skips auto-download/cache in train/sample):

```bash
python train.py --config configs/004_ProMoE_B.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_repa.py --config configs/004_ProMoE_B_repa.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_MoS_repa.py --config configs/004_ProMoE_B_repa_MoS.yaml --vae-path /path/to/sd-vae-ft-mse
python train_with_mae.py --config configs/004_ProMoE_B_mae_align.yaml --vae-path /path/to/sd-vae-ft-mse
python sample.py --config configs/004_ProMoE_B_repa.yaml --vae-path /path/to/sd-vae-ft-mse
```

Precompute VAE latents:

```bash
python preprocess/preprocess_vae.py \
  --latent_save_root /path/to/ImageNet/sd-vae-ft-mse_Latents_256img_npz
```

Sampling (explicit step/CFG/num samples):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python sample.py \
  --config configs/004_ProMoE_B_repa.yaml \
  --step_list_for_sample 300000,500000 \
  --guide_scale_list 1.0,1.5 \
  --num_fid_samples 50000
```

Run FLOPs / activated-parameter / expert-frequency analysis:

```bash
python analyses/run_compute_flops.py \
  --ckpt outputs/ProMoE_TC_REPA_B/004_ProMoE_B_repa/checkpoints/ckpt_step_500000.pth \
  --num-samples-per-class 5 \
  --guide-scale 1.0 \
  --save-every-steps 50
```

Run MoS routing analysis:

```bash
python analyses/run_mos_routing_analysis.py \
  --ckpt outputs/ProMoE_TC_REPA_MoS_Naive_Choice_B/004_ProMoE_B_repa_MoS_naive_choice_b3_5/checkpoints/ckpt_step_500000.pth
```

Common wrappers:

```bash
bash scripts/repa/train_repa_B.sh
bash scripts/repa/train_repa_shared_B.sh
bash scripts/repa/train_repa_cond_B.sh

bash scripts/repa/sample_and_eval_repa_B.sh
bash scripts/repa/sample_and_eval_repa_shared_B.sh
bash scripts/repa/sample_and_eval_repa_cond_B.sh
bash scripts/repa/run_B_repa_cross_global_pre_train_sample_eval.sh
bash scripts/repa/run_B_repa_cross_global_block_train_sample_eval.sh
bash scripts/repa/run_B_repa_cross_expert_local_train_sample_eval.sh
bash scripts/repa/run_B_repa_cross_proto_train_sample_eval.sh
bash scripts/repa/run_B_repa_router_train_sample_eval.sh
bash scripts/repa/run_B_repa_router_contra_train_sample_eval.sh
bash scripts/repa/run_B_repa_routed_train_sample_eval.sh
bash scripts/repa/run_B_repa_double_share_train_sample_eval.sh

bash scripts/MoS_repa/run_B_repa_mos_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_b3_5_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_sep_b3_5_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_choice_per_block_b3_5_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_blockwise_b3_5_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_naive_choice_b3_5_fused_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_multi_align_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_multi_align_no_dynamic_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_cross_global_pre_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_cross_global_block_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_cross_expert_local_train_sample_eval.sh
bash scripts/MoS_repa/run_B_repa_mos_cross_proto_train_sample_eval.sh

bash scripts/dynamic_repa/run_B_repa_dyna_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_r25_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_select_r75_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_scale_train_sample_eval.sh
bash scripts/dynamic_repa/run_B_repa_dyna_only_train_sample_eval.sh

bash scripts/hierar/run_B_hierar_train.sh
bash scripts/hierar/run_B_hierar_infer_eval.sh
bash scripts/hierar/run_B_hetero_expert_train.sh
bash scripts/hierar/run_B_hetero_expert_infer_eval.sh
bash scripts/hierar/run_B_hetero_expert_NoPenalty_train.sh
bash scripts/hierar/run_B_hetero_expert_NoPenalty_infer_eval.sh
bash scripts/hierar/run_B_hetero_expert_repa_dyna_train_sample_eval.sh

bash scripts/mae_align/run_B_mae_align_train_sample_eval.sh
bash scripts/mae_align/run_B_mae_align_proj_train_sample_eval.sh
bash scripts/noise_expert/run_B_noise_expert_train_sample_eval.sh
bash scripts/noise_expert/run_B_noise_expert_proj_train_sample_eval.sh
bash scripts/noise_expert/run_B_noise_expert_ema_on_shared_train_sample_eval.sh
bash scripts/noise_expert/run_B_noise_expert_ema_on_noise_train_sample_eval.sh
bash scripts/expert_choice/run_B_ec_bc_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_output_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_output_b4_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_param_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_param_b4_train_sample_eval.sh
bash scripts/structured_batch/run_B_tc_structbatch_train_sample_eval.sh
bash scripts/structured_batch/run_B_ec_bc_structbatch_train_sample_eval.sh
bash scripts/proto_t/run_B_tc_proto_t_direct_v2_train_sample_eval.sh
bash scripts/proto_t/run_B_tc_proto_t_residual_v2_train_sample_eval.sh
bash scripts/proto_t/run_B_ec_bc_proto_t_direct_v2_train_sample_eval.sh
bash scripts/proto_t/run_B_ec_bc_proto_t_residual_v2_train_sample_eval.sh
bash scripts/anchor/run_B_anchor_routing_train_sample_eval.sh
bash scripts/anchor/run_B_anchor_replace_train_sample_eval.sh
bash scripts/proto_choice/run_B_proto_choice_083_train_sample_eval.sh
bash scripts/proto_choice/run_B_proto_choice_125_train_sample_eval.sh

bash scripts/lbcontra/run_B_lbcontra_reweight_b0p25_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_reweight_b0p5_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_reweight_b1_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_reweight_b2_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_logitadj_t0p5_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_logitadj_t1_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_logitadj_t2_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_logitadj_t4_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_balance_l0p001_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_balance_l0p01_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_balance_l0p1_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_balance_l1_train_sample_eval.sh
bash scripts/lbcontra/run_B_lbcontra_soft_only_train_sample_eval.sh
bash scripts/dagfuse/run_B_dagfuse_condfromshared_train_sample_eval.sh
bash scripts/dagfuse/run_B_dagfuse_sharedfromcond_train_sample_eval.sh
bash scripts/dagfuse/run_B_dagfuse_bidirectional_train_sample_eval.sh
bash scripts/dagfuse_shared/run_B_dagfuse_dense_all_train_sample_eval.sh
bash scripts/dagfuse_shared/run_B_dagfuse_densenet_all_train_sample_eval.sh
bash scripts/dagfuse_shared/run_B_dagfuse_sharedroute_all_top1_train_sample_eval.sh
bash scripts/dagfuse_shared/run_B_dagfuse_region_shared_all_dag_train_sample_eval.sh
bash scripts/adepth/run_B_adepth_q0p1_train_sample_eval.sh
bash scripts/adepth/run_B_adepth_q0p2_train_sample_eval.sh
bash scripts/adepth/run_B_adepth_q0p3_train_sample_eval.sh
bash scripts/adepth/run_B_adepth_q0p4_train_sample_eval.sh
bash scripts/lossfree/run_B_lossfree_u1e2_train_sample_eval.sh
bash scripts/lossfree/run_B_lossfree_u1e3_train_sample_eval.sh
bash scripts/lossfree/run_B_lossfree_u1e4_train_sample_eval.sh
bash scripts/lsreg/run_B_lsreg_fixed0p10_train_sample_eval.sh
bash scripts/lsreg/run_B_lsreg_dynboth_train_sample_eval.sh
bash scripts/lsreg/run_B_lsreg_diag_idea1_s0p30_train_sample_eval.sh
bash scripts/lsreg/run_B_lsreg_diag_inv_s0p40_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_param_cos_train_sample_eval.sh
bash scripts/expert_contra/run_B_expert_contra_param_shared_uncond_train_sample_eval.sh

bash tb_smoke_200/run_all.sh
bash tb_smoke_500/run_all.sh

bash scripts/run_all_infer_eval_500K.sh
```

### Shell Script Convention

When adding any new train + sample + eval all-in-one `.sh` wrapper, start from the current `scripts/template.sh`; this is required for compatibility with the experiment server. Read `CLAUDE.md`'s "Shell Script Convention" alongside this section. Source `scripts/_python_env.sh` for the pinned absolute interpreters (`/mnt/workspace/yujie/.conda/envs/promoe/bin/python` and `/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python`). Do not re-hardcode them or rely on `conda activate` at runtime.

Template-specific requirements:

- `scripts/template.sh` uses a sequential train-stop-sample/eval-resume loop. For each `step_list_for_sample` item, it trains to that checkpoint, exits, frees GPUs for sample/eval, then resumes training for the next step.
- New scripts should only change `CONFIG`, `LOG`, and the training entrypoint (`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`) unless the experiment genuinely needs extra logic.
- The template defaults `PROMOE_LATENT_PATH` to `/mnt/data2/yujie/code/Dataset/ImageNet/sd-vae-ft-mse_Latents_256img_npz` and writes it into the temporary training YAML's `latent_data_path`; exporting the variable alone would be overridden by an older YAML value. An explicitly supplied `PROMOE_LATENT_PATH` takes precedence. The 2026-10-26 continuation configs also record this deployment path directly.
- Preserve `set -euo pipefail`, repo-root discovery via `SCRIPT_DIR` / `REPO_ROOT`, inline Python YAML parsing, absolute Python interpreter paths, and `find ... -name images | sort -V` evaluation traversal.
- Keep the temporary YAML's basename identical to the original config so `custom_cfg_name` does not change. Every phase sets `resume_checkpoint: True`; intermediate phases use `num_steps = step + 1`, and the final phase uses the original `num_steps`. The fresh-output check, not a per-phase resume flag, ensures a fresh launch starts at step 0.
- Preserve `PROMOE_GPU_IDS_OVERRIDE` validation and propagation: it must contain distinct GPU IDs and the same count as YAML `gpu_ids`. Write the actual IDs into the temporary training YAML, use them for sampling, and use the first one for evaluation. Preserve the parsed-field count check so a failed inline Python process substitution cannot silently continue.
- Default `PROMOE_RESUME=0` requires an empty, real output directory. `PROMOE_RESUME=1` requires this run's own checkpoint; if the phase checkpoint exists, skip training, and skip the whole phase only when every requested CFG has valid evaluator metrics. Do not substitute another experiment's checkpoint.
- Write combined logs to `LOG="${REPO_ROOT}/logs/log_<name>.log"` and create the parent with `mkdir -p "$(dirname "$LOG")"`. Keep per-experiment `training.log` / `sample.log` under its output directory. Logs must not contain ANSI colour codes; retired logs belong under `logs/archived/<date>/` (see `logs/README.md`).
- Preserve `log_code_version`: each executed phase records the Git commit, modified tracked-file count, root Python and config hashes, and combined `models/*.py` / `repa/*.py` hashes in the combined log.
- Legacy split scripts such as `scripts/repa/train_repa_B.sh` and `scripts/repa/sample_and_eval_repa_B.sh` predate the template; do not introduce new split-purpose experiment scripts.
- Keep `scripts/_python_env.sh`'s fail-fast behavior before touching an output bucket when pinned interpreters are unavailable. Local interpreter fallback requires explicit `PROMOE_ALLOW_LOCAL_FALLBACK=1`.
- Reuse the strict evaluator predicates in `scripts/_eval_metric_helpers.sh` (for example `promoe_eval_file_metrics_valid` and `promoe_eval_file_fid`) for gate decisions; do not parse evaluator output with an ad-hoc command that can turn malformed values into zero.
- The template executes the configured checkpoints; it does not implement the 300K FID gate itself. Preserve or add the required experiment/supervisor gate explicitly before allowing 500K, except for a documented user-approved no-gate study.

### Run-Time GPU Allocation

- After creating a semantic script under `scripts/<family>/`, create its launch wrapper with `scripts/_run_times/new_run.sh --script scripts/<family>/run_<...>.sh --gpus N [--date YYYY_MM_DD] [--desc <name>] [--dry-run]`. Use `--dry-run` first, inspect the plan, then write. The allocator accepts any positive integer; experiment defaults are 4 GPUs, or 8 for XL.
- `new_run.sh` writes placeholder YAML `gpu_ids: [0, ..., N-1]` to fix DDP world size, and creates `scripts/_run_times/<date>/<desc>.sh` with no slot prefix. `--date` only groups files; `--desc` defaults to the distinguishing semantic-script name. It refuses to overwrite an existing wrapper.
- At launch, the wrapper sources `scripts/_run_times/claim_gpus.sh`, calls `promoe_claim_gpus N` directly (never through command substitution), and exports `PROMOE_GPU_IDS_OVERRIDE` before `exec bash` of the semantic script. A manual override takes precedence; the semantic script still validates its count.
- Idle GPUs have memory use at or below `PROMOE_GPU_IDLE_MAX_MIB` (default 100 MiB) and no live PID claim. Selection uses a shared `flock` and claims under `/tmp/promoe_gpu_claims`; stale claims are reusable. Too few idle GPUs is a hard failure, not a queue. Do not reimplement this logic in experiment scripts.
- All arms of one comparison must use the same GPU count and valid global-batch divisibility: per-rank routing statistics change with world size. A 2-GPU arm is not directly comparable with the 4-GPU canonical baseline.
- Pre-2026-09-23 wrappers retain their legacy slot prefixes and fixed GPU assignments. The `2026_08_05` batch's `.1`/`.2`/`.3`/`.4` prefixes represent historical manual 2-GPU packing; neither legacy layout governs new wrappers.
- Queue or supervisor scripts that chain experiments must run inside an attached tmux session, hold a repository-local `flock`, and advance only after the preceding wrapper exits. They must not use shell backgrounding (`&`, `nohup`, or detached execution).

### Run-Time Command Tables and Descriptions

Every affected `scripts/_run_times/<date>/` directory must carry `commands.md`. Create or refresh it whenever a wrapper is added, renamed, or removed; `new_run.sh` does not generate it, so the calling workflow must. The file contains only a Markdown table with this exact header, a separator row, and one row per launch wrapper:

```text
| 实验描述 | git分支 | 启动命令 | 输出位置 |
|---|---|---|---|
| <中文实验描述>（N 卡） | <branch> | bash scripts/_run_times/<date>/<desc>.sh | outputs/<model_name>/<custom_cfg_name>/ |
```

Use plain-text cells with no wrapping quotes or backticks, no title, introduction, or notes. Sort current wrapper filenames; sort legacy slot prefixes naturally, preserving their `Slot X.Y · GPU a-b ·` description prefix. Resolve each row through wrapper → semantic script → YAML, use the current branch unless a per-experiment branch is specified, and derive output paths from the effective output root. Exclude helpers. Put unresolved-row notes in chat.

After creating or re-bucketing a wrapper, generate its adjacent `<wrapper-stem>-describe.txt` with `describe-experiment`. The Markdown table is the maintained launch index; the `command-table` skill separately exports RFC 4180 `commands.csv` for Notion/Excel and does not replace or retire `commands.md`. Report an existing CSV as stale when its wrapper set changes, unless CSV regeneration was also requested.

### Evaluation Environment

Create evaluation env (TensorFlow-based):

```bash
conda create -n fid_eval python=3.9 -y
conda activate fid_eval
cd evaluation
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python run_eval.py /path/to/generated/images --count 50000
```

The pinned environment preserves the historical OpenAI evaluation stack. The
`run_eval.py` default `--eval-device auto` uses a visible GPU only when this
TensorFlow build covers its compute capability; on newer architectures such as
Hopper it falls back to CPU rather than emitting invalid activations.

Pack PNGs to NPZ without running evaluator:

```bash
python run_eval.py /path/to/generated/images --count 50000 --no-eval
```

## Coding Style & Naming Conventions
- Follow existing style in touched files: 4-space indentation, `snake_case` functions/variables, `PascalCase` classes.
- Preserve existing import grouping and logging-heavy style.
- Keep config filenames in the numeric-prefixed style (for example `004_*.yaml`).
- Wrapper/config names like `b3_5` are human-readable 1-indexed block ranges; YAML `align_blocks` stays 0-indexed Python-style (for example `b3_5` pairs with `align_blocks: [2, 3, 4]`).
- Add overrides in YAML and rely on `deep_update` to only replace intended keys.

## Adding or Rerunning Experiments
For config-only ablations, create a new YAML and a template-based shell wrapper while keeping `model_name` unchanged. Add config flags with defaults that preserve existing behavior, and create a run-time wrapper through `scripts/_run_times/new_run.sh`. If the user asks for only one stage (for example Python only or YAML only), complete that stage without creating later artifacts or launching anything.

For new model variants:

1. Add the model under `models/models_ProMoE_TC_<variant>.py` or, for Expert-Choice family work, `models/models_ProMoE_EC_<variant>.py`.
2. Register the `model_name` in the appropriate `model_dict` in `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`; `sample.py` merges all registries automatically.
3. Add `configs/004_ProMoE_<size>_<variant>.yaml`, keeping `model_name`, config filename, and wrapper name aligned.
4. Create a `scripts/<family>/run_<size>_<variant>_train_sample_eval.sh` wrapper from `scripts/template.sh`.
5. Validate with `python -m py_compile <modified_python_files>` and `python scripts/check_output_dir.py --config <config>`.
6. Preview and create the launch wrapper with `scripts/_run_times/new_run.sh --script <wrapper> --gpus N`, first with `--dry-run`.
7. Generate the adjacent `*-describe.txt` and update the date directory's `commands.md`.

Before rerunning an experiment after model-code changes, do not reuse the old config name if an output directory already exists. Use `python scripts/check_output_dir.py --suggest-version <config>` and move the config, semantic script, and run-time wrapper together to a fresh `_vN` name so the new run writes to a clean `outputs/<model_name>/<custom_cfg_name>/` directory.

### Codex Experiment Workflow Skills

Use the matching `.agents/skills/<name>/SKILL.md` for the four workflows below. They are one-shot experiment workflows rather than review loops. Preserve unrelated staged, unstaged, and untracked work. For Codex operation details, the `.agents/skills/` definition takes precedence over the analogous Claude slash-command text; for repository facts and experiment claims, use this `AGENTS.md` first, executable code/config next, and `CLAUDE.md` only as supplemental documentation. These workflows never launch training, sampling, evaluation, preprocessing, downloads, or GPU jobs; never modify runtime data under `outputs/`, `pretrained_ckpt/`, `training_logs/`, smoke/TensorBoard directories, or uppercase `REPA/`; and never stage, commit, push, force-push, amend, reset, or stash.

#### `new-experiment`

- Use for a new model variant, config-only ablation, experiment config, or all-in-one train + sample + eval wrapper. Distinguish variants that need model code and registry changes from ablations already supported by existing runtime code.
- Inventory compatible partial work before editing. Reuse a valid partial chain, create only missing artifacts, and make repeated invocations idempotent; never overwrite incompatible files or create a second launcher for the same semantic script and output bucket.
- Preserve template structure and validate consistency across the model/registry, numeric-prefixed YAML, semantic wrapper, and dated run-time wrapper. Run syntax checks plus `python scripts/check_output_dir.py --config <config>` before allocation.
- Allocate only when no equivalent run-time wrapper exists. Preview `scripts/_run_times/new_run.sh` with `--dry-run`, inspect the date, GPU count, placeholder `gpu_ids`, script, config, and wrapper, then rerun the same command without `--dry-run`. Any positive GPU count is supported; actual IDs are claimed at launch.
- After the target wrapper validates, compose `describe-experiment` for its adjacent description and maintain `commands.md` in the required table-only format. Report an existing `commands.csv` as stale if affected; regenerate that CSV only when `command-table` was also requested.

#### `rerun-experiment`

- Use after model/config semantic changes, or when an unchanged `model_name` and config basename would collide with or resume an old output. Resolve the complete config, semantic-script, dated-wrapper, and description set before editing.
- Enumerate existing version chains and use `python scripts/check_output_dir.py --suggest-version <config>` to choose a coordinated fresh `_vN` basename unless the user supplied one. Missing or stale descriptions are generated-sidecar repairs and never justify advancing `_vN`.
- Rename text files with `apply_patch` add/delete operations, not `git mv`, so the Git index is untouched. Update exact `CONFIG=`, `LOG=`, and `exec bash` references in lock-step and preserve executable modes.
- Keep the existing GPU-count header, claim logic, and YAML `gpu_ids` unchanged (including fixed assignments on legacy wrappers), never call `new_run.sh`, and never alter or delete the old output directory. Regenerate each affected description and `commands.md` after validation; report an existing CSV as stale unless its regeneration was requested.
- A missing run-time wrapper requires a handoff to `new-experiment`. Requests for a parallel seed/independent copy, or to keep both old and new launch definitions, likewise use `new-experiment` to create a distinct output bucket and launch wrapper.

#### `command-table`

- For one dated batch, trace each launch wrapper through its semantic script and YAML config, and write or replace only `scripts/_run_times/<date>/commands.csv`.
- Read `command-tables/command-table-template.csv` and preserve the exact columns `实验描述,git分支,启动命令,输出位置`. Use plain-text cells and RFC 4180 quoting.
- Sort current wrappers by filename and legacy slots naturally, including historical `.3`/`.4` slots. Use the current Git branch unless the user specifies a per-experiment branch, and derive output as `{output_root}/{model_name}/{config_basename}/`; a top-level `output_dir` changes only `output_root`.
- Report unresolved rows. This CSV-only workflow leaves the required `commands.md` intact; report any actual mismatch, but do not call it obsolete or suggest deleting it. Wrapper-changing workflows maintain that Markdown table automatically.

#### `describe-experiment`

- Trace each dated wrapper through the semantic script, config, registry, model implementation, and relevant project documentation. Ground every claim in inspected code, config, or documentation rather than inferring behavior from filenames.
- Write Chinese-first bilingual plain text beside the wrapper as `<wrapper-basename>-describe.txt`, with two to four numbered changes in importance order.
- Compare the experiment with both the base ProMoE TC/EC design and its immediate inheritance parent or meaningful sibling/default flag setting. Claim step-0 equivalence, parameter changes, checkpoint-loading behavior, or train-fresh requirements only when the implementation or documentation proves them.
- Descriptions are generated sidecars: write only the adjacent `*-describe.txt`, and regenerate a missing or stale description in place without changing the experiment suffix or GPU allocation.

## Testing Guidelines
No dedicated `tests/` directory. Use smoke checks aligned to your change surface:

- Training changes: `python train.py --config ...`, `python train_with_repa.py --config ...`, `python train_with_MoS_repa.py --config ...`, or `python train_with_mae.py --config ...`.
- Sampling changes: `python sample.py --config ...` (with `--step_list_for_sample` / `--guide_scale_list` as needed).
- Evaluation changes: run from inside `evaluation/`: `python run_eval.py ...`.
- Analysis changes: run the relevant entry script under `analyses/`; for FLOPs/statistics, use `python analyses/run_compute_flops.py --help` for import/CLI validation or a checkpoint-backed smoke run with `--ckpt ...`.
- Syntax checks: `python -m py_compile <modified_python_files>`.
- End-to-end REPA-DYNA smoke check: run one wrapper in `scripts/dynamic_repa/` and verify train/sample/eval logs are produced.
- End-to-end MoS-REPA smoke check: run `bash scripts/MoS_repa/run_B_repa_mos_train_sample_eval.sh` and verify train/sample/eval logs are produced.
- Cross-alignment / monitoring changes: run `bash tb_smoke_200/run_all.sh` or `bash tb_smoke_500/run_all.sh` and verify `monitor/` TensorBoard scalars are emitted.
- One-click experimental wrappers take their interpreters the way `scripts/template.sh` does: `source scripts/_python_env.sh`, which pins `/mnt/workspace/yujie/.conda/envs/promoe/bin/python` for training/sampling and `/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python` for evaluation and fails before any output bucket is touched when they are missing. Do not re-hardcode the paths.
- When writing a new training + sampling + evaluation three-in-one shell script, start from `scripts/template.sh`, preserve its interpreter-launch pattern and its `PROMOE_GPU_IDS_OVERRIDE` / `PROMOE_RESUME` handling, swap the training entrypoint as needed (`train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`), and do not replace it with `conda activate`; otherwise the script may fail on the experiment server.
- For new analysis entrypoints, add a matching `analyses/<basename>.md` usage guide and keep shared logic in a subpackage under `analyses/` rather than embedding everything in the root script.

If you touch dataset traversal, latent mapping, or preprocessing logic, clear/regenerate `preprocess/image_paths_cache.txt` before re-running checks.

## Workflow Rules
- Clean up smoke-test artifacts immediately after a smoke or sanity run finishes, whether it succeeds or fails. Remove temporary configs, temporary scripts, and smoke-only output directories such as `outputs/<model>/<smoke_cfg>/`; do not delete long-lived real outputs, `pretrained_ckpt/`, `training_logs/`, or project-level caches unless explicitly requested.
- Run long-lived background processes in a new tmux window of the current session. Do not use `command &`, `nohup`, or detached shell backgrounding for training, sampling, long evals, watch loops, or dev servers.
- Before launching a long-lived tmux command, require an attached tmux session:

```bash
test -n "${TMUX:-}" || { echo "not inside tmux - attach first"; exit 1; }
tmux new-window -t "$(tmux display-message -p '#S')" -n <name> '<command>'
```

- If `$TMUX` is unset, abort and ask the user to attach to tmux first. Short synchronous commands such as `ls`, `rg`, `py_compile`, and `git status` should still run in the foreground.
- Treat the vendored `REPA/` uppercase subproject as separate scope. Do not edit it unless the user explicitly asks for that subproject.

## Commit & Pull Request Guidelines
Use concise, imperative, single-scope commit subjects. In PR descriptions, include:

- Affected model family/config(s).
- Whether the path is `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, or `train_with_mae.py`.
- Dataset layout assumptions (especially `train/` path and latent sibling path).
- GPU assumptions (`gpu_ids`, world size, sampling GPUs).
- Evidence of behavior change (logs, sample folders, metric outputs).

When the user asks to push, treat that as permission to commit relevant current WIP first, then push. Still stage explicit paths only; do not use `git add -A` or `git add .`, do not stage secrets or `*.local.json`, do not bypass hooks with `--no-verify`, and do not force-push to `main` or `master`.

## Configuration Notes
- Set `cfg.data_path` (or the `PROMOE_DATA_PATH` env var, or a YAML override) to the ImageNet `train/` root before preprocess/train; it defaults to the shared `/lustre01/yujie/dataset/imagenet/train`. This absolute path is `train`-once-safe — no component but the `train/` dir contains the substring 'train' — which train.py's latent derivation via `str.replace('train', ...)` requires (keep any override the same way).
- On a fresh server, run `preprocess/prepare_imagenet.sh` **once manually** before launching a batch (it is intentionally NOT auto-invoked by the run scripts — co-scheduled jobs would otherwise both trigger download/encode, and the later job would idle-block on the `flock`). It auto-downloads full-resolution ImageNet-1K (HuggingFace→ModelScope), materialises it to `/lustre01/yujie/dataset/imagenet/train/<label:04d>/`, provisions the VAE, and VAE-encodes everything — idempotent, resume-safe, `flock`-locked as a safety net, and it verifies every image has a latent before finishing. Training then reads `/lustre01/yujie/dataset/imagenet/train` via `config.py`'s default. The data lives outside the repo (nothing to commit).
- Parquet-direct mode: if raw HF parquet shards already exist (default `/lustre01/qianyuan/data/ILSVRC/imagenet-1k/data`, override `PROMOE_PARQUET_DIR` / `--parquet-dir`), `prepare_imagenet.py` auto-detects them and encodes latents DIRECTLY from parquet via `preprocess/encode_latents_from_parquet.py` (shard-parallel, per-file resume), skipping re-download and the intermediate JPEG folder. Output `<latent_root>/<label:04d>/*.latent.npz` uses the same 8-channel `.latent_dist.parameters` format as `preprocess_vae.py`. Training reads it via the `LatentFolder` dataset when a config sets `use_encoded_latents: True`; numeric directories use `int(<label:04d>)`, while standard ImageNet synset directories use their sorted `ImageFolder` index. Neither layout needs an image folder or `replace('train',...)`; the latent path comes from `cfg.latent_data_path` / `PROMOE_LATENT_PATH`. The 2026_07_01 non-REPA configs set the flag; REPA still needs the JPEG path. `preprocess/latent_paths_cache.txt` is the LatentFolder equivalent of the image cache (prepare rebuilds it sorted).
- `custom_cfg_name` is auto-injected from `--config` filename stem and used in output path construction.
- Training uses `gpu_ids` from YAML to set `CUDA_VISIBLE_DEVICES` when provided.
- `total_train_batch_size` is global; `train.py` derives `train_batch_size = total_train_batch_size // world_size`. Changing a run from four to two GPUs therefore changes the per-GPU batch but not the configured global batch (verify divisibility before launching).
- Sampling uses `sample_gpu_ids` only if provided; otherwise it uses all visible GPUs.
- Clean-route KL scheduling uses optional `align_start_step` and `align_warmup_steps`
  under `DiT_*_config.clean_route_config`, both nonnegative integers defaulting to
  zero (the original fixed coefficient). Before the start the KL weight is zero;
  with zero warmup it reaches `align_lam` at the start, otherwise it rises linearly
  to `align_lam` at start + warmup. `train.py` supplies the absolute optimizer
  `training_step`, shared by gradient-accumulation micro-batches and restored on
  resume. Scheduled training requires that argument; eval ignores the schedule.
  This changes only the KL weight, leaves FM/student contrastive active, and adds
  no checkpoint parameters or buffers. Teacher passes still run for diagnostics
  while the weight is zero; logs/TensorBoard include `clean_route_align_weight`.
  Both clean-route model classes support the fields. The 2026-10-30 standalone
  `student_ratio0_align_delay100k` and `student_ratio0_align_warmup200k` configs
  use (start, warmup) = (100000, 0) and (0, 200000), respectively, with final weight
  5e-4. Each trains fresh to 300K, evaluates, resumes to 500K, then evaluates
  again on four GPUs, with CFG 1.0/1.5 and 50K samples per CFG at both steps;
  the original fixed-weight experiment is the control.
- Correction-router training supports optional `DiT_*_config.corr_router_config.previous_state_mode`:
  `model_velocity` (default) keeps the original two auxiliary no-grad forwards;
  `same_noise` constructs the previous state as `(1 - sigma_ref) * z + sigma_ref * noise`
  using the current training sample's exact clean latent and noise. This deliberately
  reintroduces the leak the default avoids: two same-noise points on the straight
  conditional path determine its clean end. Only one auxiliary
  forward remains, to predict velocity at that state and extrapolate the clean reference.
  Gap/no-reference draws, stop-gradient, losses and sampling are unchanged. `train.py`
  supplies `corr_router_z` / `corr_router_noise` to the plain model; the clean-route
  combination reuses `clean_route_z` / `clean_route_noise`, with its teacher unchanged.
- User rule (2026-10-30): all future correction-router experiment definitions
  must explicitly set top-level `corr_reference_velocity: cfg` for sampling and
  evaluation. This includes plain corr_router, clean-route combinations,
  same-noise variants, continuation scripts, and every evaluated CFG scale.
  Generate the next step's routing reference with the CFG-fused velocity.
  Keep any `--corr-reference-velocity` CLI override consistent with `cfg`.
  New evaluation definitions for existing weights use a separate CFG-reference
  output identity so historical conditional-reference results stay distinguishable.
- Correction-router inference supports `corr_reference_velocity: conditional` (legacy fallback)
  or `cfg`, also selectable with `sample.py --corr-reference-velocity`. The latter
  uses the CFG-fused velocity at the pre-update latent to form the next step's
  reference, and by default adds `_corr_ref_cfg` to sample directory names. CFG 1.0 is
  equivalent in both modes. `sample_checkpoint_dir` lets an inference config read
  an original checkpoint into a separate output bucket; it requires explicit
  steps and exact EMA state-dict loading. The inference-only wrappers under
  `scripts/_run_times/2026_10_30/` use the two 2026-10-22 correction models' own
  300K/500K checkpoints on the training server, four GPUs, and CFG 1.0/1.5 with
  50K samples per case. The two unified `*_cfg_ref.sh` launchers replace the old
  `*_cfg_ref_300k.sh` definitions, with fresh `*_cfg_ref` output buckets; do not
  create overlapping per-step launchers. They do not call training.
  `PROMOE_RESUME` defaults to `0` and requires an empty output bucket, so all four
  cases per model are resampled, including CFG 1.0. Explicit `PROMOE_RESUME=1`
  skips valid FID/IS only within this reevaluation. Each pending case uses a
  separate sampler process, after checking all pending checkpoints.
  Original conditional-reference and historical `*_cfg_ref_300k` results are
  neither copied nor migrated. Required weights stay in the original training buckets.
  `corr_reference_separate_output: True` instead uses standard internal sample
  names for both modes. It requires an explicit `sample_checkpoint_dir` and an
  experiment output directory that does not overlap the training bucket.
  The two unified `*_cfg_ref` configs use this standard layout.
- The two `*_same_noise` pipelines under `scripts/_run_times/2026_10_27/` each
  train once to 300K on four GPUs, then run both reference modes as separate
  evaluation experiments (`*_same_noise_cond_ref` and `*_same_noise_cfg_ref`).
  Each evaluation YAML reads the same training bucket's checkpoint, uses standard
  internal names, and samples CFG 1.0 then 1.5 with 50K images each. No weights are
  copied. The shared `scripts/corr_router/_dual_reference_eval.sh` checks all four
  mode/CFG metrics and resumes the two experiments independently.
- `train_with_repa.py` and `train_with_MoS_repa.py` read REPA behavior from top-level `repa_config`; model-level REPA knobs live under `DiT_*_config.repa_config` in YAML.
- MoS-REPA configs (for example `004_ProMoE_B_repa_MoS.yaml`) additionally set `DiT_*_config.repa_config.num_teacher_blocks`; keep it aligned with the chosen teacher encoder depth.
- Dynamic-select configs (`004_ProMoE_B_repa_dyna_select*.yaml`) control token selection via `DiT_B_config.repa_config.repa_select_ratio`.
- Router configs use model-level REPA knobs such as `router_repa_coeff` (`004_ProMoE_B_repa_router.yaml`) or `router_loss_decay_steps` (`004_ProMoE_B_repa_router_contra.yaml`).
- `004_ProMoE_B_repa_dyna_only.yaml` also carries a model-level `DiT_B_config.repa_config.proj_coeff` for the capped dynamic-weight ablation.
- Cross-alignment configs use model names such as `ProMoE_TC_REPA_CROSS_GLOBAL_PRE_B`, `ProMoE_TC_REPA_CROSS_GLOBAL_BLOCK_B`, `ProMoE_TC_REPA_CROSS_EXPERT_LOCAL_B`, `ProMoE_TC_REPA_CROSS_PROTO_B`, and their `ProMoE_TC_REPA_MoS_*` counterparts.
- MoS-REPA block-range configs may vary `align_blocks`, projector sharing, router norm, fused routing, dynamic coefficients, or `proj_coeff`; keep wrapper names and YAML values aligned.
- `proto_t` configs choose `proto_t_update_mode` (`direct` or `residual`) under `DiT_B_config`; the `EC_BC` proto-t configs use the expert-choice batch-choice model registry key.
- Anchor configs set `anchor_apply_mode` (`routing` or `replace`) under `DiT_B_config`.
- Proto-choice ratio configs set `contrastive_proto_choice_ratio`; wrapper/config suffixes such as `083` and `125` refer to the ratio sweep values.
- Load-balance-contrastive (lbcontra) configs set `lb_contra_mode` (`reweight`/`logit_adjust`/`balance_term`/`soft_only`) and the mode's scalar (`lb_reweight_beta` / `lb_logit_adj_tau` / `lb_balance_lambda`); only the routing-contrastive loss changes, so step-0-identical to base ProMoE.
- DAG-Fuse (dagfuse) configs set `fusion_arm` (`cond_from_shared`/`shared_from_cond`/`bidirectional`) and `fusion_num_iter`; the fusion module's up_proj is zero-init, so step-0-identical (non-strict checkpoint load).
- LS-Reg configs (`004_ProMoE_B_lsreg_*.yaml`) keep `model_name: ProMoE_TC_B_lsreg` and change only the routing-contrastive objective. `ls_apply: label` supports `ls_mode: fixed`, `dyn_both`, `dyn_under`, or `dyn_over` (with `ls_eps_base`, `ls_slope`, `ls_eps_cap`, `ls_load_map`, and optional `ls_warmup`); `ls_apply: diag` applies a detached load-proportional similarity-diagonal offset controlled by `ls_diag_strength` and `ls_diag_sign` (`+1` idea-1, `-1` inverse). The variant has no new parameters and is step-0-identical to base ProMoE; rank-0 logs realized `lsreg/mean_eps`.
- Shared-expert DAG-Fuse configs (`004_ProMoE_B_dagfuse_{dense,densenet,sharedroute,region}_*.yaml`) use `fuse_apply: none|cond|all`, `fuse_mech: dag|softmax`, and `fuse_dim` (default 64). `sharedroute` adds `fuse_top_k: 1|2`; `region` adds `region_size` (default 3) and `region_attach: shared|resid`. The augmentation is zero-initialized, so these variants are step-0-identical and can load base checkpoints non-strictly. The four sources differ in whether they consume prior dense output, prior shared outputs, router-selected shared outputs, or fixed-size region representations.
- Expert-Contra parameter ablations retain `model_name: ProMoE_TC_B_expert_contra`. `expert_contrastive_mode` is `output`, `param` (L2 RBF repulsion), or `param_cos` (magnitude-invariant cosine repulsion); parameter variants can set `expert_contrastive_include_bias`, `expert_contrastive_include_shared`, and `expert_contrastive_include_uncond`. Keep `expert_contrastive_blocks` restricted to MoE blocks. The v2 configs (`nobias`, `cos`, `shared`, `shared_uncond`, `tau0p07`, `tau7`) are one-factor-at-a-time ablations and remain backward-compatible with the baseline defaults.
- Adaptive-depth (adepth) configs set `alloc_mode=fixed_q`, `depth_q`, and `depth_warmup`; requires `top_k==1`, and zero-init gates keep it step-0-identical (non-strict).
- Image-wise shared-expert configs use `ProMoE_TC_B_image_shared` with `num_image_experts` (default 3), `image_expert_backend` (`bmm` by default, or `loop`), `image_pool_num_heads` (default 1), `image_routing_contrastive_lam`, and `image_routing_contrastive_temperature` under `DiT_B_config.MoE_config`. Only the shared branch changes: conditional and unconditional images both use the same learned-query pooling and select one image expert independently by default. Token-wise routing stays inherited from base TC; train this variant fresh.
- `ProMoE_TC_B_shared_plus_cond_image` retains the fixed shared expert and adds the same image pool only for images whose post-dropout label is conditional. Its original `e4` B config keeps shared/unconditional widths at 1536 (dense FFN width / 2), and conditional token/image widths at 768 (dense width / 4), via `moe_intermediate_size`, `shared_expert_intermediate_size`, `uncond_expert_intermediate_size`, and `image_expert_intermediate_size`. The `e4_ffn_thirds` config reuses the same model with shared/conditional-token/conditional-image widths of 1024 (dense width / 3) and an unconditional width of 2048 (2 * dense width / 3). Both schemes match FFN matrix work on conditional and unconditional branches; pooling/router overhead is extra. The token router and attention-pooling design are inherited.
- The two image-expert launch configs in `scripts/_run_times/2026_10_21/` explicitly use `num_image_experts: 4` and the `e4` filename suffix. The configurable model constructors retain their default of 3; use the launch YAML to reproduce the selected experiment.
- The additive image model accepts integer token `top_k` from 1 through
  `num_routed_experts`; image routing uses `image_top_k` (default 1). The `e4_token_top2` config
  uses 12 conditional token experts, four conditional image experts, and widths
  of 1536 for token/image/shared/unconditional FFNs. Conditional tokens activate
  two token experts, one image expert, and shared (width sum 6144), matching
  base token Top-3 plus shared in FFN matrix work. Unconditional tokens still
  execute only shared and the dedicated unconditional expert (width sum 3072).
  Image pooling/router overhead and total parameter count are additional; this
  is not an equality of complete-model FLOPs or total parameters.
- Both image expert pools support integer `image_top_k` in `[1, num_image_experts]`.
  Each image shares its selected experts across all tokens; raw selected cosine
  scores weight their summed outputs. Image contrastive guidance includes every
  selected assignment. Default Top-1 retains its original max/tie behavior.
  The `e4_image_top2` config is the counterpart of `e4_token_top2`: token Top-1
  plus image Top-2, with the same 12 token experts, four image experts, widths 1536,
  367,743,008 trainable parameters and conditional/unconditional FFN width sums
  6144/3072. It retains shared and bypasses the image branch for unconditional images.
- Loss-free (lossfree) configs set `use_lossfree_bias` and `bias_update_rate` (`u`); the per-prototype bias is a non-trainable buffer added only to top-1 selection, step-0-identical (non-strict).
- Most provided YAMLs set `resume_checkpoint: True`; when no checkpoint exists the loader logs an error and training starts from step 0.
- `sample.py` behavior:
  - if `step_list_for_sample` is set, it loads only those checkpoints;
  - otherwise it scans `checkpoints/` and loads steps divisible by `sample_every_step`;
  - `--num_fid_samples` also updates `save_img_num`.

Model registry and forward conventions:

- `train.py`, `train_with_repa.py`, `train_with_MoS_repa.py`, and `train_with_mae.py` each define a `model_dict` from `model_name` to `(ModelClass, config_key)`. Add new registrations in the training script that owns the family.
- The current `train.py` registrations include teacher-free Base-FDRR as `ProMoE_TC_B_FDRR` (`models/models_ProMoE_TC_denoising_regret.py`), `ProMoE_TC_B_lsreg` (`models/models_ProMoE_TC_lsreg.py`), and the shared-expert DAG-Fuse keys `ProMoE_TC_B_dagfuse_{dense,densenet,sharedroute,region}` (`models/models_ProMoE_TC_dagfuse_*.py`). The LS-Reg files are config-only sweeps within that registered class; each shared-expert source has its own registered class.
- `train.py` hosts base DiT/baselines, ProMoE TC/EC, EC batch-choice, proto-t, anchor, proto-choice, lbcontra (load-balance-aware routing contrastive), dagfuse (DAG-MoE shared↔cond fusion), dagfuse_shared (shared-expert augmentation), adepth (adaptive routed-FFN depth), lossfree (loss-free balancing bias), teacher-free Base-FDRR, lsreg (routing-contrastive label smoothing), structured-batch, noise-expert, and expert-contrastive families.
- `train_with_repa.py` hosts standard REPA, REPA shared/cond, dynamic REPA, router/routed/double-share REPA, and heterogeneous-expert REPA-DYNA families.
- `train_with_MoS_repa.py` hosts MoS, naive/choice/separate/blockwise/per-block/fused variants, multi-align, teacher-affinity multi-align, shared-routed spectral-responsibility multi-align, teacher-conditioned expert-geometry multi-align, first-order denoising-regret routing (`ProMoE_TC_REPA_Multi_Align_FDRR_B`), and standard REPA + MoS cross-alignment families.
- `train_with_mae.py` hosts `group_align` and `group_align_proj` families.
- Plain DiT and most ProMoE variants return a tensor. Teacher-free Base-FDRR returns `(prediction, denoising_regret_loss)` only during training and a tensor during eval; `train.py` owns this contract. REPA variants return `(pred, zs_proj)` during training. MoS-REPA, multi-align, fused MoS, and cross-alignment variants return `(pred, alignment_loss)` during training. Teacher-affinity, spectral-responsibility, teacher-conditioned expert-geometry, and first-order denoising-regret multi-align return `(pred, alignment_loss, auxiliary_loss)`; `train_with_MoS_repa.py` dispatches the third value by registered family and weights it with `teacher_affinity_coeff`, `spectral_responsibility_coeff`, `expert_geometry_coeff`, or `denoising_regret_coeff`, respectively.
- ProMoE routing contrastive loss generally flows through the `AddAuxiliaryLoss` autograd wrapper even when `forward()` returns a plain tensor.

REPA config scope:

- Top-level `repa_config` belongs to the training loop and controls teacher loading, such as `enc_type`, global alignment weighting such as `proj_coeff`, and registered third-loss coefficients including `denoising_regret_coeff`.
- Nested `DiT_*_config.repa_config` belongs to the model and controls projectors, `encoder_depth`, `z_dims`, `align_blocks`, `num_teacher_blocks`, router REPA knobs, dynamic/select behavior, and FDRR probe/label settings.
- Keep `enc_type` and teacher block depth consistent across both scopes for REPA and MoS-REPA configs.

Cross-alignment stability notes:

- In cross-alignment loss code, clamp cosine-similarity matrices to `[-1, 1]` after normalization and `torch.bmm`; bf16 precision can otherwise create values slightly outside the valid range and trigger loss spikes.
- For block-wise cross-alignment weight predictors (`cross_global_block` and `cross_expert_local`, including MoS counterparts), feed detached block outputs into the attention/weight module while keeping the projection path on the original tensor. This prevents competing gradient paths into the same DiT block.
- `cross_global_pre` is exempt because attention is applied before DiT blocks; `cross_proto` is exempt because weights come from MoE routing rather than a dedicated weight predictor.
- Use `TrainingMonitor` from `utils.py` for cross-alignment crash diagnosis. Wire it after `backward()` and gradient clipping, before `zero_grad()`, and pass the existing TensorBoard writer when available so `monitor/*` scalars are emitted.

Variant-specific notes:

- `anchor_apply_mode` is `routing` or `replace`; anchor variants are not step-0-identical to base ProMoE because anchors are randomly initialized, so train them fresh.
- `contrastive_proto_choice_ratio` controls proto-choice positive-set size; suffixes such as `083` and `125` map to ratios such as `0.083` and `0.125`.
- `proto_t_update_mode` supports `direct` and `residual`; script/config names remain human-readable and 1-indexed while YAML `align_blocks` remains 0-indexed.
- `noise_expert_ema` parameters are frozen and updated through EMA after optimizer steps; they should stay excluded from the optimizer.
- FDRR model-level fields are `denoising_regret_block`, `denoising_regret_probe_interval`, `denoising_regret_token_ratio`, `denoising_regret_candidate_mode`, `denoising_regret_confidence_quantile`, `denoising_regret_temperature`, `denoising_regret_warmup_steps`, `denoising_regret_ramp_steps`, `denoising_regret_label_roll`, `denoising_regret_seed`, and `denoising_regret_eps`. Teacher-free Base-FDRR places them under `DiT_B_config.denoising_regret_config`; Multi-Align FDRR keeps them under nested `DiT_B_config.repa_config`. The positive arm uses label roll `0`; the archived REPA matched control uses `1`.
- Both FDRR implementations run the inner diffusion-MSE `autograd.grad` under `suppress_auxiliary_loss_backward()` so routing-contrastive `AddAuxiliaryLoss` nodes do not enter the utility-label query; ordinary outer backward must retain the auxiliary injection. The probe-v4 `fdrr_gate_v1` checkpoint gate belongs to the archived Multi-Align FDRR experiment and requires a Base Multi-Align checkpoint step >=10K. Teacher-free Base-FDRR instead requires its Base-equivalence tests, a same-seed fresh `ProMoE_TC_B` control, and the separate Base-shape <=10% overhead/four-GPU memory gate before launch.

Latent mode and cache behavior:

- With `use_pre_latents=True`, both training and preprocessing rely on `preprocess/image_paths_cache.txt`. `prepare_imagenet.py` rebuilds this cache deterministically (sorted, atomic write) before preprocessing so DDP ranks don't race to regenerate it; `preprocess_vae.py` also skips images whose `.latent.npz` already exists and writes latents atomically.
- Latent path rule in training is string-based: image path replaces `train` with `sd-vae-ft-mse_Latents_256img_npz`, extension becomes `.latent.npz`.
- Keep dataset naming aligned with this replacement rule, or update the code.
- In REPA training with `use_pre_latents=True`, the dataset additionally loads raw images for teacher feature extraction.
- In MoS-REPA training, teacher features are extracted from all teacher blocks and aligned block-wise against every DiT block.

Weight caching:

- VAE auto-cache path: `pretrained_ckpt/vae/<hf_repo_id_with_slash_replaced>/` (unless `--vae-path` is passed).
- REPA teacher cache path: `pretrained_ckpt/encoder/<hub_name>/state_dict.pth` (unless `--repa-enc-path` is passed).
- For REPA and MoS-REPA, rank 0 performs initial teacher download/cache, then other ranks load from local cache after barrier.

Analysis notes:

- `analyses/run_compute_flops.py` resolves the YAML config from the checkpoint path, reads `gpu_ids` from that YAML, and sets `CUDA_VISIBLE_DEVICES` before spawning workers.
- The FLOPs analysis entrypoint accepts both `--ckpt` and the legacy positional checkpoint argument; hyphenated and underscore flag spellings are both supported for its sampling/reporting options.
- FLOPs instrumentation temporarily switches image-wise shared experts to `loop` and restores the previous backend afterward. It counts pooling and image-routing matrix products; activated-parameter tracking counts only selected image experts. See `analyses/run_compute_flops.md` for the counting convention.
- `analyses/run_image_expert_benchmark.py` compares image-expert backends with full DDP training steps on fixed synthetic latent batches; it records time, throughput, peak memory, and source hashes under `outputs/benchmarks/image_experts/`. GPU benchmarks require explicit user authorization and an attached tmux window, as for other long-running GPU jobs.
- FLOPs/statistics outputs are written under `outputs/<model_name>/<config_stem>/sample/step<ckpt_step>/flops_eval/`, including `flops_result.txt`, expert-frequency plots, and optional per-step reports.
- `analyses/run_mos_routing_analysis.py` infers the YAML config from `--ckpt`, does not require a teacher encoder, and writes plots plus `metadata.yaml` under `outputs/<model_name>/<config_stem>/sample/step<ckpt_step>/mos_routing/`.

Evaluation notes:

- `evaluation/run_eval.py` calls `ensure_ref_batches()` before evaluation, but skips reference downloads with `--no-eval`.
- It accepts sampling's rounded-up output count, requires contiguous unique names `img<index>_class<label>.png` from index zero, and packs exactly the requested `--count`; a missing image, malformed index, or invalid ImageNet label is a hard error.
- It writes `<image_folder>.npz`, and when evaluation runs it also writes `<image_folder>_eval_openai.txt`.
- Evaluator subprocess failures are logged and propagated as nonzero exits. `run_eval.py` resolves `evaluator.py` relative to its own file, so it may be launched from the repository root or `evaluation/`.
