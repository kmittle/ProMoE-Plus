#!/bin/bash
#
# Same-noise correction-router ablation, based on scripts/template.sh.
# Train once; evaluate pre-CFG (conditional) and post-CFG (cfg) reference velocities.
# Each mode evaluates CFG 1.0/1.5 into a separate experiment directory with standard internal names.
#
# Pipeline (sequential, safe for XL-scale models):
#   For each checkpoint step in step_list_for_sample:
#     1. Train (or resume) up to that step, then stop.
#     2. Sample + eval using that checkpoint (GPUs fully free).
#   This avoids concurrent training + sampling, which may exceed GPU memory
#   for XL-scale models even on A100 80GB.
#
# GPUs: the YAML gpu_ids fixes the DDP world size.  The run-time wrapper
# (scripts/_run_times/new_run.sh) claims idle GPUs at launch and passes them in
# PROMOE_GPU_IDS_OVERRIDE, which replaces gpu_ids for training, sampling and
# evaluation and must name the same number of GPUs.  Run the script directly
# and it uses the YAML gpu_ids.
#
# Resume: a launch trains from step 0 and requires an empty output bucket.  Set
# PROMOE_RESUME=1 to continue an interrupted run from its own checkpoints; a
# phase whose checkpoint already exists is not trained again, and a step whose
# every CFG already has a valid evaluation is not sampled again.
#
# Prerequisites:
#   - conda envs: promoe (train/sample), fid_eval (evaluation), pinned in
#     scripts/_python_env.sh
#   - A100 80GB or equivalent
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
source "${REPO_ROOT}/scripts/_python_env.sh"
source "${REPO_ROOT}/scripts/_eval_metric_helpers.sh"

CONFIG="configs/004_ProMoE_B_clean_route_student_ratio0_corr_router_same_noise.yaml"
LOG="${REPO_ROOT}/logs/log_B_clean_route_student_ratio0_corr_router_same_noise_train_sample_eval.log"
mkdir -p "$(dirname "$LOG")"
exec 9>"${LOG}.lock"
flock -n 9 || { echo "ERROR: this experiment is already running" >&2; exit 1; }

# Deployment default; also overwrite the temporary YAML so an older config's
# latent_data_path cannot silently override this environment setting.
export PROMOE_LATENT_PATH="${PROMOE_LATENT_PATH:-/mnt/data2/yujie/code/Dataset/ImageNet/sd-vae-ft-mse_Latents_256img_npz}"

RESUME="${PROMOE_RESUME:-0}"
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "ERROR: PROMOE_RESUME must be 0 or 1, got: ${RESUME}" >&2
    exit 1
fi

PYTHON="${PROMOE_TRAIN_PYTHON}"
PYTHON_EVAL="${PROMOE_EVAL_PYTHON}"
if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} cannot import PyYAML; install it in the promoe environment" >&2
    exit 1
fi

readarray -t YAML_INFO < <("$PYTHON" - "$CONFIG" <<'PY'
import os
from pathlib import Path
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

if cfg["DiT_B_config"]["corr_router_config"].get("previous_state_mode") != "same_noise":
    raise ValueError("This ablation requires previous_state_mode: same_noise")
if cfg.get("guide_scale_list") != [1.0, 1.5] or cfg.get("step_list_for_sample") != [300000]:
    raise ValueError("This pipeline requires the 300K checkpoint and CFG 1.0/1.5")
if cfg.get("save_inception_features", False):
    raise ValueError("The evaluator requires PNG images")
if cfg.get("sample_checkpoint_dir") or cfg.get("sample_gpu_ids") is not None:
    raise ValueError("Use this run's own checkpoint and the claimed GPUs for both reference modes")

# The two sampling configs may differ only by reference mode and output identity.
training_output = Path(cfg.get("output_dir", "outputs")) / cfg["model_name"] / Path(cfg_path).stem
for suffix, mode in (("cond_ref", "conditional"), ("cfg_ref", "cfg")):
    inference_path = Path(cfg_path).with_stem(Path(cfg_path).stem + "_" + suffix)
    with inference_path.open() as f:
        inference_cfg = yaml.safe_load(f)
    expected = dict(cfg, sample_checkpoint_dir=str(training_output / "checkpoints"),
                    corr_reference_velocity=mode, corr_reference_separate_output=True)
    if inference_cfg != expected:
        raise ValueError(f"Inference settings must match this training run: {inference_path}")

model_name = cfg.get("model_name")
if not model_name:
    raise ValueError(f"model_name not found in {cfg_path}")

num_fid_samples = int(cfg.get("num_fid_samples", 50000))
gpu_ids = cfg.get("gpu_ids", [0])
gpu_str = ','.join(map(str, gpu_ids)) if isinstance(gpu_ids, list) else "0"
eval_gpu = str(gpu_ids[0]) if isinstance(gpu_ids, list) and len(gpu_ids) > 0 else "0"
# GPUs claimed at launch replace the placeholder gpu_ids; the config and its
# output identity stay unchanged.  The count must match, because the world size
# sets the per-GPU batch and every per-rank statistic.
gpu_override = os.environ.get("PROMOE_GPU_IDS_OVERRIDE", "").strip()
if gpu_override:
    override_ids = [item.strip() for item in gpu_override.split(",")]
    if not all(item.isdigit() for item in override_ids) or len(set(override_ids)) != len(override_ids):
        raise ValueError(f"PROMOE_GPU_IDS_OVERRIDE must be distinct GPU ids like 2,3; got {gpu_override!r}")
    if not isinstance(gpu_ids, list) or len(override_ids) != len(gpu_ids):
        raise ValueError(
            f"PROMOE_GPU_IDS_OVERRIDE={gpu_override} names {len(override_ids)} GPU(s) but "
            f"{cfg_path} gpu_ids {gpu_ids} fixes the world size")
    gpu_str = ','.join(override_ids)
    eval_gpu = override_ids[0]
custom_cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
step_list = cfg.get("step_list_for_sample", [])
step_str = ','.join(map(str, step_list)) if step_list else ""
orig_num_steps = int(cfg.get("num_steps", 0))
# sample.py names each CFG directory img..._cfg<scale>_...; empty disables the resume skip.
guide_scales = cfg.get("guide_scale_list") or []
guide_str = ','.join(str(scale) for scale in guide_scales)

print(model_name)
print(custom_cfg_name)
print(num_fid_samples)
print(eval_gpu)
print(gpu_str)
print(step_str)
print(orig_num_steps)
print(guide_str)
print(f"img{cfg['image_size']}")
print(f"_seed{cfg['global_seed']}_FID{int(num_fid_samples / 1000)}K_bs{cfg['sample_batch_size']}_ema")
PY
)
# The process substitution hides a Python failure from set -e; check the result.
if [[ ${#YAML_INFO[@]} -ne 10 ]]; then
    echo "ERROR: could not read ${CONFIG} (see the Python error above)" >&2
    exit 1
fi

MODEL_NAME="${YAML_INFO[0]}"
CUSTOM_CFG_NAME="${YAML_INFO[1]}"
NUM_FID_SAMPLES="${YAML_INFO[2]}"
EVAL_GPU="${YAML_INFO[3]}"
GPU_IDS="${YAML_INFO[4]}"
STEP_LIST_STR="${YAML_INFO[5]}"
ORIG_NUM_STEPS="${YAML_INFO[6]}"
GUIDE_SCALES_STR="${YAML_INFO[7]}"
SAMPLE_FOLDER_PREFIX="${YAML_INFO[8]}"
SAMPLE_FOLDER_SUFFIX="${YAML_INFO[9]}"
SAMPLE_BASE="${REPO_ROOT}/outputs/${MODEL_NAME}/${CUSTOM_CFG_NAME}/sample"
OUTPUT_BASE="${SAMPLE_BASE%/sample}"

# ── Output bucket: a fresh run starts empty; a resume needs its own checkpoints ─
if [[ -L "$OUTPUT_BASE" ]]; then
    echo "ERROR: output bucket must be a real directory, not a symlink: $OUTPUT_BASE" >&2
    exit 1
fi
if [[ "$RESUME" == "1" ]]; then
    if ! find "${OUTPUT_BASE}/checkpoints" -maxdepth 1 -name 'ckpt_step_*.pth' -print -quit 2>/dev/null | grep -q .; then
        echo "ERROR: PROMOE_RESUME=1 but this run has no checkpoint to continue: ${OUTPUT_BASE}/checkpoints" >&2
        exit 1
    fi
elif [[ -e "$OUTPUT_BASE" ]] && find "$OUTPUT_BASE" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: training-from-scratch output bucket is not empty: $OUTPUT_BASE" >&2
    echo "       Set PROMOE_RESUME=1 to continue this run from its own checkpoints." >&2
    exit 1
fi

# ── Parse step_list ──────────────────────────────────────────────────────────
if [ -z "$STEP_LIST_STR" ]; then
    echo "ERROR: step_list_for_sample is empty or missing in ${CONFIG}" >&2
    exit 1
fi
IFS=',' read -ra ALL_STEPS <<< "$STEP_LIST_STR"
NUM_ALL_STEPS=${#ALL_STEPS[@]}

# ── Temp config: same basename as CONFIG so custom_cfg_name is preserved ─────
TEMP_DIR=$(mktemp -d)
TEMP_CONFIG="${TEMP_DIR}/$(basename "$CONFIG")"
trap 'rm -rf "$TEMP_DIR"' EXIT

# Mode-aware evaluation and resume checks for pre-/post-CFG reference velocities.
source "${REPO_ROOT}/scripts/corr_router/_dual_reference_eval.sh"

# ── Helper: record which code this phase runs ────────────────────────────────
# Without strict provenance nothing else records it, and a pull between two
# phases would silently change the second half of the run.  Never fatal.
log_code_version() {
    {
        echo "[$(date '+%H:%M:%S')] Code version (phase $1):"
        echo "  git HEAD: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
        if modified="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null)"; then
            echo "  modified tracked files: $(printf '%s' "$modified" | grep -c .)"
        else
            echo "  modified tracked files: unavailable"
        fi
        sha256sum ./*.py "$CONFIG" "${BASH_SOURCE[0]}" "${REPO_ROOT}/scripts/corr_router/_dual_reference_eval.sh" 2>/dev/null | sed 's/^/  /'
        echo "  models/*.py: $(cat models/*.py 2>/dev/null | sha256sum | cut -c1-64)"
        echo "  repa/*.py: $(cat repa/*.py 2>/dev/null | sha256sum | cut -c1-64)"
    } >> "$LOG" 2>&1 || true
}

# ══════════════════════════════════════════════════════════════════════════════
# Sequential pipeline: train → stop → sample + eval → resume → ...
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$RESUME" == "1" ]]; then
    echo "============================================================" | tee -a "$LOG"
    echo "Resuming sequential pipeline: ${MODEL_NAME}" | tee -a "$LOG"
else
    echo "============================================================" | tee "$LOG"
    echo "Sequential pipeline: ${MODEL_NAME}" | tee -a "$LOG"
fi
echo "Config: ${CONFIG}" | tee -a "$LOG"
echo "Latents: ${PROMOE_LATENT_PATH}" | tee -a "$LOG"
echo "GPUs: ${GPU_IDS} (eval: ${EVAL_GPU})" | tee -a "$LOG"
echo "Steps: ${STEP_LIST_STR}" | tee -a "$LOG"
echo "Reference modes: ${CORR_EVAL_MODES[*]}; CFG scales: ${GUIDE_SCALES_STR}" | tee -a "$LOG"
echo "Evaluation outputs: ${CORR_EVAL_OUTPUTS[*]}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for i in "${!ALL_STEPS[@]}"; do
    step="${ALL_STEPS[$i]}"
    phase=$((i + 1))

    # For intermediate steps: num_steps = step + 1 (train through step, save, exit).
    # For the final step: use original num_steps from config.
    if [ "$phase" -lt "$NUM_ALL_STEPS" ]; then
        TARGET_NUM_STEPS=$((step + 1))
    else
        TARGET_NUM_STEPS="$ORIG_NUM_STEPS"
    fi

    # Under PROMOE_RESUME=1 a phase whose checkpoint is already saved goes
    # straight to sample + eval.  Training it again would redo that step and,
    # with trainers that resume at the saved step, overwrite the checkpoint
    # that is being sampled.
    if [[ "$RESUME" == "1" && -f "${OUTPUT_BASE}/checkpoints/ckpt_step_${step}.pth" ]]; then
        if step_already_evaluated "$step"; then
            echo "Phase ${phase}/${NUM_ALL_STEPS}: step ${step} is trained and evaluated; skipping" | tee -a "$LOG"
        else
            echo "Phase ${phase}/${NUM_ALL_STEPS}: checkpoint for step ${step} exists; skipping training" | tee -a "$LOG"
            log_code_version "$phase"
            sample_and_eval_step "$step"
        fi
        continue
    fi
    log_code_version "$phase"

    # Generate the temp config: adjusted num_steps; gpu_ids replaced by the GPUs
    # claimed at launch, because train.py sets CUDA_VISIBLE_DEVICES and the world
    # size from the config's gpu_ids.  resume_checkpoint stays True in every
    # phase: the bucket check above already guarantees a fresh launch has no
    # checkpoint, and a constant value keeps the strict-provenance config hash
    # the same across phases.
    "$PYTHON" - "$CONFIG" "$TARGET_NUM_STEPS" "$TEMP_CONFIG" "$GPU_IDS" <<'PY'
import os, sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
cfg['num_steps'] = int(sys.argv[2])
cfg['resume_checkpoint'] = True
cfg['gpu_ids'] = [int(item) for item in sys.argv[4].split(',')]
cfg['latent_data_path'] = os.environ['PROMOE_LATENT_PATH']
with open(sys.argv[3], 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PY

    # ── Train ─────────────────────────────────────────────────────────────────
    echo "============================================================" | tee -a "$LOG"
    echo "Phase ${phase}/${NUM_ALL_STEPS}: Train to step ${step} (num_steps=${TARGET_NUM_STEPS})" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "$PYTHON" train.py \
        --config "${TEMP_CONFIG}" \
        >> "$LOG" 2>&1
    TRAIN_RC=$?
    set -e

    if [ $TRAIN_RC -ne 0 ]; then
        echo "Training FAILED at phase ${phase} (exit code $TRAIN_RC)" | tee -a "$LOG"
        exit $TRAIN_RC
    fi
    echo "Phase ${phase} training completed successfully" | tee -a "$LOG"

    # ── Sample + eval ─────────────────────────────────────────────────────────
    echo "============================================================" | tee -a "$LOG"
    echo "Phase ${phase}/${NUM_ALL_STEPS}: Sample+eval step ${step}" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    sample_and_eval_step "$step"
done

echo "============================================================" | tee -a "$LOG"
echo "All done." | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
