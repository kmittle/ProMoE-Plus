#!/bin/bash
#
# ProMoE-TC-B with a cosine regularizer on the pooled expert representations,
# weight lam=0.1 (one arm of the 0.1 / 0.3 / 1.0 / 3.0 sweep).  Trains from
# step 0 to 500K on 2 GPUs (128 images per GPU, global batch still 256).  A
# 2-GPU run is not directly comparable with the 4-GPU canonical baseline, so
# the evidence is the trend across the four lam values.  Samples and evaluates
# at 300K and 500K; the 300K result does not stop the run.  An interrupted run
# continues from its own checkpoints with PROMOE_RESUME=1.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
source "${SCRIPT_DIR}/expert_cos_eval_helpers.sh"
source "${REPO_ROOT}/scripts/_python_env.sh"

CONFIG="configs/004_ProMoE_B_expert_cos_lam0p1.yaml"
LOG="${REPO_ROOT}/logs/log_ProMoE_B_expert_cos_lam0p1_train_sample_eval.log"
mkdir -p "$(dirname "$LOG")"

RESUME="${PROMOE_RESUME:-0}"
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "ERROR: PROMOE_RESUME must be 0 or 1, got: ${RESUME}" >&2
    exit 1
fi

PYTHON="${PROMOE_TRAIN_PYTHON}"
PYTHON_EVAL="${PROMOE_EVAL_PYTHON}"
[[ -x "$PYTHON" ]] || {
    echo "ERROR: required training interpreter is missing: $PYTHON" >&2
    exit 1
}
[[ -x "$PYTHON_EVAL" ]] || {
    echo "ERROR: required evaluation interpreter is missing: $PYTHON_EVAL" >&2
    exit 1
}
if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} cannot import PyYAML; install it in the promoe environment" >&2
    exit 1
fi
readarray -t YAML_INFO < <("$PYTHON" - "$CONFIG" <<'PY'
import os
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

model_name = cfg.get("model_name")
if not model_name:
    raise ValueError(f"model_name not found in {cfg_path}")

num_fid_samples = int(cfg.get("num_fid_samples", 50000))
gpu_ids = cfg.get("gpu_ids", [0])
gpu_str = ','.join(map(str, gpu_ids)) if isinstance(gpu_ids, list) else "0"
eval_gpu = str(gpu_ids[0]) if isinstance(gpu_ids, list) and len(gpu_ids) > 0 else "0"
# Launch-time GPU selection without changing the config or its output identity.
gpu_override = os.environ.get("PROMOE_GPU_IDS_OVERRIDE", "").strip()
if gpu_override:
    gpu_str = gpu_override
    eval_gpu = gpu_override.split(",", 1)[0].strip()
custom_cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
step_list = cfg.get("step_list_for_sample", [])
step_str = ','.join(map(str, step_list)) if step_list else ""
orig_num_steps = int(cfg.get("num_steps", 0))

print(model_name)
print(custom_cfg_name)
print(num_fid_samples)
print(eval_gpu)
print(gpu_str)
print(step_str)
print(orig_num_steps)
PY
)

MODEL_NAME="${YAML_INFO[0]}"
CUSTOM_CFG_NAME="${YAML_INFO[1]}"
NUM_FID_SAMPLES="${YAML_INFO[2]}"
EVAL_GPU="${YAML_INFO[3]}"
GPU_IDS="${YAML_INFO[4]}"
STEP_LIST_STR="${YAML_INFO[5]}"
ORIG_NUM_STEPS="${YAML_INFO[6]}"
SAMPLE_BASE="${REPO_ROOT}/outputs/${MODEL_NAME}/${CUSTOM_CFG_NAME}/sample"
OUTPUT_BASE="${SAMPLE_BASE%/sample}"
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

# ── Helper: sample + eval one checkpoint step ────────────────────────────────
sample_and_eval_step() {
    local step=$1
    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} started" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "$PYTHON" sample.py \
        --config "${TEMP_CONFIG}" --step_list_for_sample "${step}" \
        >> "$LOG" 2>&1
    expert_cos_eval_images "$SAMPLE_BASE" "$step" "$LOG" "$EVAL_GPU" \
        "$PYTHON_EVAL" "$NUM_FID_SAMPLES"
    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} done" | tee -a "$LOG"
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
echo "Steps: ${STEP_LIST_STR}" | tee -a "$LOG"
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

    # A fresh launch starts its first phase from step 0; later phases and an
    # explicit PROMOE_RESUME=1 continue this run's own checkpoints.
    "$PYTHON" - "$CONFIG" "$TARGET_NUM_STEPS" "$TEMP_CONFIG" "$phase" "$RESUME" <<'PY'
import os, sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
cfg['num_steps'] = int(sys.argv[2])
cfg['resume_checkpoint'] = int(sys.argv[4]) > 1 or sys.argv[5] == "1"
gpu_override = os.environ.get("PROMOE_GPU_IDS_OVERRIDE", "").strip()
if gpu_override:
    cfg['gpu_ids'] = [int(item.strip()) for item in gpu_override.split(',') if item.strip()]
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
