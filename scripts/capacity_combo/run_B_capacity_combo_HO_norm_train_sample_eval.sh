#!/bin/bash
#
# ProMoE-TC capacity-combo output-view scale diagnostic (HO-norm).
# This is the H+O ablation with only the pooled-output normalization changed;
# it remains a sequential fresh train + sample + OpenAI-eval pipeline.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
source "${SCRIPT_DIR}/capacity_combo_eval_helpers.sh"
source "${REPO_ROOT}/scripts/_python_env.sh"

CONFIG="configs/004_ProMoE_B_capacity_combo_HO_norm.yaml"
LOG="${REPO_ROOT}/logs/log_ProMoE_B_capacity_combo_HO_norm_train_sample_eval.log"
mkdir -p "$(dirname "$LOG")"

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
gpu_str = ",".join(map(str, gpu_ids)) if isinstance(gpu_ids, list) else "0"
eval_gpu = str(gpu_ids[0]) if isinstance(gpu_ids, list) and gpu_ids else "0"
custom_cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
step_list = cfg.get("step_list_for_sample", [])
step_str = ",".join(map(str, step_list)) if step_list else ""
orig_num_steps = int(cfg.get("num_steps", 0))
output_root = cfg.get("output_dir", "outputs")
sample_base = os.path.join(output_root, model_name, custom_cfg_name, "sample")

print(model_name)
print(custom_cfg_name)
print(num_fid_samples)
print(eval_gpu)
print(gpu_str)
print(step_str)
print(orig_num_steps)
print(sample_base)
PY
)

MODEL_NAME="${YAML_INFO[0]}"
CUSTOM_CFG_NAME="${YAML_INFO[1]}"
NUM_FID_SAMPLES="${YAML_INFO[2]}"
EVAL_GPU="${YAML_INFO[3]}"
GPU_IDS="${YAML_INFO[4]}"
STEP_LIST_STR="${YAML_INFO[5]}"
ORIG_NUM_STEPS="${YAML_INFO[6]}"
SAMPLE_BASE="${YAML_INFO[7]}"
if [[ "$SAMPLE_BASE" != /* ]]; then
    SAMPLE_BASE="${REPO_ROOT}/${SAMPLE_BASE}"
fi
OUTPUT_BASE="${SAMPLE_BASE%/sample}"

if [[ -L "$OUTPUT_BASE" ]]; then
    echo "ERROR: output bucket must be a real directory, not a symlink: $OUTPUT_BASE" >&2
    exit 1
fi
if [[ -e "$OUTPUT_BASE" ]] && find "$OUTPUT_BASE" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: training-from-scratch output bucket is not empty: $OUTPUT_BASE" >&2
    exit 1
fi

if [[ -z "$STEP_LIST_STR" ]]; then
    echo "ERROR: step_list_for_sample is empty or missing in ${CONFIG}" >&2
    exit 1
fi
IFS=',' read -ra ALL_STEPS <<< "$STEP_LIST_STR"
NUM_ALL_STEPS=${#ALL_STEPS[@]}

TEMP_DIR=$(mktemp -d)
TEMP_CONFIG="${TEMP_DIR}/$(basename "$CONFIG")"
trap 'rm -rf "$TEMP_DIR"' EXIT

sample_and_eval_step() {
    local step=$1
    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} started" | tee -a "$LOG"

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "$PYTHON" sample.py \
        --config "${CONFIG}" --step_list_for_sample "${step}" \
        >> "$LOG" 2>&1

    capacity_combo_eval_images "$SAMPLE_BASE" "$step" "$LOG" \
        "$EVAL_GPU" "$PYTHON_EVAL" "$NUM_FID_SAMPLES"

    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} done" | tee -a "$LOG"
}

echo "============================================================" | tee "$LOG"
echo "Sequential pipeline: ${MODEL_NAME}" | tee -a "$LOG"
echo "Config: ${CONFIG}" | tee -a "$LOG"
echo "Steps: ${STEP_LIST_STR}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for i in "${!ALL_STEPS[@]}"; do
    step="${ALL_STEPS[$i]}"
    phase=$((i + 1))
    if [[ "$phase" -lt "$NUM_ALL_STEPS" ]]; then
        TARGET_NUM_STEPS=$((step + 1))
    else
        TARGET_NUM_STEPS="$ORIG_NUM_STEPS"
    fi

    "$PYTHON" - "$CONFIG" "$TARGET_NUM_STEPS" "$TEMP_CONFIG" "$phase" <<'PY'
import sys
import yaml

source, target_steps, target, phase = sys.argv[1:]
with open(source) as f:
    cfg = yaml.safe_load(f)
cfg["num_steps"] = int(target_steps)
cfg["resume_checkpoint"] = int(phase) > 1
with open(target, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PY

    echo "============================================================" | tee -a "$LOG"
    echo "Phase ${phase}/${NUM_ALL_STEPS}: Train to step ${step} (num_steps=${TARGET_NUM_STEPS})" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "$PYTHON" train.py --config "${TEMP_CONFIG}" \
        >> "$LOG" 2>&1
    TRAIN_RC=$?
    set -e
    if [[ "$TRAIN_RC" -ne 0 ]]; then
        echo "Training FAILED at phase ${phase} (exit code ${TRAIN_RC})" | tee -a "$LOG"
        exit "$TRAIN_RC"
    fi

    echo "Phase ${phase}/${NUM_ALL_STEPS}: Sample+eval step ${step}" | tee -a "$LOG"
    sample_and_eval_step "$step"

    # The first configured point is the mandatory decision boundary.  A gate
    # failure is a normal scientific outcome, so leave the 300K artifacts in
    # place and exit successfully without starting the next training phase.
    if [[ "$step" == "300000" && "$phase" -lt "$NUM_ALL_STEPS" ]]; then
        if capacity_combo_check_300k_gate "$SAMPLE_BASE" "$step" "$LOG"; then
            :
        else
            gate_rc=$?
            if [[ "$gate_rc" -eq 1 ]]; then
                # A genuine 300K FID gate failure is a valid negative result.
                exit 0
            fi
            echo "ERROR: 300K gate evaluation failed (rc=${gate_rc})" | tee -a "$LOG" >&2
            exit "$gate_rc"
        fi
    fi
done

echo "All done." | tee -a "$LOG"
