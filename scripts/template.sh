#!/bin/bash
#
# Template for ProMoE train + sample + eval end-to-end scripts.
#
# Pipeline (sequential, safe for XL-scale models):
#   For each checkpoint step in step_list_for_sample:
#     1. Train (or resume) up to that step, then stop.
#     2. Sample + eval using that checkpoint (GPUs fully free).
#   This avoids concurrent training + sampling, which may exceed GPU memory
#   for XL-scale models even on A100 80GB.
#
# Prerequisites:
#   - conda envs: promoe (train/sample), promoe_eval (evaluation)
#   - A100 80GB or equivalent
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/004_ProMoE_B_repa_dyna_only.yaml"
LOG="log_ProMoE_B_repa_dyna_only_train_sample_eval.log"

readarray -t YAML_INFO < <(python - "$CONFIG" <<'PY'
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

PYTHON="/mnt/workspace/yujie/.conda/envs/promoe/bin/python"
PYTHON_EVAL="/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python"

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

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" $PYTHON sample.py \
        --config "${CONFIG}" --step_list_for_sample "${step}" \
        >> "$LOG" 2>&1

    if [ -d "$SAMPLE_BASE" ]; then
        while IFS= read -r IMG_DIR; do
            echo "[$(date '+%H:%M:%S')] Evaluating: ${IMG_DIR}" | tee -a "$LOG"
            (cd evaluation && CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
                $PYTHON_EVAL run_eval.py "$IMG_DIR" --count "${NUM_FID_SAMPLES}") \
                >> "$LOG" 2>&1
        done < <(find "$SAMPLE_BASE" -mindepth 3 -maxdepth 3 -path "*/step${step}/*" -type d -name images | sort -V)
    fi

    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} done" | tee -a "$LOG"
}

# ══════════════════════════════════════════════════════════════════════════════
# Sequential pipeline: train → stop → sample + eval → resume → ...
# ══════════════════════════════════════════════════════════════════════════════
echo "============================================================" | tee "$LOG"
echo "Sequential pipeline: ${MODEL_NAME}" | tee -a "$LOG"
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

    # Generate temp config with adjusted num_steps and resume enabled
    python - "$CONFIG" "$TARGET_NUM_STEPS" "$TEMP_CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
cfg['num_steps'] = int(sys.argv[2])
cfg['resume_checkpoint'] = True
with open(sys.argv[3], 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PY

    # ── Train ─────────────────────────────────────────────────────────────────
    echo "============================================================" | tee -a "$LOG"
    echo "Phase ${phase}/${NUM_ALL_STEPS}: Train to step ${step} (num_steps=${TARGET_NUM_STEPS})" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" $PYTHON train_with_repa.py \
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
