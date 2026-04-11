#!/bin/bash
#
# Template for ProMoE train + sample + eval end-to-end scripts.
#
# Pipeline:
#   1. Training runs in background to num_steps.
#   2. For each checkpoint in step_list_for_sample *except the last*,
#      a watcher polls for the checkpoint file and launches sample+eval
#      as soon as it appears — while training continues on the same GPUs.
#   3. After training finishes, sample+eval runs for the final checkpoint.
#
# Prerequisites:
#   - conda envs: promoe (train/sample), promoe_eval (evaluation)
#   - A100 80GB or equivalent (training + sampling share GPUs)
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

print(model_name)
print(custom_cfg_name)
print(num_fid_samples)
print(eval_gpu)
print(gpu_str)
print(step_str)
PY
)

MODEL_NAME="${YAML_INFO[0]}"
CUSTOM_CFG_NAME="${YAML_INFO[1]}"
NUM_FID_SAMPLES="${YAML_INFO[2]}"
EVAL_GPU="${YAML_INFO[3]}"
GPU_IDS="${YAML_INFO[4]}"
STEP_LIST_STR="${YAML_INFO[5]}"
SAMPLE_BASE="${REPO_ROOT}/outputs/${MODEL_NAME}/${CUSTOM_CFG_NAME}/sample"
CKPT_DIR="${REPO_ROOT}/outputs/${MODEL_NAME}/${CUSTOM_CFG_NAME}/checkpoints"

PYTHON="/mnt/workspace/yujie/.conda/envs/promoe/bin/python"
PYTHON_EVAL="/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python"

# ── Parse step_list into early steps (eval during training) + final step ─────
if [ -z "$STEP_LIST_STR" ]; then
    echo "ERROR: step_list_for_sample is empty or missing in ${CONFIG}" >&2
    exit 1
fi
IFS=',' read -ra ALL_STEPS <<< "$STEP_LIST_STR"
NUM_STEPS=${#ALL_STEPS[@]}
FINAL_STEP="${ALL_STEPS[-1]}"
if [ "$NUM_STEPS" -gt 1 ]; then
    EARLY_STEPS=("${ALL_STEPS[@]:0:$((NUM_STEPS-1))}")
else
    EARLY_STEPS=()
fi

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
# Step 1: Training (background)
# ══════════════════════════════════════════════════════════════════════════════
echo "============================================================" | tee "$LOG"
echo "Step 1: Training ${MODEL_NAME} (background)" | tee -a "$LOG"
echo "Config: ${CONFIG}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" $PYTHON train_with_repa.py \
    --config "${CONFIG}" \
    >> "$LOG" 2>&1 &
TRAIN_PID=$!

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Watch for early checkpoints → sample + eval while training runs
# ══════════════════════════════════════════════════════════════════════════════
EVAL_PIDS=()
if [ ${#EARLY_STEPS[@]} -gt 0 ]; then
    for step in "${EARLY_STEPS[@]}"; do
        CKPT_FILE="${CKPT_DIR}/ckpt_step_${step}.pth"
        echo "Watching for checkpoint: step ${step} ..." | tee -a "$LOG"

        while kill -0 "$TRAIN_PID" 2>/dev/null && [ ! -f "$CKPT_FILE" ]; do
            sleep 60
        done

        if [ -f "$CKPT_FILE" ]; then
            sleep 120  # wait for torch.save to finish (non-atomic write, large models may take minutes)
            sample_and_eval_step "$step" &
            EVAL_PIDS+=($!)
        else
            echo "WARNING: training exited before step ${step} checkpoint" | tee -a "$LOG"
        fi
    done
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Wait for training to finish
# ══════════════════════════════════════════════════════════════════════════════
set +e
wait "$TRAIN_PID"
TRAIN_RC=$?
set -e

if [ $TRAIN_RC -ne 0 ]; then
    echo "Training FAILED (exit code $TRAIN_RC)" | tee -a "$LOG"
    # Still wait for any in-flight eval jobs before exiting
    for pid in "${EVAL_PIDS[@]+"${EVAL_PIDS[@]}"}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit $TRAIN_RC
fi

echo "Training completed successfully" | tee -a "$LOG"

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Sample + eval final step (foreground, GPUs now free from training)
# ══════════════════════════════════════════════════════════════════════════════
echo "============================================================" | tee -a "$LOG"
echo "Step 4: Final sample+eval step ${FINAL_STEP}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
sample_and_eval_step "$FINAL_STEP"

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Wait for any remaining background eval jobs
# ══════════════════════════════════════════════════════════════════════════════
for pid in "${EVAL_PIDS[@]+"${EVAL_PIDS[@]}"}"; do
    wait "$pid" 2>/dev/null || true
done

echo "============================================================" | tee -a "$LOG"
echo "All done." | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
