#!/bin/bash
#
# Train ProMoE-TC-REPA-DYNA-B, then sample and evaluate FID.
#
# Step 1: Train with configs/004_ProMoE_B_repa_dyna.yaml
# Step 2: Sample 50K images at step 300K & 500K (CFG 1.0 & 1.5)
# Step 3: Evaluate generated images with OpenAI evaluator
#
# Prerequisites:
#   - conda envs: promoe (train/sample), promoe_eval (evaluation)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/004_ProMoE_B_repa_dyna.yaml"
MODEL_NAME="ProMoE_TC_REPA_DYNA_B"
CUSTOM_CFG_NAME="004_ProMoE_B_repa_dyna"
LOG="log_ProMoE_B_repa_dyna_train_sample_eval.log"

STEP_LIST_FOR_SAMPLE="300000,500000"
STEPS="300000 500000"
GUIDE_SCALE_LIST="1.0,1.5"
SCALES="1.0 1.5"
SEED=0
FID_K=50
BS=128
NUM_FID_SAMPLES=50000
GPUS="0,1,2,3"

eval "$(conda shell.bash hook 2>/dev/null)"

echo "============================================================" | tee "$LOG"
echo "Step 1: Training ${MODEL_NAME}" | tee -a "$LOG"
echo "Config: ${CONFIG}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

conda activate promoe

python train_with_repa.py \
  --config "${CONFIG}" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "Step 2: Sampling at steps ${STEP_LIST_FOR_SAMPLE}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${GPUS}" python sample.py \
  --config "${CONFIG}" \
  --step_list_for_sample "${STEP_LIST_FOR_SAMPLE}" \
  --guide_scale_list "${GUIDE_SCALE_LIST}" \
  --num_fid_samples "${NUM_FID_SAMPLES}" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "Step 3: Evaluation" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

conda activate promoe_eval

for step in $STEPS; do
  for scale in $SCALES; do
    IMG_DIR="${REPO_ROOT}/outputs/${MODEL_NAME}/${CUSTOM_CFG_NAME}/sample/step${step}/img256_cfg${scale}_seed${SEED}_FID${FID_K}K_bs${BS}_ema/images"
    if [ -d "$IMG_DIR" ]; then
      echo "-------------------------------" | tee -a "$LOG"
      echo "Evaluating: ${IMG_DIR}" | tee -a "$LOG"
      echo "-------------------------------" | tee -a "$LOG"
      (cd evaluation && CUDA_VISIBLE_DEVICES=0 python run_eval.py "$IMG_DIR" --count "${NUM_FID_SAMPLES}") 2>&1 | tee -a "$LOG"
    else
      echo ">>> step=${step} cfg=${scale}: image dir not found, skipping" | tee -a "$LOG"
    fi
  done
done

conda activate promoe

