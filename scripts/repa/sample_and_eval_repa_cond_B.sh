#!/bin/bash
#
# Sample from ProMoE-TC-REPA-Cond-B at step 300K & 500K, then evaluate FID.
#
# Step 1: Generate 50K images using each checkpoint (CFG 1.0 & 1.5)
# Step 2: Pack images into .npz and run OpenAI evaluator for each CFG scale
#
# Prerequisites:
#   - Training checkpoints at outputs/ProMoE_TC_REPA_Cond_B/004_ProMoE_B_repa_cond/checkpoints/
#   - conda envs: promoe (sampling), promoe_eval (evaluation)
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Allow conda activate in non-interactive shell
eval "$(conda shell.bash hook)"

CONFIG="configs/004_ProMoE_B_repa_cond.yaml"
SAMPLE_STEPS="300000,500000"
NUM_FID_SAMPLES=50000
GPUS="0,1,2,3"

# ============================================================
# Step 1: Sample (promoe env, same GPUs as training)
# ============================================================
echo "=============================="
echo "Step 1: Sampling at steps ${SAMPLE_STEPS}"
echo "=============================="

conda activate promoe

CUDA_VISIBLE_DEVICES="${GPUS}" python sample.py \
    --config "${CONFIG}" \
    --step_list_for_sample "${SAMPLE_STEPS}" \
    --num_fid_samples "${NUM_FID_SAMPLES}"

# ============================================================
# Step 2: Evaluate (promoe_eval env)
# ============================================================
echo "=============================="
echo "Step 2: Evaluation"
echo "=============================="

conda activate promoe_eval

cd "${PROJECT_ROOT}/evaluation"

for step in 300000 500000; do
    SAMPLE_ROOT="${PROJECT_ROOT}/outputs/ProMoE_TC_REPA_Cond_B/004_ProMoE_B_repa_cond/sample/step${step}"
    if [ ! -d "${SAMPLE_ROOT}" ]; then
        echo ">>> step=${step}: sample dir not found, skipping"
        continue
    fi
    for IMAGE_DIR in "${SAMPLE_ROOT}"/*/images; do
        if [ -d "${IMAGE_DIR}" ]; then
            echo "-------------------------------"
            echo "Evaluating: ${IMAGE_DIR}"
            echo "-------------------------------"
            CUDA_VISIBLE_DEVICES=0 python run_eval.py "${IMAGE_DIR}" \
                --count "${NUM_FID_SAMPLES}"
        fi
    done
done
