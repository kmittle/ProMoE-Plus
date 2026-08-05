#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG="configs/004_ProMoE_B_hetero_expert.yaml"
LOG="log_ProMoE_B_hetero_expert.log"

SAMPLE_BASE="outputs/ProMoE_TC_B_hetero_expert/004_ProMoE_B_hetero_expert/sample"
STEPS="300000 500000"
SCALES="1.0 1.5"
SEED=0
FID_K=50
BS=128

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate promoe

python sample.py \
  --config "$CONFIG" \
  --step_list_for_sample 300000,500000 \
  --guide_scale_list 1.0,1.5 \
  2>&1 | tee -a "$LOG"

conda activate promoe_eval

echo "" >> "$LOG"
echo "============ Evaluation Results ============" >> "$LOG"

for step in $STEPS; do
  for scale in $SCALES; do
    IMG_DIR="${REPO_ROOT}/${SAMPLE_BASE}/step${step}/img256_cfg${scale}_seed${SEED}_FID${FID_K}K_bs${BS}_ema/images"
    if [ -d "$IMG_DIR" ]; then
      echo "-------------------------------" | tee -a "$LOG"
      echo "Evaluating: ${IMG_DIR}" | tee -a "$LOG"
      echo "-------------------------------" | tee -a "$LOG"
      (cd evaluation && CUDA_VISIBLE_DEVICES=0 python run_eval.py "$IMG_DIR" --count 50000) 2>&1 | tee -a "$LOG"
    else
      echo ">>> step=${step} cfg=${scale}: image dir not found, skipping" | tee -a "$LOG"
    fi
  done
done

conda activate promoe
