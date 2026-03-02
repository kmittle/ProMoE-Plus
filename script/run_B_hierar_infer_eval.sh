#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG="configs/004_ProMoE_B_hierar.yaml"
LOG="log_ProMoE_B_hierar.log"

SAMPLE_BASE="outputs/ProMoE_TC_B_hierar/004_ProMoE_B_hierar/sample"
STEPS="300000"
SCALES="1.0 1.5"
SEED=0
FID_K=50
BS=128

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate promoe

python sample.py \
  --config "$CONFIG" \
  --step_list_for_sample 300000 \
  --guide_scale_list 1.0,1.5 \
  2>&1 | tee -a "$LOG"

conda activate promoe_eval

echo "" >> "$LOG"
echo "============ Evaluation Results ============" >> "$LOG"

declare -a IMG_DIRS=()
for step in $STEPS; do
  for scale in $SCALES; do
    IMG_DIR="${REPO_ROOT}/${SAMPLE_BASE}/step${step}/img256_cfg${scale}_seed${SEED}_FID${FID_K}K_bs${BS}_ema/images"
    if [ -d "$IMG_DIR" ]; then
      IMG_DIRS+=("$IMG_DIR")
    else
      echo ">>> step=${step} cfg=${scale}: image dir not found, skipping" | tee -a "$LOG"
    fi
  done
done

if [ ${#IMG_DIRS[@]} -gt 0 ]; then
  (cd evaluation && python run_eval.py "${IMG_DIRS[@]}" --count 50000) 2>&1 | tee -a "$LOG"
fi

conda activate promoe
