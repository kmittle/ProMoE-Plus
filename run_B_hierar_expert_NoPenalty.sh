#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG="configs/004_ProMoE_B_hierar_expert_NoPenalty.yaml"
LOG="log_ProMoE_B_hierar_expert_NoPenalty.log"

# ============ Training (stop at 500K) ============
python train.py --config $CONFIG \
 2>&1 | tee $LOG

# ============ Sampling ============
python sample.py \
 --config $CONFIG \
 --step_list_for_sample 300000,500000 \
 --guide_scale_list 1.0,1.5 \
 2>&1 | tee -a $LOG

# ============ Evaluation ============
SAMPLE_BASE="outputs/ProMoE_TC_B_hierar_expert/004_ProMoE_B_hierar_expert_NoPenalty/sample"
STEPS="300000 500000"
SCALES="1.0 1.5"
SEED=0
FID_K=50
BS=128

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate promoe_eval

echo "" >> $LOG
echo "============ Evaluation Results ============" >> $LOG

IMG_DIRS=()
for step in $STEPS; do
  for scale in $SCALES; do
    IMG_DIR="${SAMPLE_BASE}/step${step}/img256_cfg${scale}_seed${SEED}_FID${FID_K}K_bs${BS}_ema/images"
    if [ -d "$IMG_DIR" ]; then
      IMG_DIRS+=("../${IMG_DIR}")
    else
      echo ">>> step=${step} cfg=${scale}: image dir not found, skipping" | tee -a $LOG
    fi
  done
done

if [ ${#IMG_DIRS[@]} -gt 0 ]; then
  (cd evaluation && python run_eval.py "${IMG_DIRS[@]}" --count 50000) 2>&1 | tee -a $LOG
fi

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate promoe
