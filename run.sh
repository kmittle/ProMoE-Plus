#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2

# ============ Sampling ============
python sample.py\
 --config configs/004_ProMoE_S_sigmoid.yaml\
 --step_list_for_sample 100000,300000,500000\
 --guide_scale_list 1.0,1.5\
 2>&1 | tee log_sample_ProMoE_S_sigmoid.log

# ============ Evaluation ============
SAMPLE_BASE="outputs/ProMoE_TC_S_sigmoid/004_ProMoE_S_sigmoid/sample"
STEPS="100000 300000 500000"
SCALES="1.0 1.5"
SEED=0
FID_K=50
BS=48
LOG="log_sample_ProMoE_S_sigmoid.log"

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
