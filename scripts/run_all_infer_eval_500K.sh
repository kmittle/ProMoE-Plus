#!/bin/bash
#
# Sample (500K checkpoint) + Evaluate for all models in scripts/.
# Usage: bash run_all_infer_eval_500K.sh
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Shared parameters (must match YAML configs)
STEP=500000
SCALES="1.0 1.5"
SEED=0
FID_K=50
BS=128
COUNT=50000
GPUS="0,1,2,3"

# Config list: <yaml_config_path> <model_name> <custom_cfg_name>
# model_name and custom_cfg_name determine the output directory:
#   outputs/<model_name>/<custom_cfg_name>/sample/step500000/...
CONFIGS=(
  "configs/004_ProMoE_B_hierar.yaml               ProMoE_TC_B_hierar          004_ProMoE_B_hierar"
  "configs/004_ProMoE_B_hierar_expert.yaml         ProMoE_TC_B_hierar_expert   004_ProMoE_B_hierar_expert"
  "configs/004_ProMoE_B_hierar_expert_NoPenalty.yaml ProMoE_TC_B_hierar_expert 004_ProMoE_B_hierar_expert_NoPenalty"
  "configs/004_ProMoE_B_repa.yaml                  ProMoE_TC_REPA_B            004_ProMoE_B_repa"
  "configs/004_ProMoE_B_repa_cond.yaml             ProMoE_TC_REPA_Cond_B       004_ProMoE_B_repa_cond"
  "configs/004_ProMoE_B_repa_shared.yaml           ProMoE_TC_REPA_Shared_B     004_ProMoE_B_repa_shared"
)

LOG="${REPO_ROOT}/log_all_infer_eval_500K.log"
FAILED=()

eval "$(conda shell.bash hook 2>/dev/null)"

# ==============================================================
# Step 1: Sampling (promoe env)
# ==============================================================
conda activate promoe

echo "============================================================" | tee "$LOG"
echo "Step 1: Sampling all models at step ${STEP}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for entry in "${CONFIGS[@]}"; do
  read -r cfg_path model_name custom_cfg_name <<< "$entry"
  echo "" | tee -a "$LOG"
  echo ">>> Sampling: ${model_name} (${cfg_path})" | tee -a "$LOG"

  if CUDA_VISIBLE_DEVICES="${GPUS}" python sample.py \
    --config "$cfg_path" \
    --step_list_for_sample "${STEP}" \
    --guide_scale_list 1.0,1.5 \
    --num_fid_samples "${COUNT}" \
    2>&1 | tee -a "$LOG"; then
    echo ">>> Sampling done: ${model_name}" | tee -a "$LOG"
  else
    echo ">>> FAILED sampling: ${model_name}" | tee -a "$LOG"
    FAILED+=("sample:${model_name}")
  fi
done

# ==============================================================
# Step 2: Evaluation (promoe_eval env)
# ==============================================================
conda activate promoe_eval

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "Step 2: Evaluating all models" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for entry in "${CONFIGS[@]}"; do
  read -r cfg_path model_name custom_cfg_name <<< "$entry"

  for scale in $SCALES; do
    IMG_DIR="${REPO_ROOT}/outputs/${model_name}/${custom_cfg_name}/sample/step${STEP}/img256_cfg${scale}_seed${SEED}_FID${FID_K}K_bs${BS}_ema/images"
    if [ -d "$IMG_DIR" ]; then
      echo "" | tee -a "$LOG"
      echo "-------------------------------" | tee -a "$LOG"
      echo "Evaluating: ${model_name} cfg=${scale}" | tee -a "$LOG"
      echo "  ${IMG_DIR}" | tee -a "$LOG"
      echo "-------------------------------" | tee -a "$LOG"
      if ! (cd evaluation && CUDA_VISIBLE_DEVICES=0 python run_eval.py "$IMG_DIR" --count "${COUNT}") 2>&1 | tee -a "$LOG"; then
        echo ">>> FAILED eval: ${model_name} cfg=${scale}" | tee -a "$LOG"
        FAILED+=("eval:${model_name}_cfg${scale}")
      fi
    else
      echo ">>> ${model_name} cfg=${scale}: image dir not found, skipping" | tee -a "$LOG"
    fi
  done
done

# ==============================================================
# Summary
# ==============================================================
echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All done. No failures." | tee -a "$LOG"
else
  echo "Done with ${#FAILED[@]} failure(s):" | tee -a "$LOG"
  for f in "${FAILED[@]}"; do
    echo "  - ${f}" | tee -a "$LOG"
  done
fi
echo "============================================================" | tee -a "$LOG"
