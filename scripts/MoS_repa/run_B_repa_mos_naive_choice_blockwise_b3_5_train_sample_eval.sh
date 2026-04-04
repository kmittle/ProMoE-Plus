#!/bin/bash
#
# Train ProMoE-TC-REPA-MoS-Naive-Choice-Blockwise-B (align blocks 3-5),
# then sample and evaluate FID.
#
# Ablation: block-wise routing (all tokens share the same routing weights)
# vs token-wise routing in the original naive_choice variant.
# All other hyperparameters identical to 004_ProMoE_B_repa_MoS_naive_choice_b3_5.yaml.
#
# Step 1: Train with configs/004_ProMoE_B_repa_MoS_naive_choice_blockwise_b3_5.yaml
# Step 2: Sample with parameters from YAML
# Step 3: Evaluate generated images with OpenAI evaluator (count from YAML)
#
# Prerequisites:
#   - conda envs: promoe (train/sample), promoe_eval (evaluation)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/004_ProMoE_B_repa_MoS_naive_choice_blockwise_b3_5.yaml"
LOG="log_ProMoE_B_repa_MoS_naive_choice_blockwise_b3_5_train_sample_eval.log"

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
gpu_str = ",".join(map(str, gpu_ids)) if isinstance(gpu_ids, list) else "0"
eval_gpu = str(gpu_ids[0]) if isinstance(gpu_ids, list) and len(gpu_ids) > 0 else "0"
custom_cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]

print(model_name)
print(custom_cfg_name)
print(num_fid_samples)
print(eval_gpu)
print(gpu_str)
PY
)

MODEL_NAME="${YAML_INFO[0]}"
CUSTOM_CFG_NAME="${YAML_INFO[1]}"
NUM_FID_SAMPLES="${YAML_INFO[2]}"
EVAL_GPU="${YAML_INFO[3]}"
GPU_IDS="${YAML_INFO[4]}"
SAMPLE_BASE="${REPO_ROOT}/outputs/${MODEL_NAME}/${CUSTOM_CFG_NAME}/sample"

echo "============================================================" | tee "$LOG"
echo "Step 1: Training ${MODEL_NAME}" | tee -a "$LOG"
echo "Config: ${CONFIG}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" /mnt/workspace/yujie/.conda/envs/promoe/bin/python train_with_MoS_repa.py \
  --config "${CONFIG}" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "Step 2: Sampling (all params from YAML)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" /mnt/workspace/yujie/.conda/envs/promoe/bin/python sample.py \
  --config "${CONFIG}" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "Step 3: Evaluation" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

if [ ! -d "$SAMPLE_BASE" ]; then
  echo ">>> sample root not found: $SAMPLE_BASE" | tee -a "$LOG"
else
  while IFS= read -r IMG_DIR; do
    echo "-------------------------------" | tee -a "$LOG"
    echo "Evaluating: ${IMG_DIR}" | tee -a "$LOG"
    echo "-------------------------------" | tee -a "$LOG"
    (cd evaluation && CUDA_VISIBLE_DEVICES="${EVAL_GPU}" /mnt/workspace/yujie/.conda/envs/fid_eval/bin/python run_eval.py "$IMG_DIR" --count "${NUM_FID_SAMPLES}") 2>&1 | tee -a "$LOG"
  done < <(find "$SAMPLE_BASE" -mindepth 3 -maxdepth 3 -type d -name images | sort -V)
fi
