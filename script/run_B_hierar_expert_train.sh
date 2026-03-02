#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG="configs/004_ProMoE_B_hierar_expert.yaml"
LOG="log_ProMoE_B_hierar_expert.log"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate promoe

python train.py --config "$CONFIG" \
  2>&1 | tee "$LOG"
