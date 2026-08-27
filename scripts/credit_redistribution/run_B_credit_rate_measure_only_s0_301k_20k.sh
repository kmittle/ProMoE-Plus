#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/004_ProMoE_B_credit_rate_measure_only_s0_301k_20k.yaml"
LOG="log_ProMoE_B_credit_rate_measure_only_s0_301k_20k.log"
PYTHON="/mnt/workspace/yujie/.conda/envs/promoe/bin/python"

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="4,5,6,7"
fi

"$PYTHON" analyses/run_credit_redistribution_gate.py verify-launch \
  --branch measure_only_control
"$PYTHON" train.py --config "$CONFIG" \
  2>&1 | tee "$LOG"
