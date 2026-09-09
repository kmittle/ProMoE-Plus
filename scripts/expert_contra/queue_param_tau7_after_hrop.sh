#!/usr/bin/env bash

# Start the tau=7 parameter ablation as soon as the current 4-7 GPU job
# (HROP) has completed its required 300K evaluation.  param_shared remains
# independent on GPUs 0-3 and is intentionally not a prerequisite here.

set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
    echo "ERROR: this long-lived queue must run inside an attached tmux session." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/_eval_metric_helpers.sh

queue_log="logs/expert_param_tau7_queue.log"
mkdir -p logs
exec > >(tee -a "$queue_log") 2>&1

hrop="outputs/ProMoE_TC_B_capacity_combo/004_ProMoE_B_capacity_combo_HROP"
hrop_log="logs/log_ProMoE_B_capacity_combo_HROP_300k_gate.log"
: "${PROMOE_HROP_PID:?PROMOE_HROP_PID must identify the current HROP launcher}"
hrop_pid="$PROMOE_HROP_PID"
if [[ ! "$hrop_pid" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PROMOE_HROP_PID must be numeric: ${hrop_pid}" >&2
    exit 1
fi

hrop_cmd="$(ps -p "$hrop_pid" -o args= 2>/dev/null || true)"
case "$hrop_cmd" in
    *run_B_capacity_combo_HROP_train_sample_eval.sh*) ;;
    *)
        echo "ERROR: PID ${hrop_pid} is not the expected HROP launcher." >&2
        exit 1
        ;;
esac

lock_file="/tmp/promoe_param_tau7_after_hrop.lock"
exec 9>"$lock_file"
flock -n 9 || {
    echo "Another param_tau7 queue is already active; exiting."
    exit 0
}

echo "[$(date -Is)] Waiting for HROP PID ${hrop_pid} to finish before using GPUs 4-7."
while kill -0 "$hrop_pid" 2>/dev/null; do
    sleep 60
done

if [[ ! -s "$hrop/checkpoints/ckpt_step_300000.pth" ]]; then
    echo "HROP ended without a 300K checkpoint; param_tau7 NOT started."
    exit 1
fi
if ! grep -Fxq 'All done.' "$hrop_log"; then
    echo "HROP pipeline did not complete successfully; param_tau7 NOT started."
    exit 1
fi

for cfg in 1.0 1.5; do
    eval_parent="$hrop/sample/step300000/img256_cfg${cfg}_seed0_FID50K_bs128_ema"
    if [[ ! -s "$eval_parent/images.npz" ]] \
        || ! promoe_eval_file_metrics_valid "$eval_parent/images_eval_openai.txt"; then
        echo "HROP 300K evaluation is incomplete; param_tau7 NOT started."
        exit 1
    fi
done

# HROP's FID determines its own continuation, not whether this independent
# parameter-temperature ablation runs. The separate HROP report records it.
echo "[$(date -Is)] HROP 300K evaluations verified; waiting for GPUs 4-7 to be free."
while true; do
    gpu_pids="$(nvidia-smi --id=4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -z "$gpu_pids" ]] && break
    sleep 60
done

export PROMOE_ALLOW_LOCAL_FALLBACK=1
export PROMOE_GPU_IDS_OVERRIDE=4,5,6,7
echo "[$(date -Is)] Launching param_tau7 from scratch on GPUs 4-7."
exec bash scripts/expert_contra/run_B_expert_contra_param_tau7_train_sample_eval.sh
