#!/usr/bin/env bash
# Wait for prior processes to exit, then claim four GPUs for the p-only ablation.
# Run in a new window of the current tmux session.
# Usage: bash scripts/adepth/start_q0p3_ablation_when_ready.sh p_only [PID ...]
# Each PID's start time is recorded so PID reuse cannot extend the dependency.
# The generated run-time wrapper and template-based pipeline remain unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
[[ -n "${TMUX:-}" ]] || { echo "ERROR: run inside the current tmux session" >&2; exit 2; }

ARM="${1:?Usage: start_q0p3_ablation_when_ready.sh p_only [PID ...]}"
shift
case "$ARM" in p_only) ;; *) echo "ERROR: invalid arm: $ARM" >&2; exit 2 ;; esac
[[ -z "${PROMOE_GPU_IDS_OVERRIDE:-}" ]] || {
    echo "ERROR: this waiting launcher claims GPUs; use the dated wrapper to set GPU IDs manually" >&2
    exit 2
}
source "${REPO_ROOT}/scripts/_python_env.sh"
source "${REPO_ROOT}/scripts/_run_times/claim_gpus.sh"

WRAPPER="${REPO_ROOT}/scripts/_run_times/2026_10_21/B_adepth_q0p3_${ARM}.sh"
[[ -f "$WRAPPER" ]] || { echo "ERROR: missing wrapper: $WRAPPER" >&2; exit 2; }
LOG="${REPO_ROOT}/logs/log_adepth_q0p3_${ARM}_launch.log"
CLAIM_LOG="${REPO_ROOT}/logs/log_adepth_q0p3_${ARM}_gpu_wait.log"
mkdir -p "$(dirname "$LOG")"
# Keep the repository-local lock for the whole pipeline, including its wait.
exec 9>>"$LOG"
flock -n 9 || { echo "ERROR: this arm is already waiting or running: $ARM" >&2; exit 2; }

declare -A prior_start=()
for pid in "$@"; do
    [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || { echo "ERROR: invalid PID: $pid" >&2; exit 2; }
    started="$(ps -p "$pid" -o lstart= 2>/dev/null || true)"
    [[ -z "$started" ]] || prior_start["$pid"]="$started"
done
echo "[$(date -Is)] $ARM: waiting for prior process(es): ${*:-none}" | tee -a "$LOG"
while true; do
    pending=0
    for pid in "${!prior_start[@]}"; do
        current_start="$(ps -p "$pid" -o lstart= 2>/dev/null || true)"
        if [[ "$current_start" == "${prior_start[$pid]}" ]]; then
            pending=1
        fi
    done
    [[ "$pending" -eq 0 ]] && break
    sleep 30
done

echo "[$(date -Is)] $ARM: waiting for four idle GPUs" | tee -a "$LOG"
while true; do
    claim_rc=0
    # Direct call keeps the claim owned by this live parent through the pipeline.
    promoe_claim_gpus 4 >"$CLAIM_LOG" 2>&1 || claim_rc=$?
    [[ "$claim_rc" -eq 0 ]] && break
    [[ "$claim_rc" -eq 1 ]] || exit "$claim_rc"
    sleep 30
done
export PROMOE_GPU_IDS_OVERRIDE="$PROMOE_CLAIMED_GPUS"
echo "[$(date -Is)] $ARM: starting on GPU(s) $PROMOE_GPU_IDS_OVERRIDE" | tee -a "$LOG"
# Keep this parent alive so the shared helper's EXIT trap releases its own claims.
set +e
bash "$WRAPPER"
run_rc=$?
set -e
echo "[$(date -Is)] $ARM: pipeline exit=$run_rc" | tee -a "$LOG"
exit "$run_rc"
