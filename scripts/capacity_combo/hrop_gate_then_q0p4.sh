#!/usr/bin/env bash
# Finish the HROP gate before handing GPUs 4-7 to the next experiment.
#
# The HROP run was intentionally started with a temporary 300K-only config.
# This supervisor keeps its result, applies the project-level two-CFG FID gate,
# and resumes the same output bucket to 500K only after a genuine gate pass.
# It is intentionally tied to that temporary launcher and rejects the normal
# two-phase HROP wrapper, which would already perform its own continuation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
source "${SCRIPT_DIR}/capacity_combo_eval_helpers.sh"
source "${REPO_ROOT}/scripts/_python_env.sh"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ || "$1" -le 1 ]]; then
    echo "Usage: $0 HROP_WRAPPER_PID" >&2
    exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
    echo "ERROR: this long-lived supervisor must run inside an attached tmux session" >&2
    exit 1
fi

HROP_PID="$1"
HROP_TEMP_MARKER="promoe-HROP-gate"
GPU_IDS="4,5,6,7"
EVAL_GPU="4"
NUM_FID_SAMPLES=50000
BASELINE_CFG1="30.584602064850174"
BASELINE_CFG15="9.588081719517504"
HROP_OUTPUT="${REPO_ROOT}/outputs/ProMoE_TC_B_capacity_combo/004_ProMoE_B_capacity_combo_HROP"
HROP_SAMPLE_BASE="${HROP_OUTPUT}/sample"
HROP_CONFIG="${REPO_ROOT}/configs/004_ProMoE_B_capacity_combo_HROP.yaml"
Q0P4_OUTPUT="${REPO_ROOT}/outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p4"
Q0P4_WRAPPER="${REPO_ROOT}/scripts/_run_times/2026_07_01/10.2-B_adepth_q0p4.sh"
LOG="${REPO_ROOT}/logs/hrop_gate_then_q0p4.log"
mkdir -p "$(dirname "$LOG")"

LOCK_FILE="/tmp/promoe-hrop-gate-then-q0p4.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another HROP gate supervisor already owns ${LOCK_FILE}" >&2
    exit 0
fi

INITIAL_COMMAND_LINE="$(ps -p "$HROP_PID" -o args= 2>/dev/null || true)"
if [[ -z "$INITIAL_COMMAND_LINE" \
    || "$INITIAL_COMMAND_LINE" != *"${HROP_TEMP_MARKER}"* ]]; then
    echo "ERROR: PID ${HROP_PID} is not the live temporary 300K-only HROP process" >&2
    exit 1
fi

[[ -d "$HROP_OUTPUT" && ! -L "$HROP_OUTPUT" ]] || {
    echo "ERROR: HROP output bucket must be a real directory: $HROP_OUTPUT" >&2
    exit 1
}
[[ -f "$HROP_CONFIG" && ! -L "$HROP_CONFIG" ]] || {
    echo "ERROR: HROP source config is missing or is a symlink: $HROP_CONFIG" >&2
    exit 1
}
[[ -x "$Q0P4_WRAPPER" && ! -L "$Q0P4_WRAPPER" ]] || {
    echo "ERROR: q0p4 runtime wrapper is missing or not executable: $Q0P4_WRAPPER" >&2
    exit 1
}

PYTHON="${PROMOE_TRAIN_PYTHON}"
PYTHON_EVAL="${PROMOE_EVAL_PYTHON}"
[[ -x "$PYTHON" ]] || {
    echo "ERROR: required training interpreter is missing: $PYTHON" | tee -a "$LOG" >&2
    exit 1
}
[[ -x "$PYTHON_EVAL" ]] || {
    echo "ERROR: required evaluation interpreter is missing: $PYTHON_EVAL" | tee -a "$LOG" >&2
    exit 1
}
if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} cannot import PyYAML" | tee -a "$LOG" >&2
    exit 1
fi

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] HROP gate supervisor started for wrapper PID ${HROP_PID}"

TEMP_DIR=""
cleanup() {
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

wait_for_hrop() {
    echo "[$(date -Is)] Waiting for HROP 300K training and evaluation"
    while kill -0 "$HROP_PID" 2>/dev/null; do
        local command_line
        command_line="$(ps -p "$HROP_PID" -o args= 2>/dev/null || true)"
        if [[ -n "$command_line" \
            && "$command_line" != *"${HROP_TEMP_MARKER}"* ]]; then
            echo "ERROR: PID ${HROP_PID} is not the temporary 300K-only HROP process" >&2
            return 1
        fi
        sleep 60
    done
    echo "[$(date -Is)] HROP wrapper PID ${HROP_PID} ended"
}

metric_file() {
    local step=$1
    local cfg=$2
    printf '%s\n' "${HROP_SAMPLE_BASE}/step${step}/img256_cfg${cfg}_seed0_FID50K_bs128_ema/images_eval_openai.txt"
}

eval_pair_complete() {
    local step=$1
    local checkpoint="${HROP_OUTPUT}/checkpoints/ckpt_step_${step}.pth"
    local step_dir="${HROP_SAMPLE_BASE}/step${step}"
    local image_dir image_parent eval_file npz_file
    local -a image_dirs=()
    local cfg1_count=0 cfg15_count=0

    [[ -f "$checkpoint" && ! -L "$checkpoint" && -s "$checkpoint" ]] || return 1
    [[ -d "$step_dir" && ! -L "$step_dir" ]] || return 1
    mapfile -t image_dirs < <(
        find "$step_dir" -mindepth 2 -maxdepth 2 -type d -name images | sort -V
    )
    [[ "${#image_dirs[@]}" -eq 2 ]] || return 1
    for image_dir in "${image_dirs[@]}"; do
        [[ ! -L "$image_dir" ]] || return 1
        [[ -n "$(find "$image_dir" -mindepth 1 -maxdepth 1 \
            -type f -name '*.png' -print -quit 2>/dev/null)" ]] || return 1
        image_parent="$(dirname "$image_dir")"
        [[ ! -L "$image_parent" ]] || return 1
        eval_file="${image_parent}/images_eval_openai.txt"
        npz_file="${image_parent}/images.npz"
        [[ -f "$eval_file" && ! -L "$eval_file" && -s "$eval_file" ]] || return 1
        [[ -f "$npz_file" && ! -L "$npz_file" && -s "$npz_file" ]] || return 1
        promoe_eval_file_metrics_valid "$eval_file" || return 1
        case "$(basename "$image_parent")" in
            *_cfg1.0_*) cfg1_count=$((cfg1_count + 1)) ;;
            *_cfg1.5_*) cfg15_count=$((cfg15_count + 1)) ;;
            *) return 1 ;;
        esac
    done
    [[ "$cfg1_count" -eq 1 && "$cfg15_count" -eq 1 ]]
}

read_fid_pair() {
    local step=$1
    local f1 f15
    eval_pair_complete "$step" || {
        echo "ERROR: incomplete HROP evaluator pair for step ${step}" >&2
        return 1
    }
    f1="$(metric_file "$step" "1.0")"
    f15="$(metric_file "$step" "1.5")"
    promoe_eval_file_metrics_valid "$f1" || {
        echo "ERROR: invalid evaluator record: $f1" >&2
        return 1
    }
    promoe_eval_file_metrics_valid "$f15" || {
        echo "ERROR: invalid evaluator record: $f15" >&2
        return 1
    }
    printf '%s\n%s\n' "$(promoe_eval_file_fid "$f1")" "$(promoe_eval_file_fid "$f15")"
}

wait_for_gpu_idle() {
    local pids
    echo "[$(date -Is)] Waiting for GPUs 4-7 to become idle"
    while :; do
        if ! pids="$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null)"; then
            echo "[$(date -Is)] nvidia-smi query failed; treating GPUs as busy"
            sleep 30
            continue
        fi
        if [[ -z "${pids//[[:space:]]/}" ]]; then
            break
        fi
        sleep 30
    done
    echo "[$(date -Is)] GPUs 4-7 are idle"
}

launch_q0p4() {
    wait_for_gpu_idle
    if [[ -e "$Q0P4_OUTPUT" || -L "$Q0P4_OUTPUT" ]]; then
        echo "ERROR: q0p4 output already exists; refusing to overwrite: $Q0P4_OUTPUT" >&2
        return 1
    fi
    echo "[$(date -Is)] Launching q0p4 after HROP gate decision"
    trap - EXIT
    cleanup
    exec bash "$Q0P4_WRAPPER"
}

continue_hrop_to_500k() {
    local continuation_log="${REPO_ROOT}/logs/log_ProMoE_B_capacity_combo_HROP_500k_resume.log"
    local continuation_config
    wait_for_gpu_idle
    TEMP_DIR="$(mktemp -d /tmp/promoe-hrop-continue.XXXXXX)"
    continuation_config="${TEMP_DIR}/$(basename "$HROP_CONFIG")"

    [[ -f "${HROP_OUTPUT}/checkpoints/ckpt_step_300000.pth" \
        && ! -L "${HROP_OUTPUT}/checkpoints/ckpt_step_300000.pth" \
        && -s "${HROP_OUTPUT}/checkpoints/ckpt_step_300000.pth" ]] || {
        echo "ERROR: HROP 300K checkpoint is missing or is a symlink" >&2
        return 1
    }

    "$PYTHON" - "$HROP_CONFIG" "$continuation_config" <<'PY'
import sys
import yaml

source, target = sys.argv[1:]
with open(source) as handle:
    config = yaml.safe_load(handle)
config["num_steps"] = 501000
config["step_list_for_sample"] = [500000]
config["resume_checkpoint"] = True
config["resume_checkpoint_step"] = 300000
with open(target, "w") as handle:
    yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
PY

    echo "[$(date -Is)] Resuming HROP from ckpt_step_300000.pth to step 500000" \
        | tee -a "$continuation_log"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" train.py --config "$continuation_config" \
        >> "$continuation_log" 2>&1

    [[ -f "${HROP_OUTPUT}/checkpoints/ckpt_step_500000.pth" \
        && ! -L "${HROP_OUTPUT}/checkpoints/ckpt_step_500000.pth" \
        && -s "${HROP_OUTPUT}/checkpoints/ckpt_step_500000.pth" ]] || {
        echo "ERROR: HROP 500K checkpoint was not produced" >&2
        return 1
    }

    echo "[$(date -Is)] Sampling HROP step 500000" | tee -a "$continuation_log"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" sample.py \
        --config "$continuation_config" --step_list_for_sample 500000 \
        >> "$continuation_log" 2>&1
    capacity_combo_eval_images "$HROP_SAMPLE_BASE" 500000 "$continuation_log" \
        "$EVAL_GPU" "$PYTHON_EVAL" "$NUM_FID_SAMPLES"
    local fids fid1 fid15
    fids="$(read_fid_pair 500000)"
    fid1="$(printf '%s\n' "$fids" | sed -n '1p')"
    fid15="$(printf '%s\n' "$fids" | sed -n '2p')"
    echo "[$(date -Is)] HROP 500K FID CFG1.0=${fid1} CFG1.5=${fid15}" \
        | tee -a "$continuation_log"
}

wait_for_hrop

fids="$(read_fid_pair 300000)" || {
    echo "ERROR: HROP 300K evaluator pair is incomplete; q0p4 will not be launched" >&2
    exit 1
}
fid1="$(printf '%s\n' "$fids" | sed -n '1p')"
fid15="$(printf '%s\n' "$fids" | sed -n '2p')"
echo "[$(date -Is)] HROP 300K FID CFG1.0=${fid1} CFG1.5=${fid15} \
baseline=${BASELINE_CFG1}/${BASELINE_CFG15}"

if awk -v a="$fid1" -v b="$BASELINE_CFG1" -v c="$fid15" -v d="$BASELINE_CFG15" \
    'BEGIN { exit !(a < b && c < d) }'; then
    echo "[$(date -Is)] HROP 300K gate PASS; continuing the same trajectory to 500K"
    continue_hrop_to_500k
else
    echo "[$(date -Is)] HROP 300K gate FAIL; retaining the 300K result and abandoning HROP"
fi

launch_q0p4
