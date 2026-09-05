#!/usr/bin/env bash
# Serial fresh-data completion queue for the five missing parameter ablations.
# Each child wrapper remains the normal train/sample/eval pipeline; this file
# only makes the 0-3 GPU reservation explicit so historical 2-GPU launchers
# cannot be mistaken for the current completion run. The dispatcher is
# restart-safe: it never starts an active or already-complete variant twice,
# and it refuses to overwrite an incomplete non-empty output bucket.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PROMOE_ALLOW_LOCAL_FALLBACK=1
source "${REPO_ROOT}/scripts/_python_env.sh"

if [[ -z "${TMUX:-}" ]]; then
    echo "ERROR: run this long-lived completion queue from an attached tmux session" >&2
    exit 1
fi

CONFIG="configs/004_ProMoE_B_expert_contra_param_cos.yaml"
LOG="${REPO_ROOT}/logs/expert_contra_param_completion_queue.log"
mkdir -p "$(dirname "$LOG")"

PYTHON="${PROMOE_TRAIN_PYTHON}"
PYTHON_EVAL="${PROMOE_EVAL_PYTHON}"
[[ -x "$PYTHON" ]] || {
    echo "ERROR: required training interpreter is missing: $PYTHON" >&2
    exit 1
}
[[ -x "$PYTHON_EVAL" ]] || {
    echo "ERROR: required evaluation interpreter is missing: $PYTHON_EVAL" >&2
    exit 1
}

configs=(
    configs/004_ProMoE_B_expert_contra_param_cos.yaml
    configs/004_ProMoE_B_expert_contra_param_shared.yaml
    configs/004_ProMoE_B_expert_contra_param_shared_uncond.yaml
    configs/004_ProMoE_B_expert_contra_param_tau0p07.yaml
    configs/004_ProMoE_B_expert_contra_param_tau7.yaml
)
scripts=(
    scripts/expert_contra/run_B_expert_contra_param_cos_train_sample_eval.sh
    scripts/expert_contra/run_B_expert_contra_param_shared_train_sample_eval.sh
    scripts/expert_contra/run_B_expert_contra_param_shared_uncond_train_sample_eval.sh
    scripts/expert_contra/run_B_expert_contra_param_tau0p07_train_sample_eval.sh
    scripts/expert_contra/run_B_expert_contra_param_tau7_train_sample_eval.sh
)

stems=(
    004_ProMoE_B_expert_contra_param_cos
    004_ProMoE_B_expert_contra_param_shared
    004_ProMoE_B_expert_contra_param_shared_uncond
    004_ProMoE_B_expert_contra_param_tau0p07
    004_ProMoE_B_expert_contra_param_tau7
)

# ``flock`` is released by the kernel even after SIGKILL or a host failure.
# The older hand-written queue does not use this lock, so active variants are
# also detected from their command lines below.
LOCK_FILE="/tmp/promoe-expert_contra_param_completion_queue.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date -Is)] Another completion queue owns ${LOCK_FILE}; exiting" | tee -a "$LOG"
    exit 0
fi

is_active_variant() {
    local stem=$1
    # The active legacy queue passes the stem in its temporary YAML path.
    ps -eo args= | grep -F "$stem" \
        | grep -E 'train.py|sample.py|run_B_expert_contra_param_' \
        | grep -v grep >/dev/null
}

has_fresh_scratch_marker() {
    local output_dir=$1
    local stem=$2
    local world_size=${3:-4}
    local training_log="${output_dir}/training.log"

    [[ -d "$output_dir" && ! -L "$output_dir" ]] || return 1
    [[ -f "$training_log" && ! -L "$training_log" ]] || return 1
    # The marker is written by train.py before the first optimizer step.  It
    # is the only reliable on-disk evidence that this bucket began without a
    # checkpoint; filenames alone cannot distinguish a fresh run from a
    # copied/resumed historical result.
    grep -Eq \
        "Fresh run marker: run_id=[A-Za-z0-9_-]+ fresh=True config=${stem} output_dir=.*${stem} global_seed=0 world_size=${world_size}([[:space:]]|$)" \
        "$training_log"
}

has_complete_eval() {
    local output_dir=$1
    local stem
    stem=$(basename "$output_dir")
    has_fresh_scratch_marker "$output_dir" "$stem" 4 || return 1

    local step file image_parent image_dir cfg step_dir
    local cfg1_count=0
    local cfg15_count=0
    local files=()
    [[ -d "$output_dir/checkpoints" && ! -L "$output_dir/checkpoints" ]] || return 1
    [[ -d "$output_dir/sample" && ! -L "$output_dir/sample" ]] || return 1
    for step in 300000 500000; do
        [[ -f "$output_dir/checkpoints/ckpt_step_${step}.pth" \
            && -s "$output_dir/checkpoints/ckpt_step_${step}.pth" \
            && ! -L "$output_dir/checkpoints/ckpt_step_${step}.pth" ]] || return 1
        step_dir="$output_dir/sample/step${step}"
        [[ -d "$step_dir" && ! -L "$step_dir" ]] || return 1
        mapfile -t files < <(
            find "$step_dir" -mindepth 2 -maxdepth 2 \
                -type f -name images_eval_openai.txt 2>/dev/null | sort
        )
        # Exactly one successful evaluator output per CFG value is required.
        # Counting only the total number of files is insufficient: two CFG 1.0
        # files can otherwise make an incomplete run look complete.
        if [[ "${#files[@]}" -ne 2 ]]; then
            return 1
        fi
        cfg1_count=0
        cfg15_count=0
        for file in "${files[@]}"; do
            grep -q '^FID:' "$file" || return 1
            grep -q '^Inception Score:' "$file" || return 1
            image_parent=$(dirname "$file")
            image_dir="$image_parent/images"
            [[ ! -L "$file" && ! -L "$image_parent" \
                && ! -L "$image_dir" \
                && -d "$image_dir" \
                && -s "$image_parent/images.npz" \
                && -f "$image_parent/images.npz" \
                && ! -L "$image_parent/images.npz" \
                && -n "$(find "$image_dir" -mindepth 1 -maxdepth 1 \
                    -type f -name '*.png' -print -quit 2>/dev/null)" ]] || return 1
            cfg=$(basename "$image_parent")
            case "$cfg" in
                *cfg1.0_*) ((cfg1_count += 1)) ;;
                *cfg1.5_*) ((cfg15_count += 1)) ;;
                *) return 1 ;;
            esac
        done
        [[ "$cfg1_count" -eq 1 && "$cfg15_count" -eq 1 ]] || return 1
    done
    return 0
}

wait_for_variant() {
    local stem=$1
    local last_notice=0
    while is_active_variant "$stem"; do
        local now
        now=$(date +%s)
        if (( now - last_notice >= 300 )); then
            echo "[$(date -Is)] WAIT_ACTIVE ${stem}; preserving the existing trajectory" | tee -a "$LOG"
            last_notice=$now
        fi
        sleep 60
    done
}

for cfg in "${configs[@]}"; do
    "$PYTHON" - "$cfg" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle)
if config.get("gpu_ids") != [0, 1, 2, 3]:
    raise SystemExit(f"{sys.argv[1]} must reserve gpu_ids [0, 1, 2, 3]")
if config.get("global_seed") != 0:
    raise SystemExit(f"{sys.argv[1]} must use global_seed 0")
if config.get("resume_checkpoint") is not False:
    raise SystemExit(f"{sys.argv[1]} must start from scratch")
if config.get("use_encoded_latents") is not True:
    raise SystemExit(f"{sys.argv[1]} must use pre-encoded latents")
PY
done

echo "[$(date -Is)] EXPERT_PARAM_COMPLETION_QUEUE_START gpu_ids=[0,1,2,3]" | tee -a "$LOG"
for i in "${!scripts[@]}"; do
    script="${scripts[$i]}"
    stem="${stems[$i]}"
    output_dir="${REPO_ROOT}/outputs/ProMoE_TC_B_expert_contra/${stem}"

    if is_active_variant "$stem"; then
        # A hand-written legacy queue may already own this variant. Wait for it
        # to finish, then either record its complete evaluation or report a
        # recoverable incomplete bucket; never launch a second fresh run.
        wait_for_variant "$stem"
    fi
    if [[ -d "$output_dir" ]] && has_complete_eval "$output_dir"; then
        echo "[$(date -Is)] SKIP_COMPLETE ${stem}" | tee -a "$LOG"
        continue
    fi
    if [[ -e "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "[$(date -Is)] ERROR_INCOMPLETE_NONEMPTY ${output_dir}; refusing to overwrite" | tee -a "$LOG" >&2
        exit 1
    fi

    echo "[$(date -Is)] QUEUE_START ${script}" | tee -a "$LOG"
    bash "$script" 2>&1 | tee -a "$LOG"
    if ! has_complete_eval "$output_dir"; then
        echo "[$(date -Is)] ERROR_INCOMPLETE_AFTER_RUN ${stem}; refusing to advance the queue" \
            | tee -a "$LOG" >&2
        exit 1
    fi
    echo "[$(date -Is)] QUEUE_DONE ${script}" | tee -a "$LOG"
done
echo "[$(date -Is)] EXPERT_PARAM_COMPLETION_QUEUE_DONE" | tee -a "$LOG"
