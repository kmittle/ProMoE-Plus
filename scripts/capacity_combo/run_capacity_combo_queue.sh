#!/usr/bin/env bash
# Run the prepared four-point combination ablation in paired 4-GPU slots.
#
# The queue itself does not train in the current shell.  It waits for the
# historical-result completion queues, then creates one tmux window per
# experiment and advances to the next pair only after both wrappers finish.
# This keeps GPU 0-3 and GPU 4-7 occupied without using shell background jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PROMOE_ALLOW_LOCAL_FALLBACK=1
source "${REPO_ROOT}/scripts/_python_env.sh"

if [[ -z "${TMUX:-}" ]]; then
    echo "ERROR: run this queue from an attached tmux session" >&2
    exit 1
fi

SESSION="$(tmux display-message -p '#S')"
LOG="${REPO_ROOT}/logs/capacity_combo_queue.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

# The combination queue must not outrun the historical-result write-up.  The
# marker is intentionally outside the repository: it is created only after
# the missing rows in the result table have been filled from real evaluator
# files, and it is invalidated automatically when the table contents change.
HISTORY_TABLE="${REPO_ROOT}/_previous_results/results_all_experiments_2026_06_21_to_08_05.md"
HISTORY_GATE="/tmp/promoe-history-results-complete.marker"

# Keep one supervisor per repository.  The lock is kernel-owned, so a killed
# supervisor cannot leave a stale PID that blocks a later run.
LOCK_FILE="/tmp/promoe-capacity_combo_queue.lock"
exec 8>"$LOCK_FILE"
if ! flock -n 8; then
    echo "[$(date -Is)] Another capacity-combo supervisor owns ${LOCK_FILE}; exiting"
    exit 0
fi
RUN_ID="$(date +%Y%m%dT%H%M%S)-$$"

has_eval_pair() {
    local output_dir=$1
    local stem
    stem=$(basename "$output_dir")
    [[ -d "$output_dir" && ! -L "$output_dir" ]] || return 1
    [[ -f "$output_dir/training.log" && ! -L "$output_dir/training.log" ]] || return 1
    grep -Eq \
        "Fresh run marker: run_id=[A-Za-z0-9_-]+ fresh=True config=${stem} output_dir=.*${stem} global_seed=0 world_size=4([[:space:]]|$)" \
        "$output_dir/training.log" || return 1

    local step file image_parent image_dir step_dir
    local files=()
    local seen_cfg1=0
    local seen_cfg15=0
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
        [[ "${#files[@]}" -eq 2 ]] || return 1
        seen_cfg1=0
        seen_cfg15=0
        for file in "${files[@]}"; do
            grep -q '^FID:' "$file" || return 1
            grep -q '^Inception Score:' "$file" || return 1
            image_parent=$(dirname "$file")
            image_dir="$image_parent/images"
            [[ ! -L "$file" && ! -L "$image_parent" \
                && ! -L "$image_dir" \
                && -d "$image_dir" \
                && -f "$image_parent/images.npz" && ! -L "$image_parent/images.npz" \
                && -s "$image_parent/images.npz" ]] || return 1
            [[ -n "$(find "$image_dir" -mindepth 1 -maxdepth 1 \
                -type f -name '*.png' -print -quit 2>/dev/null)" ]] || return 1
            case "$(basename "$image_parent")" in
                *_cfg1.0_*) ((seen_cfg1 += 1)) ;;
                *_cfg1.5_*) ((seen_cfg15 += 1)) ;;
                *) return 1 ;;
            esac
        done
        [[ "$seen_cfg1" -eq 1 && "$seen_cfg15" -eq 1 ]] || return 1
    done
    return 0
}

has_param_completion() {
    local stem
    for stem in \
        004_ProMoE_B_expert_contra_param_cos \
        004_ProMoE_B_expert_contra_param_shared \
        004_ProMoE_B_expert_contra_param_shared_uncond \
        004_ProMoE_B_expert_contra_param_tau0p07 \
        004_ProMoE_B_expert_contra_param_tau7; do
        has_eval_pair "${REPO_ROOT}/outputs/ProMoE_TC_B_expert_contra/${stem}" || return 1
    done
    return 0
}

has_adepth_completion() {
    local suffix
    for suffix in q0p1 q0p2 q0p3 q0p4; do
        has_eval_pair "${REPO_ROOT}/outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_${suffix}" || return 1
    done
    return 0
}

history_table_gate_valid() {
    [[ -s "$HISTORY_TABLE" && -s "$HISTORY_GATE" ]] || return 1

    local table_hash gated_hash row filled name
    table_hash="$(sha256sum "$HISTORY_TABLE" | awk '{print $1}')"
    gated_hash="$(sed -n 's/^table_sha256=//p' "$HISTORY_GATE" | head -1)"
    [[ -n "$table_hash" && "$table_hash" == "$gated_hash" ]] || return 1

    # The nine rows are the exact missing entries in the historical table.
    # A row is considered filled only when none of its four metric cells is
    # the em-dash placeholder.  The complete output/evaluator checks above
    # remain the authoritative source for the numbers themselves.
    for name in \
        B_expert_contra_param_cos \
        B_expert_contra_param_shared \
        B_expert_contra_param_shared_uncond \
        B_expert_contra_param_tau0p07 \
        B_expert_contra_param_tau7 \
        B_adepth_q0p1 B_adepth_q0p2 B_adepth_q0p3 B_adepth_q0p4; do
        # The same experiment name also appears in the status section above
        # the numeric result table.  Select a row with four numeric FID/IS
        # pairs so a prose status row cannot accidentally open the gate.
        filled="$(grep -F "| \`${name}\` |" "$HISTORY_TABLE" | awk -F'|' '
            {
                pairs = 0
                for (i = 1; i <= NF; ++i) {
                    if ($i ~ /^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*\/[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/)
                        ++pairs
                }
                if (pairs >= 4) {
                    print
                    exit
                }
            }
        ' || true)"
        [[ -n "$filled" ]] || return 1
    done
    return 0
}

wait_for_history_table() {
    echo "[$(date -Is)] Waiting for the historical result table to be filled"
    local last_notice=0 now
    while ! history_table_gate_valid; do
        now="$(date +%s)"
        if (( now - last_notice >= 300 )); then
            echo "[$(date -Is)] Historical table gate is not valid; no combination will launch"
            last_notice="$now"
        fi
        sleep 60
    done
    echo "[$(date -Is)] Historical result table gate validated"
}

# Upstream queues use temporary YAML files, so the training process command
# contains the experiment stem rather than the semantic wrapper name.  Keep
# this probe independent of grep's own command line and accept both forms.
param_queue_active() {
    local config_re='(^|[[:space:]/])004_ProMoE_B_expert_contra_param_(shared_uncond|cos|shared|tau0p07|tau7)([.]yaml)?([[:space:]]|$)'
    ps -eo args= | awk -v config_re="$config_re" '
        # Do not count this awk probe (its program text contains the same
        # experiment expressions) as an active upstream process.
        /(^|[[:space:]/])(awk|gawk|mawk)([[:space:]]|$)/ { next }
        /expert_contra_param_(completion_)?queue/ || /run_B_expert_contra_param_/ {
            found = 1
            next
        }
        ($0 ~ /(^|[[:space:]/])train[.]py([[:space:]]|$)/ ||
         $0 ~ /(^|[[:space:]/])sample[.]py([[:space:]]|$)/) && $0 ~ config_re {
            found = 1
        }
        END { exit(found ? 0 : 1) }
    '
}

# The loss-free follow-up queue has appeared under both the historical
# `lossfree_adepth` name and the combined `lossfree_expert_and_adepth` name.
# Match both so a valid upstream run cannot be declared idle while it trains.
adepth_queue_active() {
    ps -eo args= | awk '
        # The probe command itself contains `adepth_queue` in its
        # program text; exclude it before testing the activity expressions.
        /(^|[[:space:]/])(awk|gawk|mawk)([[:space:]]|$)/ { next }
        /promoe_after_(param_adepth|lossfree(_expert)?(_and)?_adepth)/ ||
        /run_B_adepth_q0p[1-4]_train_sample_eval/ ||
        /adepth_queue/ {
            found = 1
        }
        END { exit(found ? 0 : 1) }
    '
}

wait_for_param_completion() {
    echo "[$(date -Is)] Waiting for the parameter-ablation queue to finish"
    local idle_since
    idle_since="$(date +%s)"
    while ! has_param_completion; do
        # A finished queue with incomplete outputs is an error, rather than an
        # invitation to wait forever.  The output checks remain authoritative;
        # log markers are only used to identify this failure state.
        if { grep -Fq "EXPERT_PARAM_QUEUE_DONE" \
                "${REPO_ROOT}/logs/expert_param_queue.log" 2>/dev/null \
            || grep -Fq "EXPERT_PARAM_COMPLETION_QUEUE_DONE" \
                "${REPO_ROOT}/logs/expert_contra_param_completion_queue.log" 2>/dev/null; } \
            && ! param_queue_active; then
            echo "ERROR: parameter queue marker exists but one evaluation is incomplete" >&2
            return 1
        fi
        if ! param_queue_active; then
            local now
            now="$(date +%s)"
            if (( now - idle_since >= 1800 )); then
                echo "ERROR: parameter queue is no longer running and outputs are incomplete" >&2
                return 1
            fi
        else
            idle_since="$(date +%s)"
        fi
        sleep 60
    done
    echo "[$(date -Is)] Parameter-ablation outputs have complete 300K/500K evaluations"
}

wait_for_adepth_completion() {
    echo "[$(date -Is)] Waiting for all four adaptive-depth evaluations to finish"
    local idle_since
    idle_since="$(date +%s)"
    while ! has_adepth_completion; do
        if grep -Fq "Adepth GPU 0-3 queue complete" \
                "${REPO_ROOT}/logs/after_param_adepth.log" 2>/dev/null \
            || grep -Fq "ADEPTH_QUEUE_4_7_DONE" \
                "${REPO_ROOT}/logs/adepth_queue_4_7.log" 2>/dev/null; then
            if ! adepth_queue_active; then
                echo "ERROR: adaptive-depth queue marker exists but one evaluation is incomplete" >&2
                return 1
            fi
        fi
        # Waiting is expected while the two upstream queues train.  If both
        # queues disappear without a completion marker, surface the failure
        # after a grace period instead of hanging indefinitely.
        if ! adepth_queue_active; then
            local now
            now="$(date +%s)"
            if (( now - idle_since >= 1800 )); then
                echo "ERROR: adaptive-depth queues are no longer running and outputs are incomplete" >&2
                return 1
            fi
        else
            idle_since="$(date +%s)"
        fi
        sleep 60
    done
    echo "[$(date -Is)] Adaptive-depth outputs have complete 300K/500K evaluations"
}

wait_for_param_completion
wait_for_adepth_completion
wait_for_history_table

# The helper keeps the experiment-server interpreter paths as the defaults and
# uses an explicit deployment fallback only when that mount is unavailable.
TRAIN_PYTHON="${PROMOE_TRAIN_PYTHON}"
EVAL_PYTHON="${PROMOE_EVAL_PYTHON}"
[[ -x "$TRAIN_PYTHON" ]] || {
    echo "ERROR: required training interpreter is missing: $TRAIN_PYTHON" >&2
    exit 1
}
[[ -x "$EVAL_PYTHON" ]] || {
    echo "ERROR: required evaluation interpreter is missing: $EVAL_PYTHON" >&2
    exit 1
}

# The two completion queues are the expected owners of all eight GPUs, but a
# stale launcher or a manually started evaluator can outlive its completion
# marker.  Do not let a combination pair claim the cards until the device
# manager reports no compute process on any card.  A failed nvidia-smi query is
# treated as busy so a transient driver problem cannot turn into an overlap.
gpu_compute_active() {
    local processes
    if command -v nvidia-smi >/dev/null 2>&1; then
        if ! processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
            return 0
        fi
        [[ -n "${processes//[[:space:]]/}" ]]
        return
    fi
    ps -eo args= | awk '
        /(^|[[:space:]/])(awk|gawk|mawk)([[:space:]]|$)/ { next }
        /(^|[[:space:]/])train[.]py([[:space:]]|$)/ ||
        /(^|[[:space:]/])sample[.]py([[:space:]]|$)/ ||
        /(^|[[:space:]/])run_eval[.]py([[:space:]]|$)/ { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

wait_for_gpu_idle() {
    local last_notice=0
    while gpu_compute_active; do
        local now
        now="$(date +%s)"
        if (( now - last_notice >= 300 )); then
            echo "[$(date -Is)] Waiting for all eight GPUs to become idle before capacity-combo launch"
            last_notice="$now"
        fi
        sleep 60
    done
    echo "[$(date -Is)] All eight GPUs are idle; capacity-combo launch may proceed"
}

run_pair() {
    local left_wrapper=$1
    local right_wrapper=$2
    local left_name=$3
    local right_name=$4
    local left_window="${left_name}-${RUN_ID}"
    local right_window="${right_name}-${RUN_ID}"
    local left_marker="/tmp/promoe-capacity-${RUN_ID}-${left_name}.status"
    local right_marker="/tmp/promoe-capacity-${RUN_ID}-${right_name}.status"

    wait_for_gpu_idle

    window_exists() {
        tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null \
            | grep -Fxq "$1"
    }

    # Run-specific names and markers prevent stale files or an older supervisor
    # from being mistaken for this pair.  Refuse an existing window rather than
    # touching a process that may belong to the user.
    if window_exists "$left_window" || window_exists "$right_window"; then
        echo "ERROR: a capacity-combo window already exists for ${RUN_ID}" >&2
        return 1
    fi
    rm -f "$left_marker" "$right_marker"
    echo "[$(date -Is)] Launching pair ${left_name} (GPU 0-3) + ${right_name} (GPU 4-7)"

    tmux new-window -d -t "$SESSION" -n "$left_window" \
        "cd '$REPO_ROOT' && export PROMOE_ALLOW_LOCAL_FALLBACK=1 && bash '$REPO_ROOT/$left_wrapper'; rc=\$?; printf '%s\\n' \"\$rc\" > '$left_marker'; exit \"\$rc\""
    if ! tmux new-window -d -t "$SESSION" -n "$right_window" \
        "cd '$REPO_ROOT' && export PROMOE_ALLOW_LOCAL_FALLBACK=1 && bash '$REPO_ROOT/$right_wrapper'; rc=\$?; printf '%s\\n' \"\$rc\" > '$right_marker'; exit \"\$rc\""; then
        echo "ERROR: failed to create ${right_window}; left pair remains running for inspection" >&2
        return 1
    fi

    while [[ ! -s "$left_marker" || ! -s "$right_marker" ]]; do
        if [[ ! -s "$left_marker" ]] && ! window_exists "$left_window"; then
            echo "ERROR: ${left_window} disappeared without a status marker" >&2
            return 1
        fi
        if [[ ! -s "$right_marker" ]] && ! window_exists "$right_window"; then
            echo "ERROR: ${right_window} disappeared without a status marker" >&2
            return 1
        fi
        sleep 60
    done
    local left_rc right_rc
    left_rc="$(tr -d '[:space:]' < "$left_marker")"
    right_rc="$(tr -d '[:space:]' < "$right_marker")"
    echo "[$(date -Is)] Pair finished: ${left_name} rc=${left_rc}, ${right_name} rc=${right_rc}"
    if [[ "$left_rc" != 0 || "$right_rc" != 0 ]]; then
        echo "ERROR: a capacity-combo wrapper failed; leaving all outputs intact" >&2
        return 1
    fi
    # A wrapper's exit code is necessary but not sufficient: older template
    # scripts could report success after finding no image directory.  Validate
    # both buckets before allowing the next pair to claim the GPUs.
    local left_output right_output
    left_output="${REPO_ROOT}/outputs/ProMoE_TC_B_capacity_combo/004_ProMoE_B_capacity_combo_${left_name#capacity-}"
    right_output="${REPO_ROOT}/outputs/ProMoE_TC_B_capacity_combo/004_ProMoE_B_capacity_combo_${right_name#capacity-}"
    if ! has_eval_pair "$left_output" || ! has_eval_pair "$right_output"; then
        echo "ERROR: pair ${left_name}/${right_name} exited cleanly but lacks complete evaluations" >&2
        return 1
    fi
}

run_pair \
    scripts/_run_times/2026_09_04/1.1-B_capacity_combo_H.sh \
    scripts/_run_times/2026_09_04/1.2-B_capacity_combo_HO.sh \
    capacity-H capacity-HO
run_pair \
    scripts/_run_times/2026_09_04/2.1-B_capacity_combo_HP.sh \
    scripts/_run_times/2026_09_04/2.2-B_capacity_combo_HOP.sh \
    capacity-HP capacity-HOP
run_pair \
    scripts/_run_times/2026_09_04/3.1-B_capacity_combo_HR.sh \
    scripts/_run_times/2026_09_04/3.2-B_capacity_combo_HRO.sh \
    capacity-HR capacity-HRO
run_pair \
    scripts/_run_times/2026_09_04/4.1-B_capacity_combo_HRP.sh \
    scripts/_run_times/2026_09_04/4.2-B_capacity_combo_HROP.sh \
    capacity-HRP capacity-HROP

echo "[$(date -Is)] CAPACITY_COMBO_QUEUE_DONE"
