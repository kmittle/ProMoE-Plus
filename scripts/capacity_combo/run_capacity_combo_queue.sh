#!/usr/bin/env bash
# Run the prepared four-point combination ablation in paired 4-GPU slots.
#
# The queue itself does not train in the current shell.  It waits for the
# historical-result completion queues, then creates one tmux window per
# experiment and advances to the next pair only after the preceding wrappers
# finish.  HROP is a single first-stage gate; the remaining arms launch only
# after HROP passes its 300K dual-FID test.  A failed 300K gate is a valid
# endpoint and does not make the queue wait for a nonexistent 500K result.
# This keeps GPU 0-3 and GPU 4-7 occupied without shell background jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PROMOE_ALLOW_LOCAL_FALLBACK=1
source "${REPO_ROOT}/scripts/_python_env.sh"
source "${REPO_ROOT}/scripts/_eval_metric_helpers.sh"

if [[ -z "${TMUX:-}" ]]; then
    echo "ERROR: run this queue from an attached tmux session" >&2
    exit 1
fi

SESSION="$(tmux display-message -p '#S')"
LOG="${REPO_ROOT}/logs/capacity_combo_queue.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

# The combination queue must not outrun the historical-result write-up.  The
# marker is intentionally outside the repository: this queue creates it only
# after the missing rows in the result table have been filled and the upstream
# output/evaluator checks have passed.  Its hash is invalidated automatically
# when the table contents change.
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

BASELINE_FID_CFG1="30.584602064850174"
BASELINE_FID_CFG15="9.588081719517504"

has_eval_step() {
    local output_dir=$1
    local step=$2
    local file image_parent image_dir step_dir
    local files=()
    local seen_cfg1=0
    local seen_cfg15=0

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
    for file in "${files[@]}"; do
        promoe_eval_file_metrics_valid "$file" || return 1
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
    [[ "$seen_cfg1" -eq 1 && "$seen_cfg15" -eq 1 ]]
}

has_gated_eval_pair() {
    local output_dir=$1
    local stem
    local fid1 fid15
    stem=$(basename "$output_dir")
    [[ -d "$output_dir" && ! -L "$output_dir" ]] || return 1
    [[ -f "$output_dir/training.log" && ! -L "$output_dir/training.log" ]] || return 1
    grep -Eq \
        "Fresh run marker: run_id=[A-Za-z0-9_-]+ fresh=True config=${stem} output_dir=.*${stem} global_seed=0 world_size=4([[:space:]]|$)" \
        "$output_dir/training.log" || return 1
    [[ -d "$output_dir/checkpoints" && ! -L "$output_dir/checkpoints" ]] || return 1
    [[ -d "$output_dir/sample" && ! -L "$output_dir/sample" ]] || return 1

    # Every candidate must have a complete 300K measurement.  A failed gate
    # is a valid scientific endpoint; only a passing gate requires 500K.
    has_eval_step "$output_dir" 300000 || return 1
    fid1="$(find "$output_dir/sample/step300000" -type f -name images_eval_openai.txt \
        -path '*_cfg1.0_*' -print -quit | xargs -r awk '/^FID:/{print $2; exit}')"
    fid15="$(find "$output_dir/sample/step300000" -type f -name images_eval_openai.txt \
        -path '*_cfg1.5_*' -print -quit | xargs -r awk '/^FID:/{print $2; exit}')"
    promoe_metric_is_finite_nonnegative "$fid1" || return 1
    promoe_metric_is_finite_nonnegative "$fid15" || return 1
    if awk -v a="$fid1" -v b="$BASELINE_FID_CFG1" \
        -v c="$fid15" -v d="$BASELINE_FID_CFG15" \
        'BEGIN { exit !(a < b && c < d) }'; then
        has_eval_step "$output_dir" 500000 || return 1
    elif [[ -e "$output_dir/checkpoints/ckpt_step_500000.pth" \
        || -d "$output_dir/sample/step500000" ]]; then
        # If a wrapper continued despite a failed gate, require that its
        # partial 500K artifacts are complete before treating the bucket as
        # reusable; never silently accept a half-written evaluation.
        has_eval_step "$output_dir" 500000 || return 1
    fi
    return 0
}

# Strictly complete pair, useful when a caller explicitly needs a 500K
# measurement.  The scheduling queues below intentionally use the gate-aware
# predicate because a failed 300K gate is a valid terminal result under the
# project rule and must not be retrained just to manufacture a 500K file.
has_full_eval_pair() {
    local output_dir=$1
    has_gated_eval_pair "$output_dir" || return 1
    has_eval_step "$output_dir" 500000
}

get_300k_fids() {
    local output_dir=$1
    local f1 f15 fid1 fid15
    f1="$(find "$output_dir/sample/step300000" -type f -name images_eval_openai.txt \
        -path '*_cfg1.0_*' -print -quit)"
    f15="$(find "$output_dir/sample/step300000" -type f -name images_eval_openai.txt \
        -path '*_cfg1.5_*' -print -quit)"
    promoe_eval_file_metrics_valid "$f1" || return 2
    promoe_eval_file_metrics_valid "$f15" || return 2
    fid1="$(promoe_eval_file_fid "$f1")"
    fid15="$(promoe_eval_file_fid "$f15")"
    printf '%s\n%s\n' "$fid1" "$fid15"
}

gate_passes() {
    local output_dir=$1
    local fids fid1 fid15
    fids="$(get_300k_fids "$output_dir")" || return $?
    fid1="$(printf '%s\n' "$fids" | sed -n '1p')"
    fid15="$(printf '%s\n' "$fids" | sed -n '2p')"
    awk -v a="$fid1" -v b="$BASELINE_FID_CFG1" \
        -v c="$fid15" -v d="$BASELINE_FID_CFG15" \
        'BEGIN { exit !(a < b && c < d) }'
}

capacity_process_active() {
    local name=$1
    local stem="${name#capacity-}"
    ps -eo args= | awk -v stem="$stem" '
        /(^|[[:space:]/])(awk|gawk|mawk)([[:space:]]|$)/ { next }
        index($0, "run_B_capacity_combo_" stem "_train_sample_eval.sh") {
            found = 1
            next
        }
        index($0, "B_capacity_combo_" stem ".sh") {
            found = 1
            next
        }
        index($0, "capacity_combo_" stem ".yaml") &&
        ($0 ~ /(^|[[:space:]/])(train|sample)[.]py([[:space:]]|$)/) {
            found = 1
            next
        }
        ($0 ~ ("ProMoE_B_capacity_combo_" stem "([._/[:space:]]|$)")) &&
        ($0 ~ /(^|[[:space:]/])run_eval[.]py([[:space:]]|$)/ ||
         $0 ~ /(^|[[:space:]/])sample[.]py([[:space:]]|$)/) { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

has_param_completion() {
    local stem
    for stem in \
        004_ProMoE_B_expert_contra_param_cos \
        004_ProMoE_B_expert_contra_param_shared \
        004_ProMoE_B_expert_contra_param_shared_uncond \
        004_ProMoE_B_expert_contra_param_tau0p07 \
        004_ProMoE_B_expert_contra_param_tau7; do
        # A 300K gate failure is a complete endpoint; do not require 500K.
        has_gated_eval_pair "${REPO_ROOT}/outputs/ProMoE_TC_B_expert_contra/${stem}" || return 1
    done
    return 0
}

has_adepth_completion() {
    local suffix
    for suffix in q0p1 q0p2 q0p3 q0p4; do
        # A 300K gate failure is a complete endpoint; do not require 500K.
        has_gated_eval_pair "${REPO_ROOT}/outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_${suffix}" || return 1
    done
    return 0
}

history_table_rows_complete() {
    [[ -s "$HISTORY_TABLE" ]] || return 1

    # The nine rows are the exact missing entries in the historical table.
    # A passing candidate must have all four metric cells.  A candidate that
    # failed the new 300K gate is allowed to remain 300K-only, but only when
    # a dedicated table row records both its two 300K FIDs and the
    # stopped/abandoned status.
    # This prevents the old historical 300K-only rows (which still say
    # "尚未启动" or "正在训练") from opening the gate prematurely.
    for name in \
        B_expert_contra_param_cos \
        B_expert_contra_param_shared \
        B_expert_contra_param_shared_uncond \
        B_expert_contra_param_tau0p07 \
        B_expert_contra_param_tau7 \
        B_adepth_q0p1 B_adepth_q0p2 B_adepth_q0p3 B_adepth_q0p4; do
        local numeric_row
        # Only a Markdown table row whose field is exactly this experiment
        # may open the gate.  Prose often mentions several experiments in one
        # sentence, so terminal words from those sentences must never be
        # associated with a different experiment's numeric row.
        numeric_row="$(awk -v expected="\`${name}\`" -F'|' '
            {
                owns_row = 0
                for (i = 1; i <= NF; ++i) {
                    field = $i
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", field)
                    if (field == expected) owns_row = 1
                }
                if (!owns_row) next
                pairs = 0
                scalar_values = 0
                for (i = 1; i <= NF; ++i) {
                    if ($i ~ /^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*[/][[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/)
                        ++pairs
                    if ($i ~ /^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/)
                        ++scalar_values
                }
                terminal = ($0 ~ /已停止|放弃|淘汰|不再续训|门禁[^；。]*失败|gate[^[:alpha:]]*(FAIL|failed)/)
                if (pairs >= 4) full = 1
                # The dedicated gate table stores two bare 300K FIDs, while
                # the main result table stores FID/IS pairs.  Accept either
                # shape only when the same row explicitly says it ended.
                if ((pairs >= 2 || scalar_values >= 2) && terminal)
                    partial_terminal = 1
            }
            END {
                if (full) print "full"
                else if (partial_terminal) print "partial_terminal"
            }
        ' "$HISTORY_TABLE")"
        [[ "$numeric_row" == "full" ]] && continue
        [[ "$numeric_row" == "partial_terminal" ]] || return 1
    done
    return 0
}

history_table_gate_valid() {
    history_table_rows_complete || return 1

    local table_hash gated_hash marker_tmp
    table_hash="$(sha256sum "$HISTORY_TABLE" | awk '{print $1}')"
    gated_hash="$(sed -n 's/^table_sha256=//p' "$HISTORY_GATE" 2>/dev/null | head -1)"
    if [[ -n "$table_hash" && "$table_hash" == "$gated_hash" ]]; then
        return 0
    fi

    # No manual step is required: write the marker atomically only after the
    # row check above succeeds.  A partial write can never validate the gate.
    marker_tmp="${HISTORY_GATE}.tmp.$$"
    if ! printf 'table_sha256=%s\n' "$table_hash" > "$marker_tmp"; then
        rm -f "$marker_tmp"
        return 1
    fi
    if ! mv -f "$marker_tmp" "$HISTORY_GATE"; then
        rm -f "$marker_tmp"
        return 1
    fi
    echo "[$(date -Is)] Wrote historical-result gate marker for current table hash"
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
    echo "[$(date -Is)] Parameter-ablation outputs have complete gated evaluations"
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
    echo "[$(date -Is)] Adaptive-depth outputs have complete gated evaluations"
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

run_single() {
    local wrapper=$1
    local name=$2
    local output_dir=$3
    local window_name="${name}-${RUN_ID}"
    local marker="/tmp/promoe-capacity-${RUN_ID}-${name}.status"

    window_exists() {
        tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null \
            | grep -Fxq "$1"
    }

    # A manually started capacity-combo run may already own the output bucket.
    # Reuse it after completion instead of launching a duplicate.
    if [[ -e "$output_dir" ]]; then
        if has_gated_eval_pair "$output_dir"; then
            echo "[$(date -Is)] Reusing completed ${name} output"
            return 0
        fi
        if capacity_process_active "$name"; then
            echo "[$(date -Is)] Waiting for existing ${name} process"
            while capacity_process_active "$name"; do sleep 60; done
            has_gated_eval_pair "$output_dir" || {
                echo "ERROR: existing ${name} ended without a complete gated result" >&2
                return 1
            }
            return 0
        fi
        echo "ERROR: ${name} output exists but is incomplete and no process is active: ${output_dir}" >&2
        return 1
    fi

    # The output directory can be created by a manually started process while
    # the queue is waiting for other GPUs.  Observe that process before and
    # after the idle wait; otherwise this queue could launch a duplicate.
    if capacity_process_active "$name"; then
        echo "[$(date -Is)] Waiting for existing ${name} process"
        while capacity_process_active "$name"; do sleep 60; done
        if has_gated_eval_pair "$output_dir"; then
            echo "[$(date -Is)] Reusing completed ${name} output"
            return 0
        fi
        echo "ERROR: existing ${name} process ended without a complete gated result" >&2
        return 1
    fi

    wait_for_gpu_idle
    if [[ -e "$output_dir" ]]; then
        if has_gated_eval_pair "$output_dir"; then
            echo "[$(date -Is)] Reusing completed ${name} output"
            return 0
        fi
        if capacity_process_active "$name"; then
            echo "[$(date -Is)] Existing ${name} appeared during GPU wait; waiting for it"
            while capacity_process_active "$name"; do sleep 60; done
            has_gated_eval_pair "$output_dir" || {
                echo "ERROR: existing ${name} ended without a complete gated result" >&2
                return 1
            }
            return 0
        fi
        echo "ERROR: ${name} output appeared during GPU wait but is incomplete: ${output_dir}" >&2
        return 1
    fi
    if capacity_process_active "$name"; then
        echo "ERROR: ${name} process appeared after idle wait without an output bucket" >&2
        return 1
    fi
    if window_exists "$window_name"; then
        echo "ERROR: capacity-combo window already exists: ${window_name}" >&2
        return 1
    fi
    rm -f "$marker"
    echo "[$(date -Is)] Launching ${name}"
    if ! tmux new-window -d -t "$SESSION" -n "$window_name" \
        "cd '$REPO_ROOT' && export PROMOE_ALLOW_LOCAL_FALLBACK=1 && bash '$REPO_ROOT/$wrapper'; rc=\$?; printf '%s\\n' \"\$rc\" > '$marker'; exit \"\$rc\""; then
        echo "ERROR: failed to create ${window_name}" >&2
        return 1
    fi

    while [[ ! -s "$marker" ]]; do
        if ! window_exists "$window_name"; then
            echo "ERROR: ${window_name} disappeared without a status marker" >&2
            return 1
        fi
        sleep 60
    done
    local rc
    rc="$(tr -d '[:space:]' < "$marker")"
    echo "[$(date -Is)] ${name} finished with rc=${rc}"
    if [[ "$rc" != 0 ]] || ! has_gated_eval_pair "$output_dir"; then
        echo "ERROR: ${name} did not produce a complete gated result" >&2
        return 1
    fi
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
    local left_output right_output
    local launch_left=1 launch_right=1

    left_output="${REPO_ROOT}/outputs/ProMoE_TC_B_capacity_combo/004_ProMoE_B_capacity_combo_${left_name#capacity-}"
    right_output="${REPO_ROOT}/outputs/ProMoE_TC_B_capacity_combo/004_ProMoE_B_capacity_combo_${right_name#capacity-}"

    window_exists() {
        tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null \
            | grep -Fxq "$1"
    }

    # Reconcile each side before claiming GPUs.  This makes a restarted queue
    # safe when one side already completed, or when a manually launched side is
    # still running; a non-empty incomplete bucket without a live process is a
    # hard error and is never overwritten.
    for side in left right; do
        if [[ "$side" == left ]]; then
            name="$left_name"
            output_dir="$left_output"
        else
            name="$right_name"
            output_dir="$right_output"
        fi
        if has_gated_eval_pair "$output_dir"; then
            echo "[$(date -Is)] Reusing completed ${name} output"
            if [[ "$side" == left ]]; then launch_left=0; else launch_right=0; fi
            continue
        fi
        if capacity_process_active "$name"; then
            echo "[$(date -Is)] Waiting for existing ${name} process"
            while capacity_process_active "$name"; do sleep 60; done
            if ! has_gated_eval_pair "$output_dir"; then
                echo "ERROR: existing ${name} ended without a complete gated result" >&2
                return 1
            fi
            if [[ "$side" == left ]]; then launch_left=0; else launch_right=0; fi
            continue
        fi
        if [[ -e "$output_dir" ]]; then
            echo "ERROR: ${name} output exists but is incomplete and no process is active: ${output_dir}" >&2
            return 1
        fi
    done

    if (( ! launch_left && ! launch_right )); then
        return 0
    fi

    wait_for_gpu_idle

    # Recheck after the idle wait because an external launcher can create a
    # bucket while this supervisor is waiting for unrelated GPUs.
    for side in left right; do
        if [[ "$side" == left ]]; then
            name="$left_name"
            output_dir="$left_output"
            should_launch=$launch_left
        else
            name="$right_name"
            output_dir="$right_output"
            should_launch=$launch_right
        fi
        (( should_launch )) || continue
        if has_gated_eval_pair "$output_dir"; then
            echo "[$(date -Is)] ${name} completed during GPU wait; reusing output"
            if [[ "$side" == left ]]; then launch_left=0; else launch_right=0; fi
            continue
        fi
        if capacity_process_active "$name"; then
            echo "ERROR: ${name} process appeared during GPU wait; refusing duplicate launch" >&2
            return 1
        fi
        if [[ -e "$output_dir" ]]; then
            echo "ERROR: ${name} output appeared during GPU wait but is incomplete: ${output_dir}" >&2
            return 1
        fi
    done

    (( ! launch_left && ! launch_right )) && return 0

    # Run-specific names and markers prevent stale files or an older supervisor
    # from being mistaken for this pair.  Refuse an existing window rather than
    # touching a process that may belong to the user.
    if window_exists "$left_window" || window_exists "$right_window"; then
        echo "ERROR: a capacity-combo window already exists for ${RUN_ID}" >&2
        return 1
    fi
    rm -f "$left_marker" "$right_marker"
    echo "[$(date -Is)] Launching pair sides: ${left_name}=${launch_left} (GPU 0-3), ${right_name}=${launch_right} (GPU 4-7)"

    if (( launch_left )); then
        if ! tmux new-window -d -t "$SESSION" -n "$left_window" \
            "cd '$REPO_ROOT' && export PROMOE_ALLOW_LOCAL_FALLBACK=1 && bash '$REPO_ROOT/$left_wrapper'; rc=\$?; printf '%s\\n' \"\$rc\" > '$left_marker'; exit \"\$rc\""; then
            echo "ERROR: failed to create ${left_window}" >&2
            return 1
        fi
    fi
    if (( launch_right )); then
        if ! tmux new-window -d -t "$SESSION" -n "$right_window" \
            "cd '$REPO_ROOT' && export PROMOE_ALLOW_LOCAL_FALLBACK=1 && bash '$REPO_ROOT/$right_wrapper'; rc=\$?; printf '%s\\n' \"\$rc\" > '$right_marker'; exit \"\$rc\""; then
            echo "ERROR: failed to create ${right_window}; any launched side remains running for inspection" >&2
            return 1
        fi
    fi

    while { (( launch_left )) && [[ ! -s "$left_marker" ]]; } \
        || { (( launch_right )) && [[ ! -s "$right_marker" ]]; }; do
        if (( launch_left )) && [[ ! -s "$left_marker" ]] && ! window_exists "$left_window"; then
            echo "ERROR: ${left_window} disappeared without a status marker" >&2
            return 1
        fi
        if (( launch_right )) && [[ ! -s "$right_marker" ]] && ! window_exists "$right_window"; then
            echo "ERROR: ${right_window} disappeared without a status marker" >&2
            return 1
        fi
        sleep 60
    done
    local left_rc=0 right_rc=0
    (( launch_left )) && left_rc="$(tr -d '[:space:]' < "$left_marker")"
    (( launch_right )) && right_rc="$(tr -d '[:space:]' < "$right_marker")"
    echo "[$(date -Is)] Pair finished: ${left_name} rc=${left_rc}, ${right_name} rc=${right_rc}"
    if [[ "$left_rc" != 0 || "$right_rc" != 0 ]]; then
        echo "ERROR: a capacity-combo wrapper failed; leaving all outputs intact" >&2
        return 1
    fi
    # A wrapper's exit code is necessary but not sufficient: older template
    # scripts could report success after finding no image directory.  Validate
    # both buckets before allowing the next pair to claim the GPUs.
    if ! has_gated_eval_pair "$left_output" || ! has_gated_eval_pair "$right_output"; then
        echo "ERROR: pair ${left_name}/${right_name} exited cleanly but lacks complete evaluations" >&2
        return 1
    fi
}

HROP_OUTPUT="${REPO_ROOT}/outputs/ProMoE_TC_B_capacity_combo/004_ProMoE_B_capacity_combo_HROP"
run_single \
    scripts/_run_times/2026_09_04/4.2-B_capacity_combo_HROP.sh \
    capacity-HROP "$HROP_OUTPUT"
gate_rc=0
gate_passes "$HROP_OUTPUT" || gate_rc=$?
if [[ "$gate_rc" -eq 1 ]]; then
    echo "[$(date -Is)] HROP failed the 300K dual-FID gate; stopping before other combination arms"
    exit 0
elif [[ "$gate_rc" -ne 0 ]]; then
    echo "ERROR: HROP 300K gate could not be validated (rc=${gate_rc})" >&2
    exit "$gate_rc"
fi
echo "[$(date -Is)] HROP passed the 300K dual-FID gate; launching remaining arms"

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
run_single \
    scripts/_run_times/2026_09_04/4.1-B_capacity_combo_HRP.sh \
    capacity-HRP \
    "${REPO_ROOT}/outputs/ProMoE_TC_B_capacity_combo/004_ProMoE_B_capacity_combo_HRP"

echo "[$(date -Is)] CAPACITY_COMBO_QUEUE_DONE"
