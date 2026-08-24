#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 LABEL REPOSITORY_ROOT < PROMPT" >&2
    echo "       $0 --cleanup RUN_DIRECTORY" >&2
}

cleanup_run_directory() {
    local run_directory=${1:?missing run directory}
    local resolved
    local run_basename
    local run_parent
    local window_id

    [[ -d "$run_directory" && ! -L "$run_directory" ]] || {
        echo "run directory is not a normal non-symlink directory: $run_directory" >&2
        return 2
    }
    resolved=$(realpath -- "$run_directory")
    [[ $run_directory == "$resolved" ]] || {
        echo "run directory must be the canonical launcher path: $run_directory" >&2
        return 2
    }
    run_parent=$(dirname -- "$resolved")
    run_basename=$(basename -- "$resolved")
    [[ $run_parent == "/tmp" \
        && $run_basename =~ ^promoe_plus_cc_[A-Za-z0-9._-]+\.[A-Za-z0-9]{6}$ ]] || {
        echo "refusing to remove unexpected path: $resolved" >&2
        return 2
    }
    [[ -O "$resolved" ]] || {
        echo "run directory is not owned by the current user: $resolved" >&2
        return 2
    }
    if [[ -f "$resolved/window_id" ]]; then
        window_id=$(<"$resolved/window_id")
        if [[ $window_id =~ ^@[0-9]+$ ]]; then
            tmux kill-window -t "$window_id" 2>/dev/null || true
        fi
    fi
    rm -rf -- "$resolved"
}

run_worker() {
    local repository_root=${1:?missing repository root}
    local run_directory=${2:?missing run directory}
    local prompt_file="$run_directory/prompt.txt"
    local findings_file="$run_directory/findings.txt"
    local bootstrap_log="$run_directory/bash-startup.log"
    local error_log="$run_directory/cc.stderr.log"
    local status_file="$run_directory/status"
    local status_tmp="$run_directory/status.tmp"
    local cc_status

    trap 'worker_status=$?; if [[ ! -e "$status_file" ]]; then printf "%s\n" "$worker_status" > "$status_tmp"; mv -- "$status_tmp" "$status_file"; fi' EXIT
    cd "$repository_root"
    set +e
    bash -ic 'cc-yolo-api --print --output-format text --no-session-persistence --tools "Read,Grep,Glob" --append-system-prompt "Act only as a report-only reviewer. Never modify, create, delete, rename, stage, or commit files." > "$1"' \
        promoe-plus-cc "$findings_file" < "$prompt_file" > "$bootstrap_log" 2> "$error_log"
    cc_status=$?
    set -e

    if [[ $cc_status -eq 0 && ! -s "$findings_file" ]]; then
        echo "cc-yolo-api returned no review text" >> "$error_log"
        cc_status=66
    fi

    printf '%s\n' "$cc_status" > "$status_tmp"
    mv -- "$status_tmp" "$status_file"
    trap - EXIT
    return "$cc_status"
}

if [[ ${1:-} == "--worker" ]]; then
    [[ $# -eq 3 ]] || {
        usage
        exit 2
    }
    run_worker "$2" "$3"
    exit $?
fi

if [[ ${1:-} == "--cleanup" ]]; then
    [[ $# -eq 2 ]] || {
        usage
        exit 2
    }
    cleanup_run_directory "$2"
    exit $?
fi

[[ $# -eq 2 ]] || {
    usage
    exit 2
}

label=$1
repository_root=$(realpath -- "$2")
[[ $label =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "label may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 2
}
[[ -d "$repository_root" \
    && -f "$repository_root/AGENTS.md" \
    && -f "$repository_root/train.py" \
    && -f "$repository_root/sample.py" \
    && -f "$repository_root/scripts/template.sh" ]] || {
    echo "invalid ProMoE-Plus repository root: $repository_root" >&2
    exit 2
}

[[ -n ${TMUX:-} ]] || {
    echo "not inside tmux - attach first" >&2
    exit 1
}

tmux_session=$(tmux display-message -p '#S')
umask 077
run_directory=$(mktemp -d "/tmp/promoe_plus_cc_${label}.XXXXXX")
trap 'launcher_status=$?; if [[ $launcher_status -ne 0 && -d ${run_directory:-} ]]; then cleanup_run_directory "$run_directory" || true; fi' EXIT

prompt_file="$run_directory/prompt.txt"
dd of="$prompt_file" status=none
[[ -s "$prompt_file" ]] || {
    cleanup_run_directory "$run_directory"
    echo "review prompt is empty" >&2
    exit 2
}

worker_path=$(realpath -- "$0")
printf -v worker_command '%q ' "$worker_path" --worker "$repository_root" "$run_directory"
window_name="cc-${label:0:24}"

if ! window_id=$(tmux new-window -d -P -F '#{window_id}' -t "$tmux_session" -n "$window_name" "$worker_command"); then
    cleanup_run_directory "$run_directory"
    exit 1
fi
printf '%s\n' "$window_id" > "$run_directory/window_id"

printf '%s\n' "$run_directory"
