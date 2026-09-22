#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# claim_gpus.sh — pick N idle GPUs at launch time, whatever the machine holds.
#
# Run-time wrappers source this instead of hard-coding GPU ids, so the same
# script runs on a 4-GPU box and an 8-GPU box.  A GPU counts as idle when it
# reports zero used memory AND carries no live claim from another launch.
#
#   promoe_claim_gpus <count>        # sets PROMOE_CLAIMED_GPUS, e.g. "2,3"
#
# Call it directly, never through $( ): a command substitution runs the function
# in a subshell, whose EXIT trap releases the claim the moment it returns, so
# the claim would be gone before the trainer ever starts.  The result therefore
# comes back in PROMOE_CLAIMED_GPUS rather than on stdout.
#
# The claim file closes the gap between "we chose these GPUs" and "the trainer
# actually allocates memory on them", which takes a minute or so: a second
# launch in that window would otherwise see the same zeros and pick the same
# GPUs.  A claim is a file under PROMOE_GPU_CLAIM_DIR named after the GPU and
# holding the owning PID; it is released on exit, and a claim whose PID is gone
# is treated as stale and overwritten.  Selection runs under one flock, so two
# launches cannot interleave.
#
# Env:
#   PROMOE_GPU_CLAIM_DIR   claim directory        [default /tmp/promoe_gpu_claims]
#   PROMOE_GPU_EXCLUDE     comma-separated GPU ids never to pick   [default none]
#   PROMOE_GPU_IDLE_MAX_MIB  used memory still counted as idle       [default 100]
#
# The idle threshold is not zero because an unused H800 on this host reports a
# few MiB of driver residue with no compute process attached; a strict zero
# would permanently skip such a GPU.  Anything holding a real job sits in the
# GiB range, far above the default.
# ---------------------------------------------------------------------------

PROMOE_GPU_CLAIM_DIR="${PROMOE_GPU_CLAIM_DIR:-/tmp/promoe_gpu_claims}"

promoe_gpu_release() {
    local gpu
    for gpu in ${PROMOE_CLAIMED_GPUS//,/ }; do
        local claim="${PROMOE_GPU_CLAIM_DIR}/${gpu}"
        # Only remove our own claim, never one a later launch installed.
        if [[ -f "$claim" ]] && [[ "$(cat "$claim" 2>/dev/null)" == "$$" ]]; then
            rm -f "$claim"
        fi
    done
}

promoe_claim_gpus() {
    local want="$1"
    if ! [[ "$want" =~ ^[0-9]+$ ]] || [[ "$want" -lt 1 ]]; then
        echo "ERROR: GPU count must be a positive integer, got: ${want}" >&2
        return 2
    fi
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "ERROR: nvidia-smi not found; cannot pick GPUs" >&2
        return 2
    }

    mkdir -p "$PROMOE_GPU_CLAIM_DIR"
    local lock="${PROMOE_GPU_CLAIM_DIR}/.lock"
    exec {lock_fd}>"$lock"
    flock "$lock_fd"

    local idle_max="${PROMOE_GPU_IDLE_MAX_MIB:-100}"
    local chosen=() report=""
    local excluded=",${PROMOE_GPU_EXCLUDE:-},"
    local line index used claim owner
    while IFS=, read -r index used; do
        index="${index// /}"; used="${used// /}"; used="${used%%MiB*}"
        local state="idle"
        [[ "$excluded" == *",${index},"* ]] && state="excluded"
        [[ "$state" == "idle" && "$used" -gt "$idle_max" ]] && state="${used}MiB used"
        if [[ "$state" == "idle" ]]; then
            claim="${PROMOE_GPU_CLAIM_DIR}/${index}"
            if [[ -f "$claim" ]]; then
                owner="$(cat "$claim" 2>/dev/null || true)"
                # A claim whose owner is gone is stale and does not block.
                if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
                    state="claimed by pid ${owner}"
                fi
            fi
        fi
        report+="  GPU ${index}: ${state}"$'\n'
        if [[ "$state" == "idle" && "${#chosen[@]}" -lt "$want" ]]; then
            chosen+=("$index")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    if [[ "${#chosen[@]}" -lt "$want" ]]; then
        echo "ERROR: need ${want} idle GPU(s), found ${#chosen[@]}." >&2
        printf '%s' "$report" >&2
        flock -u "$lock_fd"; exec {lock_fd}>&-
        return 1
    fi

    local gpu
    for gpu in "${chosen[@]}"; do
        echo "$$" > "${PROMOE_GPU_CLAIM_DIR}/${gpu}"
    done
    flock -u "$lock_fd"; exec {lock_fd}>&-

    PROMOE_CLAIMED_GPUS="$(IFS=,; echo "${chosen[*]}")"
    export PROMOE_CLAIMED_GPUS
    trap promoe_gpu_release EXIT
}
