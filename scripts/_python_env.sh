#!/usr/bin/env bash
# Canonical interpreters for experiment-server runs.
#
# The /mnt/workspace paths are the required deployment paths.  A different
# interpreter is accepted only when a local caller explicitly opts in with
# PROMOE_ALLOW_LOCAL_FALLBACK=1; ordinary experiment wrappers fail before
# touching an output bucket when the pinned environment is unavailable.

PROMOE_TRAIN_PYTHON="/mnt/workspace/yujie/.conda/envs/promoe/bin/python"
PROMOE_EVAL_PYTHON="/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python"

if [[ ! -x "$PROMOE_TRAIN_PYTHON" || ! -x "$PROMOE_EVAL_PYTHON" ]]; then
    if [[ "${PROMOE_ALLOW_LOCAL_FALLBACK:-0}" != "1" ]]; then
        echo "ERROR: required experiment-server interpreters are unavailable:" >&2
        echo "  $PROMOE_TRAIN_PYTHON" >&2
        echo "  $PROMOE_EVAL_PYTHON" >&2
        echo "Set PROMOE_ALLOW_LOCAL_FALLBACK=1 only for an explicitly audited local deployment." >&2
        return 1 2>/dev/null || exit 1
    fi
    local_train="/home/dev/miniforge3/envs/promoe/bin/python"
    local_eval="/home/dev/miniforge3/envs/fid_eval/bin/python"
    if [[ ! -x "$local_train" || ! -x "$local_eval" ]]; then
        echo "ERROR: explicit local fallback was requested but its interpreters are unavailable." >&2
        return 1 2>/dev/null || exit 1
    fi
    PROMOE_TRAIN_PYTHON="$local_train"
    PROMOE_EVAL_PYTHON="$local_eval"
    echo "WARNING: using explicitly opted-in local interpreters: $PROMOE_TRAIN_PYTHON and $PROMOE_EVAL_PYTHON" >&2
fi
