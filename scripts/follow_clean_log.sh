#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 RAW_LOG CLEAN_LOG" >&2
    exit 2
fi

RAW_LOG="$1"
CLEAN_LOG="$2"

if [ ! -f "$RAW_LOG" ]; then
    echo "Raw log does not exist: $RAW_LOG" >&2
    exit 1
fi

mkdir -p "$(dirname "$CLEAN_LOG")"

if [ "$RAW_LOG" -ef "$CLEAN_LOG" ]; then
    echo "Raw and clean logs must be different files: $RAW_LOG" >&2
    exit 1
fi

# The training process may still hold RAW_LOG open, so clean it as a live
# stream instead of rewriting that file in place.
tail -c +1 -F "$RAW_LOG" \
    | perl -pe 'BEGIN { $| = 1 } s/\e\[[0-?]*[ -\/]*[@-~]//g' \
    > "$CLEAN_LOG"
