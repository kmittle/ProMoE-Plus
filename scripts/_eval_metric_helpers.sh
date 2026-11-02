#!/usr/bin/env bash

# The OpenAI evaluator writes human-readable metric lines.  Keep the queue
# predicates strict: a malformed value must never be interpreted by awk as
# zero and mistaken for a successful (or ordinary failed) scientific gate.
promoe_metric_is_finite_nonnegative() {
    local value=${1-}

    # Accept ordinary decimal and scientific notation, including values such
    # as .5, while rejecting signs below zero, NaN, Inf, and arbitrary text.
    [[ "$value" =~ ^[+]?[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$ \
        || "$value" =~ ^[+]?[.][0-9]+([eE][+-]?[0-9]+)?$ ]] || return 1

    # The regex does not catch numeric overflow (for example 1e309).  awk's
    # finite-range comparison does, without relying on a non-portable
    # isfinite() extension.
    awk -v value="$value" '
        BEGIN {
            number = value + 0
            exit !(number == number && number >= 0 && number <= 1e308)
        }
    '
}

promoe_eval_file_metrics_valid() {
    local eval_file=$1
    local metrics fid inception

    [[ -f "$eval_file" && ! -L "$eval_file" && -s "$eval_file" ]] || return 1
    metrics="$(awk '
        BEGIN { fid_count = 0; inception_count = 0; malformed = 0 }
        # Once a line uses the evaluator FID label, it must be a complete
        # metric record.  Do not silently ignore malformed labels such as
        # `FID:` or `FID:foo` when another valid-looking line is present.
        /^FID:/ {
            ++fid_count
            if ($0 !~ /^FID:[[:space:]]+[^[:space:]]+[[:space:]]*$/ || NF != 2)
                malformed = 1
            else fid = $2
        }
        /^Inception[[:space:]]+Score:/ {
            ++inception_count
            if ($0 !~ /^Inception[[:space:]]+Score:[[:space:]]+[^[:space:]]+[[:space:]]*$/ || NF != 3)
                malformed = 1
            else inception = $3
        }
        END {
            if (fid_count != 1 || inception_count != 1 || malformed) exit 1
            print fid "\t" inception
        }
    ' "$eval_file" 2>/dev/null)" || return 1
    IFS=$'\t' read -r fid inception <<< "$metrics"
    promoe_metric_is_finite_nonnegative "$fid" || return 1
    promoe_metric_is_finite_nonnegative "$inception"
}

promoe_eval_file_fid() {
    local eval_file=$1
    awk '/^FID:/{print $2; exit}' "$eval_file" 2>/dev/null || true
}

# Canonical ProMoE-TC B gate. Return 0 only when BOTH 300K FIDs improve;
# 1 is an ordinary scientific stop, 2 means missing/invalid/ambiguous results.
# Callers opt in explicitly, so existing ungated pipelines are unaffected.
promoe_300k_fid_gate() {
    local sample_base=$1 scale fid limit failed=0
    local files=()
    for scale in 1.0 1.5; do
        files=()
        mapfile -t files < <(find "${sample_base}/step300000" -mindepth 2 -maxdepth 2 \
            -path "*_cfg${scale}_*/images_eval_openai.txt" -print 2>/dev/null || true)
        if [[ ${#files[@]} -ne 1 ]]; then
            echo "ERROR: 300K gate requires exactly one CFG=${scale} evaluation; found ${#files[@]}" >&2
            return 2
        fi
        if ! promoe_eval_file_metrics_valid "${files[0]}"; then
            echo "ERROR: 300K gate has missing or invalid FID/IS for CFG=${scale}: ${files[0]}" >&2
            return 2
        fi
        fid="$(promoe_eval_file_fid "${files[0]}")"
        if [[ "$scale" == "1.0" ]]; then limit=30.86; else limit=9.73; fi
        echo "300K baseline comparison: CFG=${scale} FID=${fid}; required FID < ${limit}"
        if ! awk -v fid="$fid" -v limit="$limit" 'BEGIN { exit !(fid + 0 < limit + 0) }'; then
            failed=1
        fi
    done
    if [[ "$failed" == "1" ]]; then
        echo "300K gate STOP: both CFG FIDs must strictly beat the baseline; no continuation to 500K."
        return 1
    fi
    echo "300K gate PASS: both CFG FIDs strictly beat the baseline; continuation to 500K is allowed."
}
