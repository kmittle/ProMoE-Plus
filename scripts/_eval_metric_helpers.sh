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
