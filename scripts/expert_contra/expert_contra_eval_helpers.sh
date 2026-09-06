#!/usr/bin/env bash

# Evaluate exactly the two CFG settings requested by the parameter-ablation
# configs and fail closed when sampling or the evaluator leaves incomplete
# output.  A wrapper must call this after sample.py returns successfully.
if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/scripts/_eval_metric_helpers.sh" ]]; then
    source "${REPO_ROOT}/scripts/_eval_metric_helpers.sh"
elif [[ -f "${PWD}/scripts/_eval_metric_helpers.sh" ]]; then
    source "${PWD}/scripts/_eval_metric_helpers.sh"
else
    SCRIPT_DIR_EXPERT_CONTRA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "${SCRIPT_DIR_EXPERT_CONTRA}/../_eval_metric_helpers.sh"
    unset SCRIPT_DIR_EXPERT_CONTRA
fi

expert_contra_eval_step() {
    local sample_base=$1
    local step=$2
    local log=$3
    local eval_gpu=$4
    local python_eval=$5
    local num_fid_samples=$6
    local -a image_dirs=()
    local image_dir image_parent eval_file npz_file
    local cfg1_count=0 cfg15_count=0

    if [[ ! -d "$sample_base" ]]; then
        echo "ERROR: sample root not found: ${sample_base}" | tee -a "$log" >&2
        return 1
    fi
    mapfile -t image_dirs < <(
        find "$sample_base" -mindepth 3 -maxdepth 3 \
            -path "*/step${step}/*" -type d -name images | sort -V
    )
    if [[ "${#image_dirs[@]}" -ne 2 ]]; then
        echo "ERROR: expected exactly two CFG image directories for step ${step}, found ${#image_dirs[@]}" \
            | tee -a "$log" >&2
        return 1
    fi

    for image_dir in "${image_dirs[@]}"; do
        image_parent=$(dirname "$image_dir")
        case "$(basename "$image_parent")" in
            *_cfg1.0_*) cfg1_count=$((cfg1_count + 1)) ;;
            *_cfg1.5_*) cfg15_count=$((cfg15_count + 1)) ;;
            *)
                echo "ERROR: unexpected CFG image directory: ${image_parent}" | tee -a "$log" >&2
                return 1
                ;;
        esac

        if [[ -z "$(find "$image_dir" -mindepth 1 -maxdepth 1 -type f -name '*.png' -print -quit)" ]]; then
            echo "ERROR: no PNG images found in ${image_dir}" | tee -a "$log" >&2
            return 1
        fi
        echo "[$(date '+%H:%M:%S')] Evaluating: ${image_dir}" | tee -a "$log"
        if ! (cd evaluation && CUDA_VISIBLE_DEVICES="${eval_gpu}" \
            "$python_eval" run_eval.py "$image_dir" --count "${num_fid_samples}") \
            >> "$log" 2>&1; then
            echo "ERROR: evaluator failed for ${image_dir}" | tee -a "$log" >&2
            return 1
        fi
        eval_file="${image_parent}/images_eval_openai.txt"
        npz_file="${image_parent}/images.npz"
        if [[ -L "$image_dir" || -L "$eval_file" || -L "$npz_file" \
            || ! -s "$npz_file" || ! -s "$eval_file" ]] \
            || ! promoe_eval_file_metrics_valid "$eval_file"; then
            echo "ERROR: evaluator produced incomplete results for ${image_dir}" | tee -a "$log" >&2
            return 1
        fi
    done

    if [[ "$cfg1_count" -ne 1 || "$cfg15_count" -ne 1 ]]; then
        echo "ERROR: expected one CFG 1.0 and one CFG 1.5 result for step ${step}" \
            | tee -a "$log" >&2
        return 1
    fi
}

# Every fresh candidate must first complete both 300K CFG evaluations.  Only
# a candidate that beats the same fresh ProMoE-TC control on both FIDs may
# spend the additional time and storage required for a 500K continuation.
expert_contra_check_300k_gate() {
    local sample_base=$1
    local step=$2
    local log=$3
    local baseline_cfg1="30.584602064850174"
    local baseline_cfg15="9.588081719517504"
    local f1="${sample_base}/step${step}/img256_cfg1.0_seed0_FID50K_bs128_ema/images_eval_openai.txt"
    local f15="${sample_base}/step${step}/img256_cfg1.5_seed0_FID50K_bs128_ema/images_eval_openai.txt"
    local fid1 fid15

    if ! promoe_eval_file_metrics_valid "$f1" \
        || ! promoe_eval_file_metrics_valid "$f15"; then
        echo "ERROR: 300K gate could not validate both evaluator records" | tee -a "$log" >&2
        return 2
    fi
    fid1="$(promoe_eval_file_fid "$f1")"
    fid15="$(promoe_eval_file_fid "$f15")"
    if ! promoe_metric_is_finite_nonnegative "$fid1" \
        || ! promoe_metric_is_finite_nonnegative "$fid15"; then
        echo "ERROR: 300K gate could not read both FID values" | tee -a "$log" >&2
        return 2
    fi

    echo "[$(date '+%H:%M:%S')] 300K gate: CFG1.0=${fid1} CFG1.5=${fid15} baseline=${baseline_cfg1}/${baseline_cfg15}" \
        | tee -a "$log"
    if awk -v a="$fid1" -v b="$baseline_cfg1" \
        -v c="$fid15" -v d="$baseline_cfg15" \
        'BEGIN { exit !(a < b && c < d) }'; then
        echo "[$(date '+%H:%M:%S')] 300K gate PASS; allowing 500K continuation" | tee -a "$log"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] 300K gate FAIL; retaining 300K output and stopping before 500K" \
        | tee -a "$log"
    return 1
}
