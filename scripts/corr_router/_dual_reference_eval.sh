#!/usr/bin/env bash
# Sourced by the same-noise pipelines. Both evaluation experiments share one
# checkpoint and use identical standard internal directory names.
CORR_EVAL_MODES=(conditional cfg)
CORR_EVAL_CONFIGS=("${CONFIG%.yaml}_cond_ref.yaml" "${CONFIG%.yaml}_cfg_ref.yaml")
CORR_EVAL_OUTPUTS=("${OUTPUT_BASE}_cond_ref" "${OUTPUT_BASE}_cfg_ref")
IFS=',' read -ra CORR_EVAL_SCALES <<< "$GUIDE_SCALES_STR"

for reference_output in "${CORR_EVAL_OUTPUTS[@]}"; do
    if [[ -L "$reference_output" || ( -e "$reference_output" && ! -d "$reference_output" ) ]]; then
        echo "ERROR: evaluation output must be a real directory: $reference_output" >&2
        exit 1
    fi
    if [[ "$RESUME" == "0" && -d "$reference_output" ]] &&
            [[ -n "$(find "$reference_output" -mindepth 1 -print -quit)" ]]; then
        echo "ERROR: evaluation output is not empty: $reference_output; use PROMOE_RESUME=1" >&2
        exit 1
    fi
done
unset reference_output

corr_eval_images_dir() {
    local step=$1 index=$2 scale=$3
    printf '%s/sample/step%s/%s_cfg%s%s/images\n' \
        "${CORR_EVAL_OUTPUTS[$index]}" "$step" "$SAMPLE_FOLDER_PREFIX" "$scale" "$SAMPLE_FOLDER_SUFFIX"
}

reference_mode_evaluated() {
    local step=$1 index=$2 scale images
    for scale in "${CORR_EVAL_SCALES[@]}"; do
        images="$(corr_eval_images_dir "$step" "$index" "$scale")"
        promoe_eval_file_metrics_valid "${images}_eval_openai.txt" || return 1
    done
}

step_already_evaluated() {
    local step=$1 index
    local checkpoint="${OUTPUT_BASE}/checkpoints/ckpt_step_${step}.pth"
    [[ -f "$checkpoint" && -s "$checkpoint" && ! -L "$checkpoint" ]] || return 1
    for index in "${!CORR_EVAL_MODES[@]}"; do
        reference_mode_evaluated "$step" "$index" || return 1
    done
}

sample_and_eval_step() {
    local step=$1 index mode scale images IMG_DIR sample_base
    local checkpoint="${OUTPUT_BASE}/checkpoints/ckpt_step_${step}.pth"
    [[ -f "$checkpoint" && -s "$checkpoint" && ! -L "$checkpoint" ]] || {
        echo "ERROR: expected this run's nonempty checkpoint: $checkpoint" >&2
        return 1
    }
    for index in "${!CORR_EVAL_MODES[@]}"; do
        mode="${CORR_EVAL_MODES[$index]}"
        if [[ "$RESUME" == "1" ]] && reference_mode_evaluated "$step" "$index"; then
            echo "Step ${step}, reference=${mode}: already evaluated; skipping" | tee -a "$LOG"
            continue
        fi
        echo "Step ${step}, reference=${mode}: ${CORR_EVAL_OUTPUTS[$index]}" | tee -a "$LOG"
        # Separate processes use identical seed and CFG order. On resume the
        # sampler draws noise/labels before skipping existing PNGs.
        CUDA_VISIBLE_DEVICES="${GPU_IDS}" "$PYTHON" sample.py \
            --config "${CORR_EVAL_CONFIGS[$index]}" --step_list_for_sample "$step" \
            --guide_scale_list "$GUIDE_SCALES_STR" --num_fid_samples "$NUM_FID_SAMPLES" \
            --corr-reference-velocity "$mode" >> "$LOG" 2>&1

        sample_base="${CORR_EVAL_OUTPUTS[$index]}/sample"
        for scale in "${CORR_EVAL_SCALES[@]}"; do
            images="$(corr_eval_images_dir "$step" "$index" "$scale")"
            if [[ "$RESUME" == "1" ]] && promoe_eval_file_metrics_valid "${images}_eval_openai.txt"; then
                continue
            fi
            [[ -d "$images" && ! -L "$images" ]] || {
                echo "ERROR: sampling did not create the expected image directory: $images" >&2
                return 1
            }
            while IFS= read -r IMG_DIR; do
                [[ "$IMG_DIR" == "$images" ]] || continue
                echo "Evaluating: $IMG_DIR" | tee -a "$LOG"
                (cd evaluation && CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
                    "$PYTHON_EVAL" run_eval.py "$IMG_DIR" --count "$NUM_FID_SAMPLES") >> "$LOG" 2>&1
            done < <(find "$sample_base" -mindepth 3 -maxdepth 3 -path "*/step${step}/*" -type d -name images | sort -V)
            promoe_eval_file_metrics_valid "${images}_eval_openai.txt" || {
                echo "ERROR: missing or invalid FID/IS for step ${step}, reference=${mode}, CFG=${scale}" >&2
                return 1
            }
        done
    done
}
