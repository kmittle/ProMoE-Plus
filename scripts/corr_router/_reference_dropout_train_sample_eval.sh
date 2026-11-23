#!/usr/bin/env bash
# Shared template pipeline for the four 2026-11-23 reference-dropout training arms.
# The caller supplies CONFIG, LOG and REPO_ROOT. Existing launchers are unchanged.
cd "$REPO_ROOT"
source "${REPO_ROOT}/scripts/_python_env.sh"
source "${REPO_ROOT}/scripts/_eval_metric_helpers.sh"

mkdir -p "$(dirname "$LOG")"

# Deployment default; also overwrite the temporary YAML so an older config's
# latent_data_path cannot silently override this environment setting.
export PROMOE_LATENT_PATH="${PROMOE_LATENT_PATH:-/mnt/data2/yujie/code/Dataset/ImageNet/sd-vae-ft-mse_Latents_256img_npz}"

RESUME="${PROMOE_RESUME:-0}"
PIN_RESUME="${PROMOE_PIN_RESUME_CHECKPOINT:-0}"
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "ERROR: PROMOE_RESUME must be 0 or 1, got: ${RESUME}" >&2
    exit 1
fi

PYTHON="${PROMOE_TRAIN_PYTHON}"
PYTHON_EVAL="${PROMOE_EVAL_PYTHON}"
if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} cannot import PyYAML; install it in the promoe environment" >&2
    exit 1
fi

readarray -t YAML_INFO < <("$PYTHON" - "$CONFIG" <<'PY'
import os
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

model_name = cfg.get("model_name")
if not model_name:
    raise ValueError(f"model_name not found in {cfg_path}")

num_fid_samples = int(cfg.get("num_fid_samples", 50000))
gpu_ids = cfg.get("gpu_ids", [0])
gpu_str = ','.join(map(str, gpu_ids)) if isinstance(gpu_ids, list) else "0"
eval_gpu = str(gpu_ids[0]) if isinstance(gpu_ids, list) and len(gpu_ids) > 0 else "0"
# GPUs claimed at launch replace the placeholder gpu_ids; the config and its
# output identity stay unchanged.  The count must match, because the world size
# sets the per-GPU batch and every per-rank statistic.
gpu_override = os.environ.get("PROMOE_GPU_IDS_OVERRIDE", "").strip()
if gpu_override:
    items = [item.strip() for item in gpu_override.split(",")]
    if not all(item.isdigit() for item in items):
        raise ValueError(f"PROMOE_GPU_IDS_OVERRIDE must contain integer GPU ids; got {gpu_override!r}")
    override_ids = [int(item) for item in items]
    if len(set(override_ids)) != len(override_ids):
        raise ValueError(f"PROMOE_GPU_IDS_OVERRIDE must be distinct GPU ids like 2,3; got {gpu_override!r}")
    if not isinstance(gpu_ids, list) or len(override_ids) != len(gpu_ids):
        raise ValueError(
            f"PROMOE_GPU_IDS_OVERRIDE={gpu_override} names {len(override_ids)} GPU(s) but "
            f"{cfg_path} gpu_ids {gpu_ids} fixes the world size")
    gpu_str = ','.join(map(str, override_ids))
    eval_gpu = str(override_ids[0])
custom_cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
step_list = cfg.get("step_list_for_sample", [])
step_str = ','.join(map(str, step_list)) if step_list else ""
orig_num_steps = int(cfg.get("num_steps", 0))
# Match sample.py's sampling_folder_name. Small-sample diagnostic reports
# must not stand in for the scheduled 50K evaluations.
guide_scales = cfg.get("guide_scale_list") or []
if cfg.get("corr_reference_velocity") != "cfg" or cfg.get("corr_reference_separate_output", False):
    raise ValueError("These training arms require CFG-fused references in the training output bucket")
sample_folders = ','.join(
    f"img{cfg['image_size']}_cfg{scale}_seed{cfg['global_seed']}_"
    f"FID{int(num_fid_samples / 1000)}K_bs{cfg['sample_batch_size']}_ema_corr_ref_cfg"
    for scale in guide_scales
)

print(model_name)
print(custom_cfg_name)
print(num_fid_samples)
print(eval_gpu)
print(gpu_str)
print(step_str)
print(orig_num_steps)
print(sample_folders)
PY
)
# The process substitution hides a Python failure from set -e; check the result.
if [[ ${#YAML_INFO[@]} -ne 8 ]]; then
    echo "ERROR: could not read ${CONFIG} (see the Python error above)" >&2
    exit 1
fi

MODEL_NAME="${YAML_INFO[0]}"
CUSTOM_CFG_NAME="${YAML_INFO[1]}"
NUM_FID_SAMPLES="${YAML_INFO[2]}"
EVAL_GPU="${YAML_INFO[3]}"
GPU_IDS="${YAML_INFO[4]}"
STEP_LIST_STR="${YAML_INFO[5]}"
ORIG_NUM_STEPS="${YAML_INFO[6]}"
SAMPLE_FOLDERS_STR="${YAML_INFO[7]}"
SAMPLE_BASE="${REPO_ROOT}/outputs/${MODEL_NAME}/${CUSTOM_CFG_NAME}/sample"
OUTPUT_BASE="${SAMPLE_BASE%/sample}"

# ── Output bucket: a fresh run starts empty; a resume needs its own checkpoints ─
if [[ -L "$OUTPUT_BASE" || ( -e "$OUTPUT_BASE" && ! -d "$OUTPUT_BASE" ) ]]; then
    echo "ERROR: output bucket must be a real directory: $OUTPUT_BASE" >&2
    exit 1
fi
if [[ "$RESUME" == "1" ]]; then
    if ! find "${OUTPUT_BASE}/checkpoints" -maxdepth 1 -name 'ckpt_step_*.pth' -print -quit 2>/dev/null | grep -q .; then
        echo "ERROR: PROMOE_RESUME=1 but this run has no checkpoint to continue: ${OUTPUT_BASE}/checkpoints" >&2
        exit 1
    fi
elif [[ -e "$OUTPUT_BASE" ]] && find "$OUTPUT_BASE" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: training-from-scratch output bucket is not empty: $OUTPUT_BASE" >&2
    echo "       Set PROMOE_RESUME=1 to continue this run from its own checkpoints." >&2
    exit 1
fi

# ── Parse step_list ──────────────────────────────────────────────────────────
if [ -z "$STEP_LIST_STR" ]; then
    echo "ERROR: step_list_for_sample is empty or missing in ${CONFIG}" >&2
    exit 1
fi
IFS=',' read -ra ALL_STEPS <<< "$STEP_LIST_STR"
NUM_ALL_STEPS=${#ALL_STEPS[@]}

# ── Temp config: same basename as CONFIG so custom_cfg_name is preserved ─────
TEMP_DIR=$(mktemp -d)
TEMP_CONFIG="${TEMP_DIR}/$(basename "$CONFIG")"
trap 'rm -rf "$TEMP_DIR"' EXIT

# ── Helper: has every CFG of a step a valid evaluation already? ─────────────
step_already_evaluated() {
    local step=$1 folder
    local folders=()
    [[ -n "$SAMPLE_FOLDERS_STR" ]] || return 1
    IFS=',' read -ra folders <<< "$SAMPLE_FOLDERS_STR"
    for folder in "${folders[@]}"; do
        promoe_eval_file_metrics_valid "${SAMPLE_BASE}/step${step}/${folder}/images_eval_openai.txt" || return 1
    done
}

# Recheck measured 300K results on fresh and resumed runs. A valid result
# that fails the scientific gate ends normally; malformed results are errors.
check_300k_gate_or_stop() {
    local gate_output gate_status=0
    if ! step_already_evaluated 300000; then
        echo "ERROR: missing or invalid configured 300K evaluations under ${SAMPLE_BASE}/step300000" | tee -a "$LOG" >&2
        return 2
    fi
    gate_output="$(promoe_300k_fid_gate "$SAMPLE_BASE" 2>&1)" || gate_status=$?
    printf '%s\n' "$gate_output" | tee -a "$LOG"
    if [[ "$gate_status" == "1" ]]; then
        exit 0
    fi
    return "$gate_status"
}

# ── Helper: record which code this phase runs ────────────────────────────────
# Without strict provenance nothing else records it, and a pull between two
# phases would silently change the second half of the run.  Never fatal.
log_code_version() {
    {
        echo "[$(date '+%H:%M:%S')] Code version (phase $1):"
        echo "  git HEAD: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
        if modified="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null)"; then
            echo "  modified tracked files: $(printf '%s' "$modified" | grep -c .)"
        else
            echo "  modified tracked files: unavailable"
        fi
        sha256sum ./*.py "$CONFIG" 2>/dev/null | sed 's/^/  /'
        echo "  models/*.py: $(cat models/*.py 2>/dev/null | sha256sum | cut -c1-64)"
        echo "  repa/*.py: $(cat repa/*.py 2>/dev/null | sha256sum | cut -c1-64)"
    } >> "$LOG" 2>&1 || true
}

# ── Helper: sample + eval one checkpoint step ────────────────────────────────
sample_and_eval_step() {
    local step=$1 sample_folder
    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} started" | tee -a "$LOG"

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "$PYTHON" sample.py \
        --config "${CONFIG}" --step_list_for_sample "${step}" \
        >> "$LOG" 2>&1

    if [ -d "$SAMPLE_BASE" ]; then
        while IFS= read -r IMG_DIR; do
            sample_folder="${IMG_DIR%/images}"
            [[ ",${SAMPLE_FOLDERS_STR}," == *",${sample_folder##*/},"* ]] || continue
            echo "[$(date '+%H:%M:%S')] Evaluating: ${IMG_DIR}" | tee -a "$LOG"
            (cd evaluation && CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
                "$PYTHON_EVAL" run_eval.py "$IMG_DIR" --count "${NUM_FID_SAMPLES}") \
                >> "$LOG" 2>&1
        done < <(find "$SAMPLE_BASE" -mindepth 3 -maxdepth 3 -path "*/step${step}/*" -type d -name images | sort -V)
    fi

    step_already_evaluated "$step" || {
        echo "ERROR: step ${step} has missing or invalid FID/IS; every requested CFG must finish before continuing" | tee -a "$LOG" >&2
        return 1
    }

    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} done" | tee -a "$LOG"
}

# ══════════════════════════════════════════════════════════════════════════════
# Sequential pipeline: train → stop → sample + eval → resume → ...
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$RESUME" == "1" ]]; then
    echo "============================================================" | tee -a "$LOG"
    echo "Resuming sequential pipeline: ${MODEL_NAME}" | tee -a "$LOG"
else
    echo "============================================================" | tee "$LOG"
    echo "Sequential pipeline: ${MODEL_NAME}" | tee -a "$LOG"
fi
echo "Config: ${CONFIG}" | tee -a "$LOG"
echo "Latents: ${PROMOE_LATENT_PATH}" | tee -a "$LOG"
echo "GPUs: ${GPU_IDS} (eval: ${EVAL_GPU})" | tee -a "$LOG"
echo "Steps: ${STEP_LIST_STR}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Once a later checkpoint exists, only the evaluations already on disk can
# authorize continuation. A run still at 300K may finish interrupted evaluation.
RESUMING_AFTER_GATE=0
if [[ "$RESUME" == 1 ]] && {
    [[ "$PIN_RESUME" == 1 ]] ||
        "$PYTHON" scripts/_resume_checkpoint_step.py "${OUTPUT_BASE}/checkpoints" 300001 "$ORIG_NUM_STEPS" >/dev/null 2>&1
}; then
    RESUMING_AFTER_GATE=1
fi

for i in "${!ALL_STEPS[@]}"; do
    step="${ALL_STEPS[$i]}"
    phase=$((i + 1))

    # The next phase checks the saved metrics even if its checkpoint exists.
    if [[ "$RESUMING_AFTER_GATE" == 1 && "$step" == 300000 ]]; then
        continue
    fi

    # This precedes the resume/skip branch: saved checkpoints cannot bypass
    # the 300K gate, even if only a later phase is requested in the YAML.
    if (( step > 300000 )); then
        check_300k_gate_or_stop
    fi

    # For intermediate steps: num_steps = step + 1 (train through step, save, exit).
    # For the final step: use original num_steps from config.
    if [ "$phase" -lt "$NUM_ALL_STEPS" ]; then
        TARGET_NUM_STEPS=$((step + 1))
    else
        TARGET_NUM_STEPS="$ORIG_NUM_STEPS"
    fi

    # Under PROMOE_RESUME=1 a phase whose checkpoint is already saved goes
    # straight to sample + eval.  Training it again would redo that step and,
    # with trainers that resume at the saved step, overwrite the checkpoint
    # that is being sampled.
    if [[ "$RESUME" == "1" && -f "${OUTPUT_BASE}/checkpoints/ckpt_step_${step}.pth" ]]; then
        if step_already_evaluated "$step"; then
            echo "Phase ${phase}/${NUM_ALL_STEPS}: step ${step} is trained and evaluated; skipping" | tee -a "$LOG"
        else
            echo "Phase ${phase}/${NUM_ALL_STEPS}: checkpoint for step ${step} exists; skipping training" | tee -a "$LOG"
            log_code_version "$phase"
            sample_and_eval_step "$step"
        fi
        continue
    fi
    log_code_version "$phase"

    # Generate the temp config: adjusted num_steps; gpu_ids replaced by the GPUs
    # claimed at launch, because train.py sets CUDA_VISIBLE_DEVICES and the world
    # size from the config's gpu_ids.  resume_checkpoint stays True in every
    # phase: the bucket check above already guarantees a fresh launch has no
    # checkpoint. Once the gate passes, pin a checkpoint from that phase or
    # later so the trainer cannot fall back below the evaluated step.
    resume_step=""
    if (( step > 300000 )) || [[ "$RESUME" == 1 ]]; then
        minimum_step=1
        if (( step > 300000 )); then
            minimum_step=300000
        fi
        resume_step="$("$PYTHON" scripts/_resume_checkpoint_step.py "${OUTPUT_BASE}/checkpoints" "$minimum_step" "$step")"
        echo "Resuming phase ${step} from checkpoint step ${resume_step}" | tee -a "$LOG"
    fi
    "$PYTHON" - "$CONFIG" "$TARGET_NUM_STEPS" "$TEMP_CONFIG" "$GPU_IDS" "$resume_step" <<'PY'
import os, sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
cfg['num_steps'] = int(sys.argv[2])
cfg['resume_checkpoint'] = True
cfg['gpu_ids'] = [int(item) for item in sys.argv[4].split(',')]
cfg['latent_data_path'] = os.environ['PROMOE_LATENT_PATH']
if sys.argv[5]:
    cfg['resume_checkpoint_step'] = int(sys.argv[5])
with open(sys.argv[3], 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PY

    # ── Train ─────────────────────────────────────────────────────────────────
    echo "============================================================" | tee -a "$LOG"
    echo "Phase ${phase}/${NUM_ALL_STEPS}: Train to step ${step} (num_steps=${TARGET_NUM_STEPS})" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "$PYTHON" train.py \
        --config "${TEMP_CONFIG}" \
        >> "$LOG" 2>&1
    TRAIN_RC=$?
    set -e

    if [ $TRAIN_RC -ne 0 ]; then
        echo "Training FAILED at phase ${phase} (exit code $TRAIN_RC)" | tee -a "$LOG"
        exit $TRAIN_RC
    fi
    echo "Phase ${phase} training completed successfully" | tee -a "$LOG"

    # ── Sample + eval ─────────────────────────────────────────────────────────
    echo "============================================================" | tee -a "$LOG"
    echo "Phase ${phase}/${NUM_ALL_STEPS}: Sample+eval step ${step}" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    sample_and_eval_step "$step"
done

echo "============================================================" | tee -a "$LOG"
echo "All done." | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
