#!/bin/bash
#
# Independent Base seed-1 train + sample + eval pipeline.
# The experiment definition lives on the analysis branch, while all model
# code executes from the fixed clean Base source used by the seed-0 control.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="/mnt/cubefs/caoboyuan/ProMoE-Plus"
EXPECTED_SOURCE_COMMIT="257d51af287ea93103d7b4cad5ecab9dc1e3b541"
EXPERIMENT_BRANCH="analysis/long-horizon-routing"

CONFIG="configs/004_ProMoE_B_seed1_control.yaml"
CONFIG_ABS="${EXPERIMENT_ROOT}/${CONFIG}"
LOG="${SOURCE_ROOT}/logs/log_ProMoE_B_seed1_control_train_sample_eval.log"
PYTHON="/mnt/workspace/yujie/.conda/envs/promoe/bin/python"
PYTHON_EVAL="/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[ -x "$PYTHON" ] || fail "training interpreter is missing: ${PYTHON}"
[ -x "$PYTHON_EVAL" ] || fail "evaluation interpreter is missing: ${PYTHON_EVAL}"

SOURCE_HEAD="$(git -C "$SOURCE_ROOT" rev-parse 'HEAD^{commit}')"
SOURCE_ORIGIN="$(git -C "$SOURCE_ROOT" rev-parse 'refs/remotes/origin/repa^{commit}')"
[ "$SOURCE_HEAD" = "$EXPECTED_SOURCE_COMMIT" ] || \
    fail "source HEAD ${SOURCE_HEAD} != ${EXPECTED_SOURCE_COMMIT}"
[ "$SOURCE_ORIGIN" = "$EXPECTED_SOURCE_COMMIT" ] || \
    fail "origin/repa ${SOURCE_ORIGIN} != ${EXPECTED_SOURCE_COMMIT}"
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || \
    fail "source worktree is not clean: ${SOURCE_ROOT}"

EXPERIMENT_HEAD="$(git -C "$EXPERIMENT_ROOT" rev-parse 'HEAD^{commit}')"
EXPERIMENT_REMOTE_HEAD="$(
    git -C "$EXPERIMENT_ROOT" ls-remote --heads origin \
        "refs/heads/${EXPERIMENT_BRANCH}" | awk 'NR == 1 {print $1}'
)"
[ "$EXPERIMENT_REMOTE_HEAD" = "$EXPERIMENT_HEAD" ] || \
    fail "experiment HEAD is not pushed to origin/${EXPERIMENT_BRANCH}"
for tracked_path in \
    "$CONFIG" \
    "scripts/fresh_routing/run_B_seed1_control_train_sample_eval.sh" \
    "scripts/_run_times/2026_08_29/1.2-B_seed1_control.sh"; do
    git -C "$EXPERIMENT_ROOT" ls-files --error-unmatch "$tracked_path" \
        >/dev/null 2>&1 || fail "experiment file is not tracked: ${tracked_path}"
    git -C "$EXPERIMENT_ROOT" diff --quiet HEAD -- "$tracked_path" || \
        fail "experiment file differs from pushed HEAD: ${tracked_path}"
done

cd "$SOURCE_ROOT"
mkdir -p "$(dirname "$LOG")"

readarray -t YAML_INFO < <("$PYTHON" - "$CONFIG_ABS" <<'PY'
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
eval_gpu = str(gpu_ids[0]) if isinstance(gpu_ids, list) and gpu_ids else "0"
custom_cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
step_list = cfg.get("step_list_for_sample", [])
step_str = ','.join(map(str, step_list)) if step_list else ""
orig_num_steps = int(cfg.get("num_steps", 0))
output_root = cfg.get("output_dir", "outputs")
sample_base = os.path.join(
    output_root,
    model_name,
    custom_cfg_name,
    "sample",
)

print(model_name)
print(custom_cfg_name)
print(num_fid_samples)
print(eval_gpu)
print(gpu_str)
print(step_str)
print(orig_num_steps)
print(sample_base)
PY
)

MODEL_NAME="${YAML_INFO[0]}"
CUSTOM_CFG_NAME="${YAML_INFO[1]}"
NUM_FID_SAMPLES="${YAML_INFO[2]}"
EVAL_GPU="${YAML_INFO[3]}"
GPU_IDS="${YAML_INFO[4]}"
STEP_LIST_STR="${YAML_INFO[5]}"
ORIG_NUM_STEPS="${YAML_INFO[6]}"
SAMPLE_BASE="${YAML_INFO[7]}"
if [[ "$SAMPLE_BASE" != /* ]]; then
    SAMPLE_BASE="${SOURCE_ROOT}/${SAMPLE_BASE}"
fi
OUTPUT_BUCKET="${SAMPLE_BASE%/sample}"

if [ -e "$OUTPUT_BUCKET" ] || [ -L "$OUTPUT_BUCKET" ]; then
    [ -d "$OUTPUT_BUCKET" ] && [ ! -L "$OUTPUT_BUCKET" ] || \
        fail "seed1 output bucket must be a real directory: ${OUTPUT_BUCKET}"
    [ -z "$(find "$OUTPUT_BUCKET" -mindepth 1 -maxdepth 1 -print -quit)" ] || \
        fail "seed1 must start from an empty output bucket: ${OUTPUT_BUCKET}"
fi

# train.py records and verifies the source, config, and CUDA environment in
# every checkpoint. It also rejects a non-empty bucket before the first step.
export PROMOE_STRICT_PROVENANCE=1

if [ -z "$STEP_LIST_STR" ]; then
    echo "ERROR: step_list_for_sample is empty or missing in ${CONFIG}" >&2
    exit 1
fi
IFS=',' read -ra ALL_STEPS <<< "$STEP_LIST_STR"
NUM_ALL_STEPS=${#ALL_STEPS[@]}

TEMP_DIR=$(mktemp -d)
TEMP_CONFIG="${TEMP_DIR}/$(basename "$CONFIG")"
trap 'rm -rf "$TEMP_DIR"' EXIT

sample_and_eval_step() {
    local step=$1
    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} started" | tee -a "$LOG"

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "$PYTHON" sample.py \
        --config "${CONFIG_ABS}" --step_list_for_sample "${step}" \
        >> "$LOG" 2>&1

    if [ -d "$SAMPLE_BASE" ]; then
        while IFS= read -r IMG_DIR; do
            echo "[$(date '+%H:%M:%S')] Evaluating: ${IMG_DIR}" | tee -a "$LOG"
            (cd evaluation && CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
                "$PYTHON_EVAL" run_eval.py "$IMG_DIR" --count "${NUM_FID_SAMPLES}") \
                >> "$LOG" 2>&1
        done < <(find "$SAMPLE_BASE" -mindepth 3 -maxdepth 3 \
            -path "*/step${step}/*" -type d -name images | sort -V)
    fi

    echo "[$(date '+%H:%M:%S')] Sample+eval step ${step} done" | tee -a "$LOG"
}

echo "============================================================" | tee "$LOG"
echo "Sequential pipeline: ${MODEL_NAME}" | tee -a "$LOG"
echo "Config: ${CONFIG_ABS}" | tee -a "$LOG"
echo "Source commit: ${EXPECTED_SOURCE_COMMIT}" | tee -a "$LOG"
echo "Steps: ${STEP_LIST_STR}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for i in "${!ALL_STEPS[@]}"; do
    step="${ALL_STEPS[$i]}"
    phase=$((i + 1))

    if [ "$phase" -lt "$NUM_ALL_STEPS" ]; then
        TARGET_NUM_STEPS=$((step + 1))
    else
        TARGET_NUM_STEPS="$ORIG_NUM_STEPS"
    fi

    "$PYTHON" - "$CONFIG_ABS" "$TARGET_NUM_STEPS" "$TEMP_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
cfg['num_steps'] = int(sys.argv[2])
cfg['resume_checkpoint'] = True
with open(sys.argv[3], 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PY

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

    echo "============================================================" | tee -a "$LOG"
    echo "Phase ${phase}/${NUM_ALL_STEPS}: Sample+eval step ${step}" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    sample_and_eval_step "$step"
done

echo "============================================================" | tee -a "$LOG"
echo "All done." | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
