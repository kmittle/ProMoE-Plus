#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE[0]")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/004_ProMoE_B_dino_route_margin_gate_shuffled_s0.yaml"
LOG="$REPO_ROOT/logs/log_ProMoE_B_dino_route_margin_gate_shuffled_s0_train_sample_eval.log"
mkdir -p "$(dirname "$LOG")"

readarray -t YAML_INFO < <(python - "$CONFIG" <<'PY'
import os
import sys
import yaml

with open(sys.argv[1], "r") as handle:
    cfg = yaml.safe_load(handle)
model_name = cfg["model_name"]
gpu_ids = cfg.get("gpu_ids", [0])
steps = cfg.get("step_list_for_sample", [])
dino = cfg.get("DiT_B_config", {}).get("MoE_config", {}).get("dino_route_config", {})
print(model_name)
print(os.path.splitext(os.path.basename(sys.argv[1]))[0])
print(int(cfg.get("num_fid_samples", 50000)))
print(str(gpu_ids[0] if gpu_ids else 0))
print(",".join(map(str, gpu_ids)))
print(",".join(map(str, steps)))
print(int(cfg.get("num_steps", 0)))
print(os.path.expanduser(str(dino.get("table_path", ""))))
PY
)

MODEL_NAME="${YAML_INFO[0]}"
CUSTOM_CFG_NAME="${YAML_INFO[1]}"
NUM_FID_SAMPLES="${YAML_INFO[2]}"
EVAL_GPU="${YAML_INFO[3]}"
GPU_IDS="${YAML_INFO[4]}"
STEP_LIST_STR="${YAML_INFO[5]}"
ORIG_NUM_STEPS="${YAML_INFO[6]}"
DINO_TABLE_PATH="${YAML_INFO[7]}"
SAMPLE_BASE="$REPO_ROOT/outputs/$MODEL_NAME/$CUSTOM_CFG_NAME/sample"
PYTHON=/mnt/workspace/yujie/.conda/envs/promoe/bin/python
PYTHON_EVAL=/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python

test -n "$STEP_LIST_STR"
test -f "$DINO_TABLE_PATH"
test -f "$DINO_TABLE_PATH.json"
IFS=',' read -ra ALL_STEPS <<< "$STEP_LIST_STR"
NUM_ALL_STEPS=${#ALL_STEPS[@]}
TEMP_DIR=$(mktemp -d)
TEMP_CONFIG="$TEMP_DIR/$(basename "$CONFIG")"
trap 'rm -rf "$TEMP_DIR"' EXIT

sample_and_eval_step() {
    step=$1
    echo "[$(date '+%H:%M:%S')] Sample+eval step $step started" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" sample.py --config "$CONFIG" --step_list_for_sample "$step" >> "$LOG" 2>&1
    if [ -d "$SAMPLE_BASE" ]; then
        while IFS= read -r IMG_DIR; do
            echo "[$(date '+%H:%M:%S')] Evaluating: $IMG_DIR" | tee -a "$LOG"
            (cd evaluation && CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_EVAL" run_eval.py "$IMG_DIR" --count "$NUM_FID_SAMPLES") >> "$LOG" 2>&1
        done < <(find "$SAMPLE_BASE" -mindepth 3 -maxdepth 3 -path "*/step$step/*" -type d -name images | sort -V)
    fi
    echo "[$(date '+%H:%M:%S')] Sample+eval step $step done" | tee -a "$LOG"
}

echo "============================================================" | tee "$LOG"
echo "Sequential pipeline: $MODEL_NAME" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Steps: $STEP_LIST_STR" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for ((i=0; i<NUM_ALL_STEPS; i++)); do
    step="${ALL_STEPS[$i]}"
    phase=$((i + 1))
    if [ "$phase" -lt "$NUM_ALL_STEPS" ]; then
        TARGET_NUM_STEPS=$((step + 1))
        TARGET_RESUME=False
    else
        TARGET_NUM_STEPS=$ORIG_NUM_STEPS
        TARGET_RESUME=True
    fi
    python - "$CONFIG" "$TARGET_NUM_STEPS" "$TARGET_RESUME" "$TEMP_CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
cfg["num_steps"] = int(sys.argv[2])
cfg["resume_checkpoint"] = sys.argv[3] == "True"
with open(sys.argv[4], "w") as handle:
    yaml.dump(cfg, handle, default_flow_style=False, sort_keys=False)
PY
    echo "Phase $phase/$NUM_ALL_STEPS: Train to step $step (num_steps=$TARGET_NUM_STEPS)" | tee -a "$LOG"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" train.py --config "$TEMP_CONFIG" >> "$LOG" 2>&1
    TRAIN_RC=$?
    set -e
    if [ $TRAIN_RC -ne 0 ]; then
        echo "Training FAILED at phase $phase (exit code $TRAIN_RC)" | tee -a "$LOG"
        exit $TRAIN_RC
    fi
    sample_and_eval_step "$step"
done

echo "All done." | tee -a "$LOG"
