#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$BASH_SOURCE")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

CONFIG="configs/004_ProMoE_B_dino_route_margin_gate_v2_correct_s0.yaml"
LOG="$REPO_ROOT/logs/log_ProMoE_B_dino_route_margin_gate_v2_correct_s0_train_sample_eval.log"
mkdir -p "$(dirname "$LOG")"

if [ -x /mnt/workspace/yujie/.conda/envs/promoe/bin/python ]; then
    PYTHON=/mnt/workspace/yujie/.conda/envs/promoe/bin/python
    PYTHON_EVAL=/mnt/workspace/yujie/.conda/envs/fid_eval/bin/python
else
    PYTHON=/home/dev/miniforge3/envs/promoe/bin/python
    PYTHON_EVAL=/home/dev/miniforge3/envs/fid_eval/bin/python
fi

readarray -t YAML_INFO < <("$PYTHON" - "$CONFIG" <<'PY'
import os
import sys
import yaml

with open(sys.argv[1], "r") as handle:
    cfg = yaml.safe_load(handle)
model_name = cfg["model_name"]
output_root = os.path.expanduser(str(cfg.get("output_dir", "outputs")))
if not os.path.isabs(output_root):
    output_root = os.path.join(os.getcwd(), output_root)
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
print(output_root)
print(os.path.expanduser(str(dino.get("table_path", ""))))
PY
)

MODEL_NAME=${YAML_INFO[0]}
CUSTOM_CFG_NAME=${YAML_INFO[1]}
NUM_FID_SAMPLES=${YAML_INFO[2]}
EVAL_GPU=${YAML_INFO[3]}
GPU_IDS=${YAML_INFO[4]}
STEP_LIST_STR=${YAML_INFO[5]}
ORIG_NUM_STEPS=${YAML_INFO[6]}
OUTPUT_ROOT=${YAML_INFO[7]}
DINO_TABLE_PATH=${YAML_INFO[8]}
SAMPLE_BASE=$OUTPUT_ROOT/$MODEL_NAME/$CUSTOM_CFG_NAME/sample

if [ -z "$STEP_LIST_STR" ]; then
    echo "ERROR: step_list_for_sample is empty or missing in $CONFIG" >&2
    exit 1
fi
if [ ! -f "$DINO_TABLE_PATH" ] || [ ! -f "$DINO_TABLE_PATH.json" ]; then
    echo "ERROR: DINO route table and metadata are required: $DINO_TABLE_PATH" >&2
    exit 1
fi
if ! "$PYTHON" - "$DINO_TABLE_PATH" <<'PY'
import json
import sys
from pathlib import Path
from preprocess.dino_route_table_contract import CORRECTED_TABLE_METHOD, CORRECTED_TABLE_VERSION

metadata_path = Path(sys.argv[1] + ".json")
with metadata_path.open("r", encoding="utf-8") as handle:
    metadata = json.load(handle)
if (type(metadata.get("version")) is not int
        or metadata["version"] != CORRECTED_TABLE_VERSION
        or metadata.get("method") != CORRECTED_TABLE_METHOD):
    raise SystemExit(
        "ERROR: this experiment requires corrected v2 DINO metadata; "
        f"found version={metadata.get('version')!r}, method={metadata.get('method')!r}"
    )
PY
then
    exit 1
fi

TEMP_DIR=$(mktemp -d)
TEMP_CONFIG=$TEMP_DIR/$(basename "$CONFIG")
trap 'rm -rf "$TEMP_DIR"' EXIT

sample_and_eval_step() {
    local step=$1
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
echo "Output root: $OUTPUT_ROOT" | tee -a "$LOG"
echo "Steps: $STEP_LIST_STR" | tee -a "$LOG"
echo "DINO table: $DINO_TABLE_PATH" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

IFS=',' read -r STEP_300K STEP_500K <<< "$STEP_LIST_STR"
if [ "$STEP_300K" != "300000" ] || [ "$STEP_500K" != "500000" ]; then
    echo "ERROR: this wrapper requires step_list_for_sample [300000, 500000]" >&2
    exit 1
fi

# Phase 1: train from step 0 to 300K, then sample and evaluate.
"$PYTHON" - "$CONFIG" 300001 False "$TEMP_CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
cfg["num_steps"] = int(sys.argv[2])
cfg["resume_checkpoint"] = sys.argv[3] == "True"
with open(sys.argv[4], "w") as handle:
    yaml.dump(cfg, handle, default_flow_style=False, sort_keys=False)
PY
echo "Phase 1: Train from scratch to step 300000" | tee -a "$LOG"
set +e
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" train.py --config "$TEMP_CONFIG" >> "$LOG" 2>&1
TRAIN_RC=$?
set -e
if [ $TRAIN_RC -ne 0 ]; then
    echo "Training FAILED in phase 1 (exit code $TRAIN_RC)" | tee -a "$LOG"
    exit $TRAIN_RC
fi
sample_and_eval_step 300000

# Phase 2: resume that same scratch run to 500K, then sample and evaluate.
"$PYTHON" - "$CONFIG" "$ORIG_NUM_STEPS" True "$TEMP_CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
cfg["num_steps"] = int(sys.argv[2])
cfg["resume_checkpoint"] = sys.argv[3] == "True"
with open(sys.argv[4], "w") as handle:
    yaml.dump(cfg, handle, default_flow_style=False, sort_keys=False)
PY
echo "Phase 2: Resume training to step 500000" | tee -a "$LOG"
set +e
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" train.py --config "$TEMP_CONFIG" >> "$LOG" 2>&1
TRAIN_RC=$?
set -e
if [ $TRAIN_RC -ne 0 ]; then
    echo "Training FAILED in phase 2 (exit code $TRAIN_RC)" | tee -a "$LOG"
    exit $TRAIN_RC
fi
sample_and_eval_step 500000

echo "============================================================" | tee -a "$LOG"
echo "All done." | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
