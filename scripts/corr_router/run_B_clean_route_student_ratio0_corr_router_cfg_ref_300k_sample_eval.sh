#!/usr/bin/env bash
# Inference-only adaptation of scripts/template.sh: reuse the original 300K EMA
# checkpoint, sample with CFG-fused routing references, then run the evaluator.
# Run on the training server. PROMOE_RESUME=1 resumes this inference output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
source "${REPO_ROOT}/scripts/_python_env.sh"
source "${REPO_ROOT}/scripts/_eval_metric_helpers.sh"

CONFIG="configs/004_ProMoE_B_clean_route_student_ratio0_corr_router_cfg_ref_300k.yaml"
LOG="${REPO_ROOT}/logs/log_B_clean_route_student_ratio0_corr_router_cfg_ref_300k_sample_eval.log"
PYTHON="${PROMOE_TRAIN_PYTHON}"
PYTHON_EVAL="${PROMOE_EVAL_PYTHON}"
RESUME="${PROMOE_RESUME:-0}"
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "ERROR: PROMOE_RESUME must be 0 or 1" >&2
    exit 1
fi

readarray -t YAML_INFO < <("$PYTHON" - "$CONFIG" "$REPO_ROOT" <<'PY'
import os
from pathlib import Path
import sys
import yaml

config_path, repo_path = sys.argv[1:]
repo = Path(repo_path).resolve()
cfg = yaml.safe_load(Path(config_path).read_text())
if cfg.get("corr_reference_velocity") != "cfg":
    raise ValueError("This ablation requires corr_reference_velocity: cfg")
if cfg.get("step_list_for_sample") != [300000] or cfg.get("guide_scale_list") != [1.5]:
    raise ValueError("This inference experiment requires step 300000 and CFG 1.5")
if cfg.get("save_inception_features", False):
    raise ValueError("The evaluator requires PNG images, not saved Inception features")

gpu_ids = cfg["gpu_ids"]
if (not isinstance(gpu_ids, list) or len(gpu_ids) != 4
        or any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)):
    raise ValueError("gpu_ids must fix a four-GPU world size")
override = os.environ.get("PROMOE_GPU_IDS_OVERRIDE", "").strip()
if override:
    parts = [part.strip() for part in override.split(",")]
    if not all(part.isdigit() for part in parts):
        raise ValueError("PROMOE_GPU_IDS_OVERRIDE must contain integer GPU IDs")
    chosen = [int(part) for part in parts]
    if len(chosen) != len(gpu_ids) or len(set(chosen)) != len(chosen):
        raise ValueError("PROMOE_GPU_IDS_OVERRIDE must contain four distinct GPU IDs")
    gpu_ids = chosen

checkpoint_dir = (repo / cfg["sample_checkpoint_dir"]).resolve()
checkpoint = checkpoint_dir / "ckpt_step_300000.pth"
if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise FileNotFoundError(f"Original 300K checkpoint is missing or empty: {checkpoint}")
output = repo / cfg.get("output_dir", "outputs") / cfg["model_name"] / Path(config_path).stem
if not output.resolve().is_relative_to(repo / "outputs"):
    raise ValueError("Inference output must stay under the repository outputs directory")
if output.resolve() == checkpoint_dir.parent:
    raise ValueError("Inference output must be separate from the training output")
count = int(cfg["num_fid_samples"])
if count <= 0:
    raise ValueError("num_fid_samples must be positive")
folder = (f"img{cfg['image_size']}_cfg1.5_seed{cfg['global_seed']}_"
          f"FID{int(count / 1000)}K_bs{cfg['sample_batch_size']}_ema_corr_ref_cfg")
print(",".join(map(str, gpu_ids)))
print(gpu_ids[0])
print(output)
print(checkpoint_dir)
print(count)
print(output / "sample" / "step300000" / folder / "images")
PY
)
# Process substitution can hide the parser's exit status.
if [[ ${#YAML_INFO[@]} -ne 6 ]]; then
    echo "ERROR: could not validate ${CONFIG}; see the Python error above" >&2
    exit 1
fi
GPU_IDS="${YAML_INFO[0]}"
EVAL_GPU="${YAML_INFO[1]}"
OUTPUT_BASE="${YAML_INFO[2]}"
CHECKPOINT_DIR="${YAML_INFO[3]}"
NUM_FID_SAMPLES="${YAML_INFO[4]}"
IMAGES="${YAML_INFO[5]}"
SAMPLE_BASE="${OUTPUT_BASE}/sample"
CHECKPOINT="${CHECKPOINT_DIR}/ckpt_step_300000.pth"

mkdir -p "$(dirname "$LOG")"
exec 9>"${LOG}.lock"
flock -n 9 || { echo "ERROR: this inference experiment is already running" >&2; exit 1; }
if [[ -L "$OUTPUT_BASE" || ( -e "$OUTPUT_BASE" && ! -d "$OUTPUT_BASE" ) ]]; then
    echo "ERROR: output bucket must be a real directory: $OUTPUT_BASE" >&2
    exit 1
fi
if [[ "$RESUME" == "0" && -d "$OUTPUT_BASE" ]] &&
        [[ -n "$(find "$OUTPUT_BASE" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: output bucket is not empty; use PROMOE_RESUME=1 to resume this inference" >&2
    exit 1
fi
if [[ "$RESUME" == "1" ]] && promoe_eval_file_metrics_valid "${IMAGES}_eval_openai.txt"; then
    echo "Completed CFG-reference 300K evaluation already exists: ${IMAGES}_eval_openai.txt"
    exit 0
fi

TEMP_DIR=$(mktemp -d)
TEMP_CONFIG="${TEMP_DIR}/$(basename "$CONFIG")"
trap 'rm -rf "$TEMP_DIR"' EXIT
"$PYTHON" - "$CONFIG" "$TEMP_CONFIG" "$GPU_IDS" "$CHECKPOINT_DIR" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys
import yaml
source, target, gpu_ids, checkpoint_dir, repo = sys.argv[1:]
cfg = yaml.safe_load(Path(source).read_text())
cfg["sample_gpu_ids"] = [int(gpu) for gpu in gpu_ids.split(",")]
cfg["sample_checkpoint_dir"] = checkpoint_dir
cfg["output_dir"] = str(Path(repo) / cfg.get("output_dir", "outputs"))
Path(target).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

log_code_version() {
    {
        echo "[$(date '+%H:%M:%S')] Code version (300K CFG-reference inference):"
        echo "  git HEAD: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
        if modified="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null)"; then
            echo "  modified tracked files: $(printf '%s' "$modified" | grep -c .)"
        else
            echo "  modified tracked files: unavailable"
        fi
        sha256sum ./*.py "$CONFIG" "$TEMP_CONFIG" 2>/dev/null | sed 's/^/  /'
        echo "  models/*.py: $(cat models/*.py 2>/dev/null | sha256sum | cut -c1-64)"
        echo "  repa/*.py: $(cat repa/*.py 2>/dev/null | sha256sum | cut -c1-64)"
        stat --format='  checkpoint: %n; bytes=%s; modified=%y' "$CHECKPOINT"
    } >> "$LOG" 2>&1 || true
}
echo "Config: $CONFIG; checkpoint: $CHECKPOINT" | tee -a "$LOG"
echo "GPUs: $GPU_IDS; CFG: 1.5; reference: cfg; samples: $NUM_FID_SAMPLES" | tee -a "$LOG"
echo "Output: $OUTPUT_BASE" | tee -a "$LOG"
log_code_version

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" sample.py \
    --config "$TEMP_CONFIG" --step_list_for_sample 300000 \
    --guide_scale_list 1.5 --num_fid_samples "$NUM_FID_SAMPLES" \
    --corr-reference-velocity cfg >> "$LOG" 2>&1

[[ -d "$IMAGES" && ! -L "$IMAGES" ]] || {
    echo "ERROR: sampling did not create the expected image directory: $IMAGES" >&2
    exit 1
}
while IFS= read -r IMG_DIR; do
    [[ "$IMG_DIR" == "$IMAGES" ]] || continue
    echo "[$(date '+%H:%M:%S')] Evaluating: $IMG_DIR" | tee -a "$LOG"
    (cd evaluation && CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_EVAL" run_eval.py \
        "$IMG_DIR" --count "$NUM_FID_SAMPLES") >> "$LOG" 2>&1
done < <(find "$SAMPLE_BASE" -mindepth 3 -maxdepth 3 -path '*/step300000/*' -type d -name images | sort -V)
promoe_eval_file_metrics_valid "${IMAGES}_eval_openai.txt" || {
    echo "ERROR: evaluation did not produce valid FID/IS metrics" >&2
    exit 1
}
echo "[$(date '+%H:%M:%S')] Completed: ${IMAGES}_eval_openai.txt" | tee -a "$LOG"
