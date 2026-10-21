#!/usr/bin/env bash
# Re-evaluate an adaptive-depth 300K checkpoint with calibrated inference cutoffs.
#
# reeval_300k_inference.sh samples with the checkpoint's quota, which forces exactly q
# skipped and q deepened tokens at every sampling step, while a training batch mixes
# timesteps and the learned split moves with t. This job calibrates per-block score cutoffs
# on training-style batches (calibrate_depth_thresholds.py), samples 50K images per CFG with
# them, and runs the OpenAI evaluator on CPU. It keeps the 4-GPU world size and seeds of the
# quota re-evaluation, so both sample the same noise and labels. It never trains.
#
#   PROMOE_ALLOW_LOCAL_FALLBACK=1 bash scripts/adepth/reeval_300k_threshold.sh q0p1|q0p2|q0p3
#
# Run it inside tmux; it waits for four idle GPUs. Each phase appends the git head and the
# sha256 of the code it loads to <arm>.provenance.jsonl rather than locking those files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

ARM="${1:?Usage: reeval_300k_threshold.sh q0p1|q0p2|q0p3}"
case "$ARM" in q0p1|q0p2|q0p3) ;; *) echo "Invalid arm: $ARM" >&2; exit 2 ;; esac
test -n "${TMUX:-}" || { echo "Run inside tmux" >&2; exit 2; }

STEP=300000
JOB_ROOT="${REPO_ROOT}/outputs/ProMoE_TC_B_adepth/reeval_threshold_20261021"
mkdir -p "${JOB_ROOT}/configs"
exec 9>"${JOB_ROOT}/${ARM}.lock"
flock -n 9 || { echo "$ARM is already queued/running" >&2; exit 2; }
exec >>"${JOB_ROOT}/${ARM}.log" 2>&1

source "${REPO_ROOT}/scripts/_python_env.sh"
source "${REPO_ROOT}/scripts/_run_times/claim_gpus.sh"
source "${REPO_ROOT}/scripts/_eval_metric_helpers.sh"
PYTHON="${PROMOE_TRAIN_PYTHON}"
PYTHON_EVAL="${PROMOE_EVAL_PYTHON}"
STEM="004_ProMoE_B_adepth_${ARM}"
CONFIG="${REPO_ROOT}/configs/${STEM}.yaml"
CHECKPOINT="${REPO_ROOT}/outputs/ProMoE_TC_B_adepth/${STEM}/checkpoints/ckpt_step_${STEP}.pth"
# sample.py writes to outputs/<model_name>/<config stem>, so the runtime config's stem names
# a separate bucket; its checkpoints/ holds a symlink to the original checkpoint.
RUNTIME_CONFIG="${JOB_ROOT}/configs/${STEM}_thr.yaml"
RUN_DIR="${REPO_ROOT}/outputs/ProMoE_TC_B_adepth/${STEM}_thr"
CALIBRATION="${JOB_ROOT}/${ARM}.calibration.json"
VAE_DIR="${REPO_ROOT}/pretrained_ckpt/vae/stabilityai--sd-vae-ft-mse"
PROMOE_CLAIMED_GPUS=""

record_status() {
    "$PYTHON" - "$JOB_ROOT" "$ARM" "$1" "$$" "${PROMOE_CLAIMED_GPUS:-}" <<'PY'
import datetime, json, os, pathlib, sys
root, arm, state, pid, gpus = sys.argv[1:]
path = pathlib.Path(root) / (arm + ".status.json")
record = dict(arm=arm, state=state, pid=int(pid), gpus=gpus,
              updated_at=datetime.datetime.now().astimezone().isoformat())
temporary = path.with_suffix(".pending")
temporary.write_text(json.dumps(record, indent=2) + "\n")
os.replace(temporary, path)
PY
}

record_provenance() {
    "$PYTHON" - "$JOB_ROOT" "$ARM" "$1" "$REPO_ROOT" <<'PY'
import datetime, hashlib, json, pathlib, subprocess, sys
root, arm, phase, repo = sys.argv[1:]
files = ["models/models_ProMoE_TC_adepth.py", "models/modules.py", "sample.py", "config.py", "utils.py",
         "train.py", "evaluation/run_eval.py", "evaluation/evaluator.py",
         "scripts/adepth/calibrate_depth_thresholds.py", "scripts/adepth/reeval_300k_threshold.sh",
         f"configs/004_ProMoE_B_adepth_{arm}.yaml"]
def git(*args):
    try:
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
record = dict(phase=phase, at=datetime.datetime.now().astimezone().isoformat(),
              git_head=git("rev-parse", "HEAD").strip(),
              modified_tracked=[line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()],
              sha256={f: hashlib.sha256((pathlib.Path(repo) / f).read_bytes()).hexdigest() for f in files})
with (pathlib.Path(root) / (arm + ".provenance.jsonl")).open("a") as handle:
    handle.write(json.dumps(record) + "\n")
PY
}

cleanup() {
    local rc=$?
    promoe_gpu_release
    if [[ "$rc" -ne 0 ]]; then
        record_status "failed_exit_${rc}" || true
        echo "[$(date -Is)] FAILED $ARM rc=$rc"
    fi
}
trap cleanup EXIT

[[ -s "$CHECKPOINT" && -d "$VAE_DIR" ]]

if [[ -s "${JOB_ROOT}/${ARM}.results.json" ]]; then
    for scale in 1.0 1.5; do
        promoe_eval_file_metrics_valid \
            "${RUN_DIR}/sample/step${STEP}/img256_cfg${scale}_seed0_FID50K_bs128_ema/images_eval_openai.txt"
    done
    record_status completed
    exit 0
fi

record_status waiting_for_4_idle_gpus
echo "[$(date -Is)] Waiting for four idle GPUs: $ARM"
while true; do
    claim_rc=0
    promoe_claim_gpus 4 >"${JOB_ROOT}/${ARM}.gpu_wait.log" 2>&1 || claim_rc=$?
    if [[ "$claim_rc" -eq 0 ]]; then
        # The shared helper installs its own EXIT trap; retain job status reporting.
        trap cleanup EXIT
        break
    elif [[ "$claim_rc" -ne 1 ]]; then
        exit "$claim_rc"
    fi
    sleep 30
done
echo "[$(date -Is)] Claimed GPUs ${PROMOE_CLAIMED_GPUS}: $ARM"

if [[ ! -s "$CALIBRATION" ]]; then
    record_status calibrating
    record_provenance calibration
    CUDA_VISIBLE_DEVICES="${PROMOE_CLAIMED_GPUS%%,*}" "$PYTHON" scripts/adepth/calibrate_depth_thresholds.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --output "$CALIBRATION"
fi

"$PYTHON" - "$CONFIG" "$RUNTIME_CONFIG" "$PROMOE_CLAIMED_GPUS" "$REPO_ROOT" "$CALIBRATION" "$CHECKPOINT" "$STEP" <<'PY'
import json, pathlib, sys, yaml
source, target, gpus, repo, calibration, checkpoint, step = sys.argv[1:]
config = yaml.safe_load(pathlib.Path(source).read_text())
report = json.loads(pathlib.Path(calibration).read_text())
assert pathlib.Path(report["checkpoint"]).resolve() == pathlib.Path(checkpoint).resolve(), "calibrated another checkpoint"
assert pathlib.Path(report["config"]).resolve() == pathlib.Path(source).resolve(), "calibrated another config"
assert len(config["gpu_ids"]) == 4
assert config["global_seed"] == 0 and config["sample_batch_size"] == 128
assert config["sample_steps"] == 250 and config["DiT_B_config"]["qk_norm"] is False
config["output_dir"] = str(pathlib.Path(repo) / "outputs")
config["sample_gpu_ids"] = [int(x) for x in gpus.split(",")]
config["step_list_for_sample"] = [int(step)]
config["guide_scale_list"] = [1.0, 1.5]
config["num_fid_samples"] = config["save_img_num"] = 50000
config["save_inception_features"] = False
config["DiT_B_config"]["MoE_config"]["depth_infer_thresholds"] = report["depth_infer_thresholds"]
pathlib.Path(target).write_text(yaml.safe_dump(config, sort_keys=False))
PY

mkdir -p "${RUN_DIR}/checkpoints"
ln -sfn "../../${STEM}/checkpoints/ckpt_step_${STEP}.pth" "${RUN_DIR}/checkpoints/ckpt_step_${STEP}.pth"
[[ -s "${RUN_DIR}/checkpoints/ckpt_step_${STEP}.pth" ]]

record_status sampling
record_provenance sampling
CUDA_VISIBLE_DEVICES="$PROMOE_CLAIMED_GPUS" "$PYTHON" sample.py \
    --config "$RUNTIME_CONFIG" --step_list_for_sample "$STEP" \
    --guide_scale_list 1.0,1.5 --num_fid_samples 50000 --vae-path "$VAE_DIR"
grep -q "depth_infer_thresholds" "${RUN_DIR}/sample.log"  # the sampled model carried the cutoffs

# The evaluator runs on CPU (TensorFlow lacks H800 kernels). Release GPUs for the next
# arm, and serialize CPU evaluations so Inception jobs do not compete.
promoe_gpu_release
PROMOE_CLAIMED_GPUS=""
record_status waiting_for_evaluator
exec 8>"${JOB_ROOT}/evaluator.lock"
flock 8
record_status evaluating
record_provenance evaluation
for scale in 1.0 1.5; do
    images="${RUN_DIR}/sample/step${STEP}/img256_cfg${scale}_seed0_FID50K_bs128_ema/images"
    [[ -d "$images" && ! -L "$images" ]]
    CUDA_VISIBLE_DEVICES="" "$PYTHON_EVAL" evaluation/run_eval.py "$images" \
        --count 50000 --eval-device cpu
    promoe_eval_file_metrics_valid "${images}_eval_openai.txt"
done

"$PYTHON" - "$RUN_DIR" "$JOB_ROOT" "$ARM" "$STEP" "$CALIBRATION" <<'PY'
import datetime, hashlib, json, os, pathlib, re, sys
run, root, arm, step, calibration = sys.argv[1:]
results = {}
for scale in ("1.0", "1.5"):
    path = pathlib.Path(run) / f"sample/step{step}" / f"img256_cfg{scale}_seed0_FID50K_bs128_ema/images_eval_openai.txt"
    text = path.read_text()
    values = {}
    for metric in ("FID", "Inception Score", "sFID", "Precision", "Recall"):
        match = re.search(r"^" + re.escape(metric) + r":\s*([0-9.eE+-]+)\s*$", text, re.M)
        if not match:
            raise ValueError(f"Missing metric {metric}: {path}")
        values[metric] = float(match.group(1))
    results[scale] = dict(metrics=values, evaluator_file=str(path),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
report = json.loads(pathlib.Path(calibration).read_text())
record = dict(arm=arm, checkpoint_step=int(step), inference="calibrated_score_cutoffs",
              depth_infer_thresholds=report["depth_infer_thresholds"], calibration=calibration,
              completed_at=datetime.datetime.now().astimezone().isoformat(), results=results)
path = pathlib.Path(root) / (arm + ".results.json")
temporary = path.with_suffix(".pending")
temporary.write_text(json.dumps(record, indent=2) + "\n")
os.replace(temporary, path)
combined = {}
for candidate in ("q0p1", "q0p2", "q0p3"):
    result_path = pathlib.Path(root) / (candidate + ".results.json")
    if result_path.is_file():
        combined[candidate] = json.loads(result_path.read_text())
summary = dict(complete=len(combined) == 3, checkpoint_step=int(step),
               updated_at=record["completed_at"], arms=combined)
summary_path = pathlib.Path(root) / "results_summary.json"
temporary = summary_path.with_suffix(".pending")
temporary.write_text(json.dumps(summary, indent=2) + "\n")
os.replace(temporary, summary_path)
print(json.dumps(record, indent=2))
PY
flock -u 8
exec 8>&-
record_status completed
echo "[$(date -Is)] COMPLETED $ARM"
