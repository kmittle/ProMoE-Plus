#!/usr/bin/env bash
# Re-evaluate an existing 300K checkpoint after the adaptive-depth inference fix.
# This entrypoint never starts or resumes training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

ARM="${1:?Usage: reeval_300k_inference.sh q0p1|q0p2|q0p3}"
case "$ARM" in q0p1|q0p2|q0p3) ;; *) echo "Invalid arm: $ARM" >&2; exit 2 ;; esac
test -n "${TMUX:-}" || { echo "Run inside an attached tmux session" >&2; exit 2; }
[[ "$(tmux display-message -p '#{session_attached}')" -gt 0 ]] || {
    echo "The tmux session must be attached at launch" >&2
    exit 2
}

JOB_ROOT="${REPO_ROOT}/outputs/ProMoE_TC_B_adepth/reeval_20261020"
[[ -f "${JOB_ROOT}/invalidation_manifest.json" && -f "${JOB_ROOT}/source.sha256" ]]
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
RUN_DIR="${REPO_ROOT}/outputs/ProMoE_TC_B_adepth/${STEM}"
CHECKPOINT="${RUN_DIR}/checkpoints/ckpt_step_300000.pth"
RUNTIME_CONFIG="${JOB_ROOT}/configs/${STEM}.yaml"
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

cleanup() {
    local rc=$?
    promoe_gpu_release
    if [[ "$rc" -ne 0 ]]; then
        record_status "failed_exit_${rc}" || true
        echo "[$(date -Is)] FAILED $ARM rc=$rc"
    fi
}
trap cleanup EXIT

sha256sum --check --quiet "${JOB_ROOT}/source.sha256"
[[ -s "$CHECKPOINT" && -d "$VAE_DIR" ]]
"$PYTHON" - "$JOB_ROOT" "$ARM" "$CHECKPOINT" <<'PY'
import json, pathlib, sys
root, arm, checkpoint = sys.argv[1:]
manifest = json.loads((pathlib.Path(root) / "invalidation_manifest.json").read_text())
assert manifest["cleanup_complete"], "Old evaluation cleanup has not completed"
expected = manifest["arms"][arm]["checkpoint"]
actual = pathlib.Path(checkpoint).stat()
assert actual.st_size == expected["size"] and actual.st_mtime_ns == expected["mtime_ns"], \
    "Checkpoint changed after validation"
PY

if [[ -s "${JOB_ROOT}/${ARM}.results.json" ]]; then
    for scale in 1.0 1.5; do
        promoe_eval_file_metrics_valid \
            "${RUN_DIR}/sample/step300000/img256_cfg${scale}_seed0_FID50K_bs128_ema/images_eval_openai.txt"
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
sha256sum --check --quiet "${JOB_ROOT}/source.sha256"

mkdir -p "${JOB_ROOT}/configs"
"$PYTHON" - "$CONFIG" "$RUNTIME_CONFIG" "$PROMOE_CLAIMED_GPUS" "$REPO_ROOT" <<'PY'
import pathlib, sys, yaml
source, target, gpus, repo = sys.argv[1:]
config = yaml.safe_load(pathlib.Path(source).read_text())
assert len(config["gpu_ids"]) == 4
assert config["global_seed"] == 0 and config["sample_batch_size"] == 128
assert config["sample_steps"] == 250 and config["DiT_B_config"]["qk_norm"] is False
config["output_dir"] = str(pathlib.Path(repo) / "outputs")
config["sample_gpu_ids"] = [int(x) for x in gpus.split(",")]
config["step_list_for_sample"] = [300000]
config["guide_scale_list"] = [1.0, 1.5]
config["num_fid_samples"] = config["save_img_num"] = 50000
config["save_inception_features"] = False
pathlib.Path(target).write_text(yaml.safe_dump(config, sort_keys=False))
PY

record_status sampling
CUDA_VISIBLE_DEVICES="$PROMOE_CLAIMED_GPUS" "$PYTHON" sample.py \
    --config "$RUNTIME_CONFIG" --step_list_for_sample 300000 \
    --guide_scale_list 1.0,1.5 --num_fid_samples 50000 --vae-path "$VAE_DIR"

# H800 historical results used this evaluator on CPU. Release GPUs for the next
# arm, and serialize CPU evaluations to avoid multiple Inception jobs competing.
promoe_gpu_release
PROMOE_CLAIMED_GPUS=""
record_status waiting_for_evaluator
exec 8>"${JOB_ROOT}/evaluator.lock"
flock 8
record_status evaluating
sha256sum --check --quiet "${JOB_ROOT}/source.sha256"
for scale in 1.0 1.5; do
    images="${RUN_DIR}/sample/step300000/img256_cfg${scale}_seed0_FID50K_bs128_ema/images"
    [[ -d "$images" && ! -L "$images" ]]
    CUDA_VISIBLE_DEVICES="" "$PYTHON_EVAL" evaluation/run_eval.py "$images" \
        --count 50000 --eval-device cpu
    promoe_eval_file_metrics_valid "${images}_eval_openai.txt"
done

"$PYTHON" - "$RUN_DIR" "$JOB_ROOT" "$ARM" <<'PY'
import datetime, hashlib, json, os, pathlib, re, sys
run, root, arm = map(str, sys.argv[1:])
results = {}
for scale in ("1.0", "1.5"):
    path = pathlib.Path(run) / "sample/step300000" / \
        f"img256_cfg{scale}_seed0_FID50K_bs128_ema/images_eval_openai.txt"
    text = path.read_text()
    values = {}
    for metric in ("FID", "Inception Score", "sFID", "Precision", "Recall"):
        match = re.search(r"^" + re.escape(metric) + r":\s*([0-9.eE+-]+)\s*$", text, re.M)
        if not match:
            raise ValueError(f"Missing metric {metric}: {path}")
        values[metric] = float(match.group(1))
    results[scale] = dict(metrics=values, evaluator_file=str(path),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
record = dict(arm=arm, checkpoint_step=300000, inference="checkpoint_annealed_skip_and_loop",
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
summary = dict(complete=len(combined) == 3, checkpoint_step=300000,
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
