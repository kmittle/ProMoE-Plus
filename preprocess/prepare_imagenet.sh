#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Idempotent ImageNet-1K preparation for ProMoE experiment scripts.
#
# Ensures the full-resolution ImageNet-1K train set + its VAE latents exist at
# datasets/imagenet/ (downloading from HuggingFace, falling back to ModelScope,
# then VAE-encoding). If everything is already present it returns immediately, so
# it is safe (and cheap) to call at the top of every run_*_train_sample_eval.sh.
#
# Usage:
#   bash preprocess/prepare_imagenet.sh --python <python> --gpus <csv-gpu-ids> [extra args...]
#
# It exports PROMOE_DATA_PATH=datasets/imagenet/train so train.py / sample.py /
# preprocess_vae.py all read the materialised dataset. Any extra args are passed
# straight through to prepare_imagenet.py (e.g. --source modelscope, --keep-parquet).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="python"
GPUS=""
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --gpus)   GPUS="$2";   shift 2 ;;
    *)        PASS+=("$1"); shift ;;
  esac
done

# Keep train.py's replace('train', ...) latent derivation safe: relative path,
# resolved from the repo root (where all run scripts cd).
export PROMOE_DATA_PATH="datasets/imagenet/train"

# Cross-process lock. The run-time slot system co-schedules two 4-GPU jobs on one
# physical server (X.1 -> GPU 0-3, X.2 -> GPU 4-7) sharing this repo/filesystem.
# Without a lock, both would race to download/materialise/encode the SAME files and
# corrupt the dataset. flock serializes prepare: the peer job blocks here, and once
# it acquires the lock prepare_imagenet.py sees the sentinel and returns immediately.
LOCK="${REPO_ROOT}/datasets/imagenet/.state/prepare.lock"
mkdir -p "$(dirname "${LOCK}")"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK}"
  echo "[prepare_imagenet.sh] acquiring prepare lock (${LOCK}) ..."
  flock 9
else
  echo "[prepare_imagenet.sh] WARNING: 'flock' not found; running without cross-process lock" >&2
fi

echo "[prepare_imagenet.sh] ensuring ImageNet-1K is ready (python=${PYTHON}, gpus=${GPUS:-all})"
"${PYTHON}" "${REPO_ROOT}/preprocess/prepare_imagenet.py" \
  --python "${PYTHON}" --gpus "${GPUS}" ${PASS[@]+"${PASS[@]}"}
echo "[prepare_imagenet.sh] ImageNet-1K ready."
