#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Idempotent ImageNet-1K preparation for ProMoE experiment scripts.
#
# Ensures the ImageNet-1K VAE latents exist at
# /lustre01/yujie/dataset/imagenet/sd-vae-ft-mse_Latents_256img_npz/. If the raw HF
# parquet shards are already on the server (PROMOE_PARQUET_DIR, default
# /lustre01/qianyuan/data/ILSVRC/imagenet-1k/data), it encodes latents DIRECTLY from
# parquet (no re-download, no intermediate JPEG folder); otherwise it downloads
# (HuggingFace -> ModelScope) + materialises + encodes. If already present it returns
# immediately. Run it manually once (per shared dataset location), not from the
# experiment scripts, before launching training.
#
# Usage:
#   bash preprocess/prepare_imagenet.sh --python <python> --gpus <csv-gpu-ids> [extra args...]
#
# It exports PROMOE_DATA_PATH=/lustre01/yujie/dataset/imagenet/train so
# preprocess_vae.py reads/writes the right place; train.py / sample.py read the same
# path via config.py's default. Any extra args are passed straight through to
# prepare_imagenet.py (e.g. --source modelscope, --keep-parquet).
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

# Shared dataset location (absolute). 'train'-once-safe for train.py's
# replace('train', ...) latent derivation (no path component but train/ has 'train').
export PROMOE_DATA_PATH="/lustre01/yujie/dataset/imagenet/train"

# Cross-process lock. The run-time slot system co-schedules two 4-GPU jobs on one
# physical server (X.1 -> GPU 0-3, X.2 -> GPU 4-7) sharing this repo/filesystem.
# Without a lock, both would race to download/materialise/encode the SAME files and
# corrupt the dataset. flock serializes prepare: the peer job blocks here, and once
# it acquires the lock prepare_imagenet.py sees the sentinel and returns immediately.
LOCK="/lustre01/yujie/dataset/imagenet/.state/prepare.lock"
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
