#!/bin/bash
#
# Train ProMoE-TC-REPA-Shared-B for 500K steps
# Config: configs/004_ProMoE_B_repa_shared.yaml
# GPUs:   4 x GPU (as specified in yaml gpu_ids: [0,1,2,3])
#

set -e

cd "$(dirname "$0")/../.."

python train_with_repa.py \
    --config configs/004_ProMoE_B_repa_shared.yaml
