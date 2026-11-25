#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="configs/004_ProMoE_B_clean_route_student_ratio0_corr_router_kl_detach_base_no_ref_drop.yaml"
LOG="${REPO_ROOT}/logs/log_B_clean_route_student_ratio0_corr_router_kl_detach_base_no_ref_drop_train_sample_eval.log"
source "${SCRIPT_DIR}/_reference_dropout_train_sample_eval.sh"
