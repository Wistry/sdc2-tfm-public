#!/usr/bin/env bash
set -euo pipefail

BLOCK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE3_ROOT="$(cd "${BLOCK_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PHASE3_ROOT}/.." && pwd)"
LOG_DIR="${PHASE3_ROOT}/outputs/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

run_step() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${STAMP}_${name}.log"
  echo "[INFO] Running ${name}"
  "$@" 2>&1 | tee "${log_path}"
  echo "[OK] ${name} log: ${log_path}"
}

run_step "01_extract_cnn_patches_external_40gb" \
  python phase_03_external_validation_40gb/05_cnn_models_40gb/01_extract_cnn_patches_external_40gb.py

run_step "02_apply_frozen_cnn_external_40gb" \
  python phase_03_external_validation_40gb/05_cnn_models_40gb/02_apply_frozen_cnn_external_40gb.py

run_step "03_score_cnn_external_40gb" \
  python phase_03_external_validation_40gb/05_cnn_models_40gb/03_score_cnn_external_40gb.py

run_step "04_apply_valid_frozen_cnns_external_40gb" \
  python phase_03_external_validation_40gb/05_cnn_models_40gb/04_apply_valid_frozen_cnns_external_40gb.py

run_step "05_score_valid_frozen_cnns_external_40gb" \
  python phase_03_external_validation_40gb/05_cnn_models_40gb/05_score_valid_frozen_cnns_external_40gb.py

run_step "06_make_valid_cnn_external_report" \
  python phase_03_external_validation_40gb/05_cnn_models_40gb/06_make_valid_cnn_external_report.py

echo "[OK] Phase 3 external CNN exploratory pipeline completed."
