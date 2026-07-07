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

run_step "01_extract_phase2_features_external_tiles_40gb" \
  python phase_03_external_validation_40gb/04_tabular_models_40gb/01_extract_phase2_features_external_tiles_40gb.py

run_step "02_apply_frozen_phase2_models_external_tiles_40gb" \
  python phase_03_external_validation_40gb/04_tabular_models_40gb/02_apply_frozen_phase2_models_external_tiles_40gb.py

run_step "03_score_filtered_phase2_external_tiles_40gb" \
  python phase_03_external_validation_40gb/04_tabular_models_40gb/03_score_filtered_phase2_external_tiles_40gb.py

echo "[OK] Phase 3 external frozen-model pipeline completed."
