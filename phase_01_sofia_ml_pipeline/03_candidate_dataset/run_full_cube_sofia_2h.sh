#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SOFIA_BIN="sofia"
BASE_DIR="phase_01_sofia_ml_pipeline/03_candidate_dataset"
CONFIG_DIR="$BASE_DIR/configs"
OUTPUT_DIR="$BASE_DIR/outputs"
MASTER_LOG="$OUTPUT_DIR/run_full_cube_sofia_2h.log"
SUMMARY="$OUTPUT_DIR/full_cube_run_summary.md"
TIMEOUT_SECONDS=7200

BASELINE_CONFIG="$CONFIG_DIR/baseline_current_full.par"
SDC2_CONFIG="$CONFIG_DIR/sdc2_team_sofia_like_full.par"
BASELINE_OUTPUT="$OUTPUT_DIR/baseline_current_full"
SDC2_OUTPUT="$OUTPUT_DIR/sdc2_team_sofia_like_full"

log() {
  echo "$*" | tee -a "$MASTER_LOG"
}

read_par_value() {
  local key="$1"
  local config="$2"
  awk -F '=' -v k="$key" '$1 ~ k {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$config"
}

validate_config() {
  local config="$1"
  local expected_output="$2"
  local input_data
  local input_region
  input_data="$(read_par_value "input.data" "$config")"
  input_region="$(read_par_value "input.region" "$config")"

  if [[ -z "$input_data" || "$input_data" == *"<"* || "$input_data" == *">"* ]]; then
    log "ERROR: input.data no es una ruta real en $config: '$input_data'"
    return 1
  fi
  if [[ ! -f "$input_data" ]]; then
    log "ERROR: input.data no existe en $config: $input_data"
    return 1
  fi
  if [[ -n "$input_region" ]]; then
    log "ERROR: input.region no esta desactivado en $config: '$input_region'"
    return 1
  fi
  if [[ "$(read_par_value "output.directory" "$config")" != "$expected_output" ]]; then
    log "ERROR: output.directory inesperado en $config"
    return 1
  fi
}

catalog_line_count() {
  local output_path="$1"
  local filename="$2"
  local catalog="$output_path/${filename}_cat.txt"
  if [[ -f "$catalog" ]]; then
    wc -l "$catalog" | awk '{print $1}'
  else
    echo "0"
  fi
}

run_config() {
  local label="$1"
  local config="$2"
  local output_path="$3"
  local filename="$4"
  local log_path="$output_path/run.log"

  mkdir -p "$output_path"
  log "==> Ejecutando $label"
  log "    config: $config"
  log "    output: $output_path"
  log "    log:    $log_path"
  echo "Inicio $label: $(date '+%Y-%m-%d %H:%M:%S')" > "$log_path"

  set +e
  timeout "$TIMEOUT_SECONDS" "$SOFIA_BIN" "$config" >> "$log_path" 2>&1
  local status=$?
  set -e

  echo "Fin $label: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_path"
  if [[ "$status" -eq 0 ]]; then
    local lines
    lines="$(catalog_line_count "$output_path" "$filename")"
    log "    OK $label | catalog_lines=$lines"
  elif [[ "$status" -eq 124 ]]; then
    log "    TIMEOUT $label tras ${TIMEOUT_SECONDS}s"
  else
    log "    ERROR $label codigo=$status"
  fi
  return "$status"
}

main() {
  mkdir -p "$OUTPUT_DIR" "$BASELINE_OUTPUT" "$SDC2_OUTPUT"
  : > "$MASTER_LOG"

  log "# Full cube SoFiA run"
  log "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"

  if ! command -v "$SOFIA_BIN" >/dev/null 2>&1; then
    log "ERROR: no se encuentra '$SOFIA_BIN' en PATH"
    exit 1
  fi
  [[ -f "$BASELINE_CONFIG" ]] || { log "ERROR: falta $BASELINE_CONFIG"; exit 1; }
  [[ -f "$SDC2_CONFIG" ]] || { log "ERROR: falta $SDC2_CONFIG"; exit 1; }

  validate_config "$BASELINE_CONFIG" "$BASELINE_OUTPUT"
  validate_config "$SDC2_CONFIG" "$SDC2_OUTPUT"

  local baseline_status
  if run_config "baseline_current_full" "$BASELINE_CONFIG" "$BASELINE_OUTPUT" "baseline_current_full"; then
    baseline_status=0
  else
    baseline_status=$?
  fi

  local sdc2_status="SKIPPED"
  if [[ "$baseline_status" -eq 0 ]]; then
    if run_config "sdc2_team_sofia_like_full" "$SDC2_CONFIG" "$SDC2_OUTPUT" "sdc2_team_sofia_like_full"; then
      sdc2_status=0
    else
      sdc2_status=$?
    fi
  else
    log "No se ejecuta sdc2_team_sofia_like_full porque baseline no termino OK."
  fi

  local baseline_lines
  local sdc2_lines
  baseline_lines="$(catalog_line_count "$BASELINE_OUTPUT" "baseline_current_full")"
  sdc2_lines="$(catalog_line_count "$SDC2_OUTPUT" "sdc2_team_sofia_like_full")"

  cat > "$SUMMARY" <<EOF
# Full cube SoFiA summary

- Fecha: $(date '+%Y-%m-%d %H:%M:%S')
- FITS: $(read_par_value "input.data" "$BASELINE_CONFIG")
- input.region: desactivado, cubo completo
- Timeout por configuracion: ${TIMEOUT_SECONDS}s

| config | status | catalog_lines | output |
| --- | --- | ---: | --- |
| baseline_current_full | ${baseline_status} | ${baseline_lines} | ${BASELINE_OUTPUT} |
| sdc2_team_sofia_like_full | ${sdc2_status} | ${sdc2_lines} | ${SDC2_OUTPUT} |

Logs:

- ${MASTER_LOG}
- ${BASELINE_OUTPUT}/run.log
- ${SDC2_OUTPUT}/run.log
EOF

  log "Resumen guardado en: $SUMMARY"
  log "Fin: $(date '+%Y-%m-%d %H:%M:%S')"
}

main "$@"
