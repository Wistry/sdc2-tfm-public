#!/usr/bin/env bash
# Ejecuta SoFiA sobre la lista activa de configuraciones de esta fase.
# El script no crea catálogos por sí mismo: toma cada .par, sustituye los
# placeholders de entrada/salida y llama al binario `sofia`.
set -u

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$BASE_DIR/../.." && pwd)"
SDC2_DATA_ROOT="${SDC2_DATA_ROOT:-$REPO_ROOT/data}"
SOFIA_BIN="${SOFIA_BIN:-sofia}"
CONFIG_DIR="$BASE_DIR/configs"
OUTPUT_DIR="$BASE_DIR/outputs/catalogs"
INPUT_FITS="${SDC2_10GB_CUBE:-$SDC2_DATA_ROOT/sky_dev_v2.fits}"

CONFIGS=(
  "baseline_current.par"
  "sofia2_default_template.par"
  "sdc2_team_sofia_like.par"
  "hi_friends_dev12_like.par"
  "hi_friends_yaml_like.par"
  "loose_recall.par"
  "strict_reliability.par"
)

fail_fast_checks() {
  if ! command -v "$SOFIA_BIN" >/dev/null 2>&1; then
    echo "ERROR: no se encuentra '$SOFIA_BIN' en PATH."
    exit 1
  fi

  if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "ERROR: no existe CONFIG_DIR: $CONFIG_DIR"
    exit 1
  fi

  if [[ ! -f "$INPUT_FITS" ]]; then
    echo "ERROR: no existe INPUT_FITS: $INPUT_FITS"
    echo "Configura SDC2_10GB_CUBE o SDC2_DATA_ROOT antes de ejecutar."
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"
}

has_placeholders() {
  local config_path="$1"
  grep -q "\\[PENDIENTE" "$config_path"
}

run_one_config() {
  # Cada configuración se ejecuta en su propia carpeta para conservar
  # el .par usado, el log y cualquier catálogo generado.
  local config_file="$1"
  local config_name="${config_file%.par}"
  local config_path="$CONFIG_DIR/$config_file"
  local run_dir="$OUTPUT_DIR/$config_name"
  local timestamp
  local log_path
  local run_config

  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_path="$run_dir/run_${timestamp}.log"

  if [[ ! -f "$config_path" ]]; then
    mkdir -p "$run_dir"
    echo "ERROR: no existe config: $config_path" > "$log_path"
    echo "$config_name|ERROR|$log_path"
    return 0
  fi

  mkdir -p "$run_dir"

  if has_placeholders "$config_path"; then
    {
      echo "ERROR: config contiene placeholders sin resolver."
      echo "Config: $config_path"
      echo "Edita los placeholders [PENDIENTE...] antes de ejecutar SoFiA."
    } > "$log_path"
    echo "$config_name|ERROR|$log_path"
    return 0
  fi

  run_config="$run_dir/config_used.par"
  sed \
    -e "s|<FITS_PATH>|$INPUT_FITS|g" \
    -e "s|<OUTPUT_DIR>|$run_dir|g" \
    "$config_path" > "$run_config"

  if grep -q "<FITS_PATH>\\|<OUTPUT_DIR>" "$run_config"; then
    {
      echo "ERROR: no se pudieron sustituir todos los placeholders."
      echo "Config generada: $run_config"
    } > "$log_path"
    echo "$config_name|ERROR|$log_path"
    return 0
  fi

  echo "==> Ejecutando $config_name" >&2
  echo "    config: $run_config" >&2
  echo "    log:    $log_path" >&2

  "$SOFIA_BIN" "$run_config" > "$log_path" 2>&1
  local status=$?

  if [[ "$status" -eq 0 ]]; then
    echo "    OK $config_name" >&2
    echo "$config_name|OK|$log_path"
  else
    echo "    ERROR $config_name (codigo $status)" >&2
    echo "$config_name|ERROR|$log_path"
  fi

  return 0
}

main() {
  fail_fast_checks

  local summary=()
  local result

  for config in "${CONFIGS[@]}"; do
    result="$(run_one_config "$config")"
    summary+=("$result")
  done

  echo "config | status | log_path"
  echo "--- | --- | ---"
  for row in "${summary[@]}"; do
    IFS="|" read -r config_name status log_path <<< "$row"
    echo "$config_name | $status | $log_path"
  done
}

main "$@"
