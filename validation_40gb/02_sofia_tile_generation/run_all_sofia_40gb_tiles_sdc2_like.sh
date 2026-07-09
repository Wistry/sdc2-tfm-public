#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="${VALIDATION_ROOT}/sofia_tile_runs/sdc2_team_sofia_like_40gb"

for tile in left_bottom left_center left_top center_bottom center_top right_bottom right_center right_top; do
  bash "${RUN_ROOT}/${tile}/run_sofia.sh"
done
