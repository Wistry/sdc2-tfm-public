#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_all_sofia_40gb_tiles_baseline.sh"
bash "${SCRIPT_DIR}/run_all_sofia_40gb_tiles_sdc2_like.sh"
