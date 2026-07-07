#!/usr/bin/env python3
"""Prepare tiled SoFiA runs for external 40GB validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PHASE3_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

from phase03_utils import (
    CONFIGS,
    CUBE_X_MAX,
    CUBE_X_MIN,
    CUBE_Y_MAX,
    CUBE_Y_MIN,
    CUBE_Z_MAX,
    CUBE_Z_MIN,
    EXTERNAL_TILE_NAMES,
    LARGE_CUBE,
    TILE_MARGIN,
    ensure_dir,
    tile_run_dir,
)


X_BANDS = {
    "left": (0, 300),
    "center": (301, 983),
    "right": (984, 1285),
}
Y_BANDS = {
    "bottom": (0, 300),
    "center": (301, 983),
    "top": (984, 1285),
}


def build_tiles() -> list[dict[str, int | str | bool]]:
    rows: list[dict[str, int | str | bool]] = []
    for x_name, (kx0, kx1) in X_BANDS.items():
        for y_name, (ky0, ky1) in Y_BANDS.items():
            tile_name = f"{x_name}_{y_name}"
            is_central = tile_name == "center_center"
            row = {
                "tile_name": tile_name,
                "keep_x_min": kx0,
                "keep_x_max": kx1,
                "keep_y_min": ky0,
                "keep_y_max": ky1,
                "keep_z_min": CUBE_Z_MIN,
                "keep_z_max": CUBE_Z_MAX,
                "process_x_min": max(CUBE_X_MIN, kx0 - TILE_MARGIN),
                "process_x_max": min(CUBE_X_MAX, kx1 + TILE_MARGIN),
                "process_y_min": max(CUBE_Y_MIN, ky0 - TILE_MARGIN),
                "process_y_max": min(CUBE_Y_MAX, ky1 + TILE_MARGIN),
                "process_z_min": CUBE_Z_MIN,
                "process_z_max": CUBE_Z_MAX,
                "is_central_10gb_region": is_central,
            }
            if tile_name in EXTERNAL_TILE_NAMES:
                rows.append(row)
    return rows


def rewrite_par(text: str, config_run_name: str, tile: dict[str, int | str | bool], output_dir: Path) -> str:
    tile_name = str(tile["tile_name"])
    region = (
        f"{tile['process_x_min']}, {tile['process_x_max']}, "
        f"{tile['process_y_min']}, {tile['process_y_max']}, "
        f"{tile['process_z_min']}, {tile['process_z_max']}"
    )
    replacements = {
        "input.data": str(LARGE_CUBE),
        "input.region": region,
        "output.directory": str(output_dir.resolve()),
        "output.filename": f"{config_run_name}_{tile_name}",
        "output.overwrite": "true",
    }
    out_lines = []
    seen = set()
    for line in text.splitlines():
        stripped = line.strip()
        if "input.region intentionally disabled" in stripped:
            out_lines.append("# Tiled run: input.region is set to the process_bbox for this tile.")
            continue
        key = stripped.split("=", 1)[0].strip() if "=" in stripped and not stripped.startswith("#") else None
        if key in replacements:
            out_lines.append(f"{key:<27}= {replacements[key]}")
            seen.add(key)
        else:
            out_lines.append(line)
    for key, value in replacements.items():
        if key not in seen:
            out_lines.append(f"{key:<27}= {value}")
    return "\n".join(out_lines) + "\n"


def run_script_text(config_run_name: str, tile_name: str) -> str:
    par_name = f"{config_run_name}_{tile_name}.par"
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
SOFIA_BIN="${{SOFIA_BIN:-sofia}}"
LOG="run_sofia.log"

echo "[START] $(date -Is) {config_run_name}/{tile_name}" | tee "$LOG"
echo "Using SoFiA binary: $SOFIA_BIN" | tee -a "$LOG"
"$SOFIA_BIN" "{par_name}" 2>&1 | tee -a "$LOG"
echo "[END] $(date -Is) {config_run_name}/{tile_name}" | tee -a "$LOG"
"""


def write_runner(path: Path, config_run_name: str, tiles: list[dict[str, int | str | bool]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for tile in tiles:
        tile_name = str(tile["tile_name"])
        lines.append(f'bash "{PHASE3_ROOT / "sofia_tile_runs" / config_run_name / tile_name / "run_sofia.sh"}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    tiles = build_tiles()
    meta_dir = ensure_dir(PHASE3_ROOT / "outputs" / "tile_metadata")
    json_path = meta_dir / "tiles_40gb_external.json"
    csv_path = meta_dir / "tiles_40gb_external.csv"
    json_path.write_text(json.dumps(tiles, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tiles[0]))
        writer.writeheader()
        writer.writerows(tiles)

    for config_key, meta in CONFIGS.items():
        config_run_name = meta["run_name"]
        base_config = Path(meta["source_config"])
        if not base_config.exists():
            raise FileNotFoundError(f"Missing source SoFiA config: {base_config}")
        text = base_config.read_text(encoding="utf-8")
        for tile in tiles:
            tile_name = str(tile["tile_name"])
            rd = ensure_dir(tile_run_dir(config_run_name, tile_name))
            out_dir = ensure_dir(rd / "outputs")
            par_path = rd / f"{config_run_name}_{tile_name}.par"
            par_path.write_text(rewrite_par(text, config_run_name, tile, out_dir), encoding="utf-8")
            run_path = rd / "run_sofia.sh"
            run_path.write_text(run_script_text(config_run_name, tile_name), encoding="utf-8")
            run_path.chmod(0o755)
            print(f"Prepared {config_run_name}/{tile_name}: {par_path}")

    runner_dir = PHASE3_ROOT / "02_sofia_tile_generation"
    write_runner(runner_dir / "run_all_sofia_40gb_tiles_baseline.sh", "baseline_current_40gb", tiles)
    write_runner(runner_dir / "run_all_sofia_40gb_tiles_sdc2_like.sh", "sdc2_team_sofia_like_40gb", tiles)
    all_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'bash "{runner_dir / "run_all_sofia_40gb_tiles_baseline.sh"}"',
        f'bash "{runner_dir / "run_all_sofia_40gb_tiles_sdc2_like.sh"}"',
        "",
    ]
    all_runner = runner_dir / "run_all_sofia_40gb_tiles.sh"
    all_runner.write_text("\n".join(all_lines), encoding="utf-8")
    all_runner.chmod(0o755)
    print(f"Wrote tile metadata: {json_path}")
    print("SoFiA was not executed.")


if __name__ == "__main__":
    main()
