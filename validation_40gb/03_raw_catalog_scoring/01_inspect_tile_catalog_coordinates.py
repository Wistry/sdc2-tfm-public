#!/usr/bin/env python3
"""Inspect whether tiled SoFiA catalogues use global or local coordinates."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from validation40gb_utils import CONFIGS, find_tile_catalog, read_sofia_catalog


META = VALIDATION_ROOT / "outputs" / "tile_metadata" / "tiles_40gb_external.csv"
OUT = VALIDATION_ROOT / "outputs" / "tile_catalog_audit" / "tile_coordinate_audit.csv"


def classify_coords(df: pd.DataFrame, tile: pd.Series) -> tuple[str, str]:
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    z = pd.to_numeric(df["z"], errors="coerce")
    px0, px1 = float(tile["process_x_min"]), float(tile["process_x_max"])
    py0, py1 = float(tile["process_y_min"]), float(tile["process_y_max"])
    pz0, pz1 = float(tile["process_z_min"]), float(tile["process_z_max"])
    width_x = px1 - px0 + 1
    width_y = py1 - py0 + 1
    width_z = pz1 - pz0 + 1
    global_like = x.between(px0 - 2, px1 + 2).mean() > 0.9 and y.between(py0 - 2, py1 + 2).mean() > 0.9
    local_like = x.between(-2, width_x + 2).mean() > 0.9 and y.between(-2, width_y + 2).mean() > 0.9 and not global_like
    z_global_like = z.between(pz0 - 2, pz1 + 2).mean() > 0.9
    z_local_like = z.between(-2, width_z + 2).mean() > 0.9 and not z_global_like
    if global_like:
        return "global", "x/y within process bbox"
    if local_like:
        return "local", f"apply offsets x+{int(px0)}, y+{int(py0)}, z+{int(pz0)}; z_local_like={z_local_like}"
    return "unknown", "ranges do not clearly match global or local tile coordinates"


def main() -> None:
    if not META.exists():
        raise FileNotFoundError(f"Missing tile metadata. Run 02_sofia_tile_generation/01_prepare_sofia_40gb_tile_runs.py first: {META}")
    tiles = pd.read_csv(META)
    rows = []
    for _, tile in tiles.iterrows():
        tile_name = tile["tile_name"]
        for _, meta in CONFIGS.items():
            config_run_name = meta["run_name"]
            try:
                catalog_path = find_tile_catalog(config_run_name, tile_name)
                df = read_sofia_catalog(catalog_path)
                if df.empty:
                    coord_type, note = "empty", "empty catalogue"
                else:
                    coord_type, note = classify_coords(df, tile)
                row = {
                    "config_name": config_run_name,
                    "tile_name": tile_name,
                    "catalog_found": True,
                    "catalog_path": str(catalog_path),
                    "n_rows": len(df),
                    "coordinate_type": coord_type,
                    "note": note,
                    "x_min": pd.to_numeric(df["x"], errors="coerce").min() if "x" in df.columns and not df.empty else None,
                    "x_max": pd.to_numeric(df["x"], errors="coerce").max() if "x" in df.columns and not df.empty else None,
                    "y_min": pd.to_numeric(df["y"], errors="coerce").min() if "y" in df.columns and not df.empty else None,
                    "y_max": pd.to_numeric(df["y"], errors="coerce").max() if "y" in df.columns and not df.empty else None,
                    "z_min": pd.to_numeric(df["z"], errors="coerce").min() if "z" in df.columns and not df.empty else None,
                    "z_max": pd.to_numeric(df["z"], errors="coerce").max() if "z" in df.columns and not df.empty else None,
                }
            except FileNotFoundError as exc:
                row = {
                    "config_name": config_run_name,
                    "tile_name": tile_name,
                    "catalog_found": False,
                    "catalog_path": "",
                    "n_rows": 0,
                    "coordinate_type": "missing",
                    "note": str(exc),
                    "x_min": None,
                    "x_max": None,
                    "y_min": None,
                    "y_max": None,
                    "z_min": None,
                    "z_max": None,
                }
            rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT, index=False)
    print(summary[["config_name", "tile_name", "catalog_found", "n_rows", "coordinate_type", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
