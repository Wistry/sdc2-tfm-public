#!/usr/bin/env python3
"""Merge tiled SoFiA catalogues and keep only external, non-overlapping candidates."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from validation40gb_utils import CONFIGS, find_tile_catalog, inside_central_10gb_region, read_sofia_catalog


META = VALIDATION_ROOT / "outputs" / "tile_metadata" / "tiles_40gb_external.csv"
AUDIT = VALIDATION_ROOT / "outputs" / "tile_catalog_audit" / "tile_coordinate_audit.csv"
OUT_DIR = VALIDATION_ROOT / "outputs" / "merged_tile_catalogs"


def inside_keep(df: pd.DataFrame, tile: pd.Series) -> pd.Series:
    x = pd.to_numeric(df["x_global"], errors="coerce")
    y = pd.to_numeric(df["y_global"], errors="coerce")
    z = pd.to_numeric(df["z_global"], errors="coerce")
    return (
        (x >= tile["keep_x_min"])
        & (x <= tile["keep_x_max"])
        & (y >= tile["keep_y_min"])
        & (y <= tile["keep_y_max"])
        & (z >= tile["keep_z_min"])
        & (z <= tile["keep_z_max"])
    )


def coordinate_type(config_name: str, tile_name: str) -> str:
    if not AUDIT.exists():
        return "auto_global"
    audit = pd.read_csv(AUDIT)
    match = audit[(audit["config_name"] == config_name) & (audit["tile_name"] == tile_name)]
    if match.empty:
        return "auto_global"
    return str(match["coordinate_type"].iloc[0])


def add_global_columns(df: pd.DataFrame, tile: pd.Series, coord_type: str) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out["x"], errors="coerce")
    y = pd.to_numeric(out["y"], errors="coerce")
    z = pd.to_numeric(out["z"], errors="coerce")
    if coord_type == "local":
        out["x_global"] = x + float(tile["process_x_min"])
        out["y_global"] = y + float(tile["process_y_min"])
        out["z_global"] = z + float(tile["process_z_min"])
    else:
        out["x_global"] = x
        out["y_global"] = y
        out["z_global"] = z
    return out


def main() -> None:
    if not META.exists():
        raise FileNotFoundError(f"Missing tile metadata. Run 02_sofia_tile_generation/01_prepare_sofia_40gb_tile_runs.py first: {META}")
    tiles = pd.read_csv(META)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for _, meta in CONFIGS.items():
        config_name = meta["run_name"]
        raw_frames = []
        kept_frames = []
        for _, tile in tiles.iterrows():
            tile_name = tile["tile_name"]
            catalog_path = find_tile_catalog(config_name, tile_name)
            df = read_sofia_catalog(catalog_path)
            if df.empty:
                continue
            coord = coordinate_type(config_name, tile_name)
            df = add_global_columns(df, tile, coord)
            df["tile_name"] = tile_name
            df["config_name"] = config_name
            df["inside_keep_bbox"] = inside_keep(df, tile)
            df["inside_central_10gb_region"] = inside_central_10gb_region(df, "x_global", "y_global")
            raw_frames.append(df)
            kept_frames.append(df[df["inside_keep_bbox"]].copy())
        raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
        kept = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
        if not kept.empty:
            external = kept[~kept["inside_central_10gb_region"]].copy()
        else:
            external = kept
        out_path = OUT_DIR / f"{config_name}_external_merged.csv"
        external.to_csv(out_path, index=False)
        summary_rows.append(
            {
                "config_name": config_name,
                "n_tiles": len(tiles),
                "n_candidates_raw": len(raw),
                "n_candidates_after_keep_filter": len(kept),
                "n_candidates_external": len(external),
                "output_path": str(out_path),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "merged_tile_catalog_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
