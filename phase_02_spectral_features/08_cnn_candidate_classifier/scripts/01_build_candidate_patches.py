#!/usr/bin/env python3
"""Build small normalized FITS cube patches around SoFiA candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits


BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier")
PATCH_DIR = BASE / "outputs" / "patches"
PATCH_XY = 32
PATCH_Z = 21


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def first_data_hdu(hdul: fits.HDUList) -> np.ndarray:
    for hdu in hdul:
        if hdu.data is not None:
            if hdu.data.ndim != 3:
                raise ValueError(f"Expected 3D FITS cube, got {hdu.data.shape}")
            return hdu.data
    raise ValueError("No FITS image data found.")


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch = patch.astype("float32", copy=False)
    finite = np.isfinite(patch)
    if not finite.any():
        return np.zeros_like(patch, dtype="float32")
    med = np.nanmedian(patch[finite])
    mad = np.nanmedian(np.abs(patch[finite] - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(patch[finite])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    out = (patch - med) / scale
    out[~np.isfinite(out)] = 0.0
    return out.astype("float32", copy=False)


def extract_patch(cube: np.ndarray, x: float, y: float, z: float) -> tuple[np.ndarray, bool]:
    z_size, y_size, x_size = cube.shape
    half_xy = PATCH_XY // 2
    half_z = PATCH_Z // 2
    xi, yi, zi = int(round(x)), int(round(y)), int(round(z))

    x0, x1 = xi - half_xy, xi + half_xy
    y0, y1 = yi - half_xy, yi + half_xy
    z0, z1 = zi - half_z, zi + half_z + 1

    src_x0, src_x1 = max(0, x0), min(x_size, x1)
    src_y0, src_y1 = max(0, y0), min(y_size, y1)
    src_z0, src_z1 = max(0, z0), min(z_size, z1)
    edge = src_x0 != x0 or src_x1 != x1 or src_y0 != y0 or src_y1 != y1 or src_z0 != z0 or src_z1 != z1

    patch = np.zeros((PATCH_Z, PATCH_XY, PATCH_XY), dtype="float32")
    dst_x0, dst_y0, dst_z0 = src_x0 - x0, src_y0 - y0, src_z0 - z0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    if src_x0 < src_x1 and src_y0 < src_y1 and src_z0 < src_z1:
        patch[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = np.asarray(
            cube[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1],
            dtype="float32",
        )
    return normalize_patch(patch), edge


def build_for_catalog(
    cube: np.ndarray,
    catalog_path: Path,
    prefix: str,
    training: bool,
) -> dict[str, Any]:
    df = pd.read_csv(catalog_path)
    source = df[df["clean_label"].isin([0, 1])].copy() if training else df.copy()
    patches: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for idx, row in source.iterrows():
        try:
            patch, edge = extract_patch(cube, float(row["x"]), float(row["y"]), float(row["z"]))
        except Exception as exc:  # noqa: BLE001 - keep building remaining candidates.
            dropped.append({"row_index": int(idx), "reason": str(exc)})
            continue
        patches.append(patch)
        meta = row.to_dict()
        meta["source_row_index"] = int(idx)
        meta["cnn_patch_edge_clipped"] = bool(edge)
        metadata_rows.append(meta)

    patch_array = np.stack(patches).astype("float32") if patches else np.empty((0, PATCH_Z, PATCH_XY, PATCH_XY), dtype="float32")
    meta_df = pd.DataFrame(metadata_rows)
    np.save(PATCH_DIR / f"{prefix}_patches.npy", patch_array)
    meta_df.to_csv(PATCH_DIR / f"{prefix}_metadata.csv", index=False)
    if training:
        labels = meta_df["clean_label"].astype("int64").to_numpy() if not meta_df.empty else np.empty((0,), dtype="int64")
        np.save(PATCH_DIR / f"{prefix}_labels.npy", labels)

    return {
        "prefix": prefix,
        "input_rows": len(df),
        "used_rows": len(source),
        "patches": len(patch_array),
        "shape": "x".join(map(str, patch_array.shape)),
        "class_counts": meta_df["clean_label"].value_counts(dropna=False).to_dict() if "clean_label" in meta_df else {},
        "edge_clipped": int(meta_df["cnn_patch_edge_clipped"].sum()) if "cnn_patch_edge_clipped" in meta_df else 0,
        "dropped": dropped[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    cube_path = Path(cfg["cube_path"])
    baseline = Path(cfg["catalogs"]["baseline_current_full"]["output_clean"])
    conservative = Path(cfg["catalogs"]["sdc2_team_sofia_like_full"]["output_clean"])

    with fits.open(cube_path, memmap=True) as hdul:
        cube = first_data_hdu(hdul)
        summaries = [
            build_for_catalog(cube, baseline, "baseline", training=True),
            build_for_catalog(cube, conservative, "sdc2_conservative", training=False),
        ]

    print(f"Wrote patches to {PATCH_DIR}")
    for item in summaries:
        print(item["prefix"], item["shape"], item["class_counts"])


if __name__ == "__main__":
    main()
