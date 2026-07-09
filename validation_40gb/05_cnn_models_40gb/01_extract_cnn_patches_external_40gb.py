#!/usr/bin/env python3
"""Extract CNN patches for 40 GB extended-validation tiled catalogues."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from validation40gb_utils import LARGE_CUBE, ensure_dir


BASE = VALIDATION_ROOT
MERGED_DIR = BASE / "outputs" / "merged_tile_catalogs"
OUT_DIR = BASE / "outputs" / "external_cnn_patches"
PATCH_Z = 21
PATCH_XY = 32

CATALOGS = {
    "baseline_current_40gb": MERGED_DIR / "baseline_current_40gb_external_merged.csv",
    "sdc2_team_sofia_like_40gb": MERGED_DIR / "sdc2_team_sofia_like_40gb_external_merged.csv",
}


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
    median = np.nanmedian(patch[finite])
    mad = np.nanmedian(np.abs(patch[finite] - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(patch[finite])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    out = (patch - median) / scale
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
    edge_clipped = (
        src_x0 != x0
        or src_x1 != x1
        or src_y0 != y0
        or src_y1 != y1
        or src_z0 != z0
        or src_z1 != z1
    )

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
    return normalize_patch(patch), edge_clipped


def candidate_position(row: pd.Series) -> tuple[float, float, float]:
    x_col = "x_global" if "x_global" in row.index else "x"
    y_col = "y_global" if "y_global" in row.index else "y"
    z_col = "z_global" if "z_global" in row.index else "z"
    return float(row[x_col]), float(row[y_col]), float(row[z_col])


def build_for_catalog(cube: np.ndarray, catalog_key: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing external merged catalogue: {path}")
    df = pd.read_csv(path)
    required = {"x", "y", "z"}
    global_required = {"x_global", "y_global", "z_global"}
    if not required.issubset(df.columns) and not global_required.issubset(df.columns):
        raise ValueError(f"{path} must contain x/y/z or x_global/y_global/z_global.")

    patches: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        try:
            x, y, z = candidate_position(row)
            patch, edge_clipped = extract_patch(cube, x, y, z)
        except Exception as exc:  # noqa: BLE001 - keep remaining candidates.
            dropped.append({"candidate_index": int(idx), "reason": str(exc)})
            continue

        patches.append(patch)
        meta = row.to_dict()
        meta["candidate_index"] = int(idx)
        meta["candidate_id"] = meta.get("id", idx)
        meta["base_catalog"] = catalog_key
        meta["cnn_x"] = x
        meta["cnn_y"] = y
        meta["cnn_z"] = z
        meta["cnn_patch_edge_clipped"] = bool(edge_clipped)
        metadata_rows.append(meta)

    patch_array = (
        np.stack(patches).astype("float32")
        if patches
        else np.empty((0, PATCH_Z, PATCH_XY, PATCH_XY), dtype="float32")
    )
    metadata = pd.DataFrame(metadata_rows)

    patch_path = OUT_DIR / f"{catalog_key}_external_patches.npz"
    metadata_path = OUT_DIR / f"{catalog_key}_external_patches_metadata.csv"
    np.savez_compressed(
        patch_path,
        patches=patch_array,
        patch_z=np.asarray([PATCH_Z], dtype="int16"),
        patch_xy=np.asarray([PATCH_XY], dtype="int16"),
    )
    metadata.to_csv(metadata_path, index=False)

    return {
        "base_catalog": catalog_key,
        "input_rows": len(df),
        "patches": len(patch_array),
        "patch_shape": "x".join(map(str, patch_array.shape)),
        "edge_clipped": int(metadata["cnn_patch_edge_clipped"].sum()) if "cnn_patch_edge_clipped" in metadata else 0,
        "dropped": len(dropped),
        "patch_path": str(patch_path),
        "metadata_path": str(metadata_path),
    }


def main() -> None:
    if not LARGE_CUBE.exists():
        raise FileNotFoundError(f"Missing 40GB FITS cube: {LARGE_CUBE}")
    ensure_dir(OUT_DIR)

    with fits.open(LARGE_CUBE, memmap=True) as hdul:
        cube = first_data_hdu(hdul)
        summaries = [build_for_catalog(cube, key, path) for key, path in CATALOGS.items()]

    summary = pd.DataFrame(summaries)
    summary_path = OUT_DIR / "external_cnn_patch_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
