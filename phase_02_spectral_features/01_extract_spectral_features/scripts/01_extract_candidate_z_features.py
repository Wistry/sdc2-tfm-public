#!/usr/bin/env python3
"""
Extract spectral/local cube features around SoFiA candidates.

Idea general
============
Este script parte de un catálogo de candidatos
previamente generado por SoFiA y, para cada candidato, extrae un pequeño
subcubo local del FITS alrededor de su posición (x, y, z).

Después convierte ese subcubo 3D en una fila de features tabulares:

    candidato SoFiA
        ↓
    ventana local 3D del cubo FITS
        ↓
    región fuente + región fondo
        ↓
    medidas canal a canal en z
        ↓
    agregación estadística
        ↓
    features espectro-espaciales para ML

Convención de ejes
==================
El cubo FITS se usa como un array NumPy con forma:

    cube[z, y, x]

- z: eje espectral/frecuencial.
- y: eje espacial vertical.
- x: eje espacial horizontal.

El catálogo de SoFiA proporciona, como mínimo, las columnas x, y, z.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits


# -----------------------------------------------------------------------------
# Columnas originales que se conservan en la salida
# -----------------------------------------------------------------------------
# Estas columnas NO son features nuevas. Solo sirven para mantener trazabilidad:
# qué candidato era, dónde estaba y qué etiqueta tenía.
IDENTIFIER_COLUMNS = [
    "name",
    "id",
    "x",
    "y",
    "z",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "clean_label",
    "label",
    "is_ambiguous",
    "matched_truth_id",
    "matching_mode",
]


# -----------------------------------------------------------------------------
# Utilidades de configuración y carga de datos
# -----------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_catalog_config(cfg: dict[str, Any], catalog_key: str | None) -> dict[str, Any]:
    """
    Resolve the configuration for one concrete catalogue.

    The YAML can define several catalogues under `catalogs`. For example:

        catalogs:
          baseline_current_full:
            candidate_catalog: ...
            output_features: ...
          sdc2_team_sofia_like_full:
            candidate_catalog: ...
            output_features: ...

    If `--catalog-key` is not provided, the script defaults to
    `baseline_current_full` when a `catalogs` section exists.

    Important:
    - One execution processes one catalogue.
    - To process several catalogues, run the script several times with
      different `--catalog-key` values.
    """
    resolved = dict(cfg)
    catalogs = cfg.get("catalogs") or {}

    if catalog_key is None:
        if catalogs:
            catalog_key = "baseline_current_full"
        else:
            resolved["catalog_key"] = "default"
            return resolved

    if catalog_key not in catalogs:
        available = ", ".join(sorted(catalogs)) or "<none>"
        raise KeyError(f"Unknown catalog_key '{catalog_key}'. Available: {available}")

    resolved.update(catalogs[catalog_key])
    resolved["catalog_key"] = catalog_key
    return resolved


def first_data_hdu(hdul: fits.HDUList) -> np.ndarray:
    """
    Return the first 3D data array found in the FITS file.

    The expected shape is:

        cube[z, y, x]
    """
    for hdu in hdul:
        if hdu.data is not None:
            data = hdu.data
            if data.ndim != 3:
                raise ValueError(f"Expected a 3D FITS cube, got shape {data.shape}")
            return data
    raise ValueError("No image data found in FITS file.")


# -----------------------------------------------------------------------------
# Estadísticos seguros con NaN
# -----------------------------------------------------------------------------
# En datos reales pueden aparecer NaN o valores inválidos. Estas funciones
# ignoran valores no finitos para evitar que una feature falle por un solo NaN.
def finite_stats(values: np.ndarray) -> tuple[float, float]:
    """Return mean and std over finite values only."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(values)), float(np.nanstd(values))


def safe_nanmax(values: np.ndarray) -> float:
    """Return max over finite values, or NaN if there are no finite values."""
    values = values[np.isfinite(values)]
    return float(np.nanmax(values)) if values.size else np.nan


def safe_nanmean(values: np.ndarray) -> float:
    """Return mean over finite values, or NaN if there are no finite values."""
    values = values[np.isfinite(values)]
    return float(np.nanmean(values)) if values.size else np.nan


def safe_nanstd(values: np.ndarray) -> float:
    """Return std over finite values, or NaN if there are no finite values."""
    values = values[np.isfinite(values)]
    return float(np.nanstd(values)) if values.size else np.nan


# -----------------------------------------------------------------------------
# 1. Ventana local alrededor del candidato
# -----------------------------------------------------------------------------
def get_candidate_position(row: pd.Series) -> tuple[float, float, float, int, int, int]:
    """
    Read candidate coordinates from one catalogue row.

    SoFiA coordinates can be decimal. For indexing the NumPy cube we use rounded
    integer coordinates, but we keep the original decimal coordinates when
    computing distances and centroid offsets.
    """
    x = float(row["x"])
    y = float(row["y"])
    z = float(row["z"])
    xi = int(round(x))
    yi = int(round(y))
    zi = int(round(z))
    return x, y, z, xi, yi, zi


def compute_window_bounds(
    *,
    xi: int,
    yi: int,
    zi: int,
    cube_shape: tuple[int, int, int],
    window_xy: int,
    window_z: int,
) -> tuple[int, int, int, int, int, int, bool]:
    """
    Compute the crop limits of the local subcube.

    Ideal window size:

        z: 2 * window_z  + 1 channels
        y: 2 * window_xy + 1 pixels
        x: 2 * window_xy + 1 pixels

    If the candidate is close to a cube border, the window is clipped to stay
    inside the array. `edge_clipped=True` records that this happened.
    """
    z_size, y_size, x_size = cube_shape

    x0 = max(0, xi - window_xy)
    x1 = min(x_size, xi + window_xy + 1)
    y0 = max(0, yi - window_xy)
    y1 = min(y_size, yi + window_xy + 1)
    z0 = max(0, zi - window_z)
    z1 = min(z_size, zi + window_z + 1)

    edge_clipped = (
        x0 != xi - window_xy
        or x1 != xi + window_xy + 1
        or y0 != yi - window_xy
        or y1 != yi + window_xy + 1
        or z0 != zi - window_z
        or z1 != zi + window_z + 1
    )

    return x0, x1, y0, y1, z0, z1, bool(edge_clipped)


def extract_local_subcube(
    cube: np.ndarray,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    z0: int,
    z1: int,
) -> np.ndarray | None:
    """
    Extract local cube region as float64.

    Returns None if the requested window is empty.
    """
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return None

    subcube = np.asarray(cube[z0:z1, y0:y1, x0:x1], dtype=np.float64)
    if subcube.size == 0:
        return None

    return subcube


# -----------------------------------------------------------------------------
# 2. Máscaras espaciales: fuente y fondo
# -----------------------------------------------------------------------------
def build_spatial_masks(
    *,
    x: float,
    y: float,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    src_radius: float,
    bg_inner: float,
    bg_outer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2D source/background masks for the local window.

    These masks are spatial only: shape (y, x).
    They are reused for every z-channel of the subcube.

    source_mask:
        circular region centered on the candidate.

    background_mask:
        annulus around the source, used to estimate local background/noise.
    """
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    source_mask = dist <= src_radius
    background_mask = (dist >= bg_inner) & (dist <= bg_outer)

    return source_mask, background_mask, xx, yy


# -----------------------------------------------------------------------------
# 3. Cálculo por canal z
# -----------------------------------------------------------------------------
def estimate_local_threshold(plane: np.ndarray, bg_vals: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate local background and threshold for one channel.

    Main rule:

        threshold = background_mean + 2 * background_std

    Fallbacks are used when the background mask has invalid values.
    """
    bg_mean, bg_std = finite_stats(bg_vals)

    # Fallback for degenerate background standard deviation.
    if not np.isfinite(bg_std) or bg_std <= 0:
        finite_plane = plane[np.isfinite(plane)]
        bg_std = float(np.nanstd(finite_plane)) if finite_plane.size else np.nan

    # Fallback for invalid background mean.
    if not np.isfinite(bg_mean):
        finite_plane = plane[np.isfinite(plane)]
        bg_mean = float(np.nanmedian(finite_plane)) if finite_plane.size else np.nan

    threshold = bg_mean + 2.0 * bg_std if np.isfinite(bg_mean) and np.isfinite(bg_std) else np.nan

    # Last fallback: high percentile of the whole local plane.
    if not np.isfinite(threshold):
        finite_plane = plane[np.isfinite(plane)]
        threshold = float(np.nanpercentile(finite_plane, 90)) if finite_plane.size else np.nan

    return bg_mean, bg_std, threshold


def compute_active_mask(plane: np.ndarray, source_mask: np.ndarray, threshold: float) -> np.ndarray:
    """
    Pixels considered active in one z-channel.

    A pixel is active if:
    - it belongs to the source region;
    - it is finite;
    - it is above the local threshold.
    """
    return source_mask & np.isfinite(plane) & (plane > threshold)


def compute_flux_features_for_channel(src_vals: np.ndarray) -> tuple[float, float, float, float]:
    """
    Per-channel intensity summaries inside the source region.

    Returns:
    - spec_sum: total accumulated flux in the source mask.
    - spec_peak: maximum point intensity in the source mask.
    - source_mean: mean intensity in the source mask.
    - source_max: same as spec_peak, kept separately for local source features.
    """
    spec_sum = float(np.nansum(src_vals))
    spec_peak = safe_nanmax(src_vals)
    source_mean, _ = finite_stats(src_vals)
    source_max = spec_peak
    return spec_sum, spec_peak, source_mean, source_max


def compute_contrast_for_channel(source_mean: float, bg_mean: float, bg_std: float) -> float:
    """
    Local contrast between source and background.

    Formula:

        contrast = (source_mean - background_mean) / background_std

    This is not an official SNR. It is a simple local contrast descriptor.
    """
    if np.isfinite(bg_std) and bg_std > 0:
        return float((source_mean - bg_mean) / bg_std)
    return np.nan


def compute_centroid_for_channel(
    *,
    plane: np.ndarray,
    active: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    x: float,
    y: float,
    threshold: float,
) -> tuple[float, float, bool]:
    """
    Compute intensity-weighted centroid offsets for one channel.

    The centroid is computed only from active pixels. Each active pixel is
    weighted by how much it exceeds the local threshold:

        weight = plane_value - threshold

    Returned values are offsets relative to the original SoFiA candidate:

        dx = centroid_x - x
        dy = centroid_y - y
    """
    weights = np.where(active, plane - threshold, 0.0)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return np.nan, np.nan, False

    cx = float(np.sum(weights * xx) / total_weight)
    cy = float(np.sum(weights * yy) / total_weight)

    return cx - x, cy - y, True


def compute_channel_measurements(
    *,
    plane: np.ndarray,
    source_mask: np.ndarray,
    background_mask: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    x: float,
    y: float,
) -> dict[str, Any]:
    """
    Compute all intermediate measurements for one z-channel.

    This function does NOT return the final features. It returns per-channel
    values that will later be aggregated across z.
    """
    src_vals = plane[source_mask]

    # If the annulus is empty, use every non-source pixel as background fallback.
    bg_vals = plane[background_mask] if np.any(background_mask) else plane[~source_mask]

    bg_mean, bg_std, threshold = estimate_local_threshold(plane, bg_vals)
    active = compute_active_mask(plane, source_mask, threshold)

    spec_sum, spec_peak, source_mean, source_max = compute_flux_features_for_channel(src_vals)
    contrast = compute_contrast_for_channel(source_mean, bg_mean, bg_std)
    centroid_dx, centroid_dy, valid_centroid = compute_centroid_for_channel(
        plane=plane,
        active=active,
        xx=xx,
        yy=yy,
        x=x,
        y=y,
        threshold=threshold,
    )

    return {
        # Flujo espectral por canal
        "spec_sum": spec_sum,
        "spec_peak": spec_peak,
        # Área activa por canal
        "area": int(np.count_nonzero(active)),
        "active_mask": active,
        # Contraste fuente/fondo por canal
        "source_mean": source_mean,
        "source_max": source_max,
        "background_std": bg_std,
        "contrast": contrast,
        # Centroide por canal
        "centroid_dx": centroid_dx,
        "centroid_dy": centroid_dy,
        "valid_centroid": valid_centroid,
    }


# -----------------------------------------------------------------------------
# 4. Agregación de features: de medidas por canal a una fila tabular
# -----------------------------------------------------------------------------
def aggregate_flux_features(spec_sum: list[float], spec_peak: list[float], z0: int, z1: int, z: float) -> dict[str, float]:
    """
    Final flux features.

    Input:
    - spec_sum: one total-flux value per z-channel.
    - spec_peak: one peak-intensity value per z-channel.

    Output features:
    - spec_flux_sum_* describe accumulated signal in the source region.
    - spec_flux_peak_* describe point-like maximum intensity.
    """
    spec_sum_arr = np.asarray(spec_sum, dtype=float)
    spec_peak_arr = np.asarray(spec_peak, dtype=float)

    spec_mean = safe_nanmean(spec_sum_arr)
    spec_std = safe_nanstd(spec_sum_arr)

    # Relative channel position of the strongest total-flux channel.
    rel_channels = np.arange(z0, z1) - z
    if np.isfinite(spec_sum_arr).any():
        argmax_rel = float(rel_channels[int(np.nanargmax(spec_sum_arr))])
    else:
        argmax_rel = np.nan

    # SNR-like descriptor: how much the maximum stands out from the flux curve.
    # It is NOT the official SoFiA SNR.
    spec_snr_like = (
        (safe_nanmax(spec_sum_arr) - spec_mean) / spec_std
        if np.isfinite(spec_std) and spec_std > 0
        else np.nan
    )

    return {
        "spec_flux_sum_max": safe_nanmax(spec_sum_arr),
        "spec_flux_sum_mean": spec_mean,
        "spec_flux_sum_std": spec_std,
        "spec_flux_sum_argmax_rel": argmax_rel,
        "spec_flux_sum_snr_like": spec_snr_like,
        "spec_flux_peak_max": safe_nanmax(spec_peak_arr),
        "spec_flux_peak_mean": safe_nanmean(spec_peak_arr),
    }


def aggregate_area_features(areas: list[int]) -> dict[str, float | int]:
    """Final active-area features from per-channel active-pixel counts."""
    areas_arr = np.asarray(areas, dtype=float)
    active_channels = areas_arr > 0

    return {
        "area_mean": safe_nanmean(areas_arr),
        "area_max": safe_nanmax(areas_arr),
        "area_std": safe_nanstd(areas_arr),
        "area_n_active_channels": int(np.count_nonzero(active_channels)),
        "area_fraction_active_channels": (
            float(np.count_nonzero(active_channels) / len(areas_arr)) if len(areas_arr) else np.nan
        ),
    }


def aggregate_centroid_features(
    centroid_dx: list[float],
    centroid_dy: list[float],
    valid_centroids: list[bool],
) -> dict[str, float]:
    """
    Final centroid/drift features.

    Important detail:
    - centroid_drift_total is the NET distance between the first and last valid
      centroid, not the accumulated path length.
    - centroid_drift_mean_step is the mean distance between consecutive valid
      centroids.
    """
    cdx = np.asarray(centroid_dx, dtype=float)
    cdy = np.asarray(centroid_dy, dtype=float)
    valid_centroid_mask = np.asarray(valid_centroids, dtype=bool)

    valid_points = np.column_stack([cdx[valid_centroid_mask], cdy[valid_centroid_mask]])

    if valid_points.shape[0] >= 2:
        steps = np.sqrt(np.sum(np.diff(valid_points, axis=0) ** 2, axis=1))
        centroid_drift_total = float(np.sqrt(np.sum((valid_points[-1] - valid_points[0]) ** 2)))
        centroid_drift_mean_step = float(np.nanmean(steps))
    else:
        centroid_drift_total = np.nan
        centroid_drift_mean_step = np.nan

    return {
        "centroid_dx_std": safe_nanstd(cdx),
        "centroid_dy_std": safe_nanstd(cdy),
        "centroid_drift_total": centroid_drift_total,
        "centroid_drift_mean_step": centroid_drift_mean_step,
        "centroid_valid_fraction": (
            float(np.count_nonzero(valid_centroid_mask) / len(valid_centroid_mask))
            if len(valid_centroid_mask)
            else np.nan
        ),
    }


def aggregate_continuity_features(active_masks: list[np.ndarray]) -> dict[str, float | int]:
    """
    Final continuity features from consecutive active masks.

    For each pair of consecutive z-channels, compute an IoU-like overlap:

        overlap = intersection(active_t, active_t+1) / union(active_t, active_t+1)

    spectral_continuity_score combines:
    - mean overlap;
    - fraction of valid pairs.
    """
    overlaps: list[float] = []

    for prev, curr in zip(active_masks[:-1], active_masks[1:]):
        union = np.count_nonzero(prev | curr)
        if union == 0:
            continue
        overlaps.append(float(np.count_nonzero(prev & curr) / union))

    overlap_arr = np.asarray(overlaps, dtype=float)
    valid_pairs = int(overlap_arr.size)
    possible_pairs = max(len(active_masks) - 1, 1)

    return {
        "overlap_mean": safe_nanmean(overlap_arr),
        "overlap_std": safe_nanstd(overlap_arr),
        "overlap_min": float(np.nanmin(overlap_arr)) if overlap_arr.size else np.nan,
        "overlap_valid_pairs": valid_pairs,
        "spectral_continuity_score": (
            float(safe_nanmean(overlap_arr) * valid_pairs / possible_pairs) if overlap_arr.size else 0.0
        ),
    }


def aggregate_local_contrast_features(
    contrast_values: list[float],
    background_std_values: list[float],
    source_mean_values: list[float],
    source_max_values: list[float],
) -> dict[str, float]:
    """Final local source/background contrast features."""
    return {
        "local_contrast_mean": safe_nanmean(np.asarray(contrast_values, dtype=float)),
        "local_contrast_max": safe_nanmax(np.asarray(contrast_values, dtype=float)),
        "local_background_std": safe_nanmean(np.asarray(background_std_values, dtype=float)),
        "local_source_mean": safe_nanmean(np.asarray(source_mean_values, dtype=float)),
        "local_source_max": safe_nanmax(np.asarray(source_max_values, dtype=float)),
    }


# -----------------------------------------------------------------------------
# 5. Feature extraction completa para un candidato
# -----------------------------------------------------------------------------
def extract_one(row: pd.Series, cube: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Extract all spectral/local features for one SoFiA candidate.

    This is the main function for a single row of the input catalogue.
    """
    x, y, z, xi, yi, zi = get_candidate_position(row)

    window_xy = int(cfg["window_xy"])
    window_z = int(cfg["window_z"])
    src_radius = float(cfg["source_radius_px"])
    bg_inner = float(cfg["background_inner_radius_px"])
    bg_outer = float(cfg["background_outer_radius_px"])

    x0, x1, y0, y1, z0, z1, edge_clipped = compute_window_bounds(
        xi=xi,
        yi=yi,
        zi=zi,
        cube_shape=cube.shape,
        window_xy=window_xy,
        window_z=window_z,
    )

    # Default output if something prevents feature extraction.
    result: dict[str, Any] = {
        "feature_extraction_ok": False,
        "edge_clipped": bool(edge_clipped),
        "n_valid_channels": 0,
    }

    subcube = extract_local_subcube(cube, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
    if subcube is None:
        return result

    source_mask, background_mask, xx, yy = build_spatial_masks(
        x=x,
        y=y,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        src_radius=src_radius,
        bg_inner=bg_inner,
        bg_outer=bg_outer,
    )

    if not np.any(source_mask):
        return result

    # Per-channel containers. Each list will have one element per z-channel.
    spec_sum: list[float] = []
    spec_peak: list[float] = []
    areas: list[int] = []
    centroid_dx: list[float] = []
    centroid_dy: list[float] = []
    valid_centroids: list[bool] = []
    active_masks: list[np.ndarray] = []
    contrast_values: list[float] = []
    background_std_values: list[float] = []
    source_mean_values: list[float] = []
    source_max_values: list[float] = []

    # ---------------------------------------------------------------------
    # Main per-channel loop
    # ---------------------------------------------------------------------
    # `plane` is a 2D slice of the local subcube: one z-channel.
    for plane in subcube:
        ch = compute_channel_measurements(
            plane=plane,
            source_mask=source_mask,
            background_mask=background_mask,
            xx=xx,
            yy=yy,
            x=x,
            y=y,
        )

        spec_sum.append(ch["spec_sum"])
        spec_peak.append(ch["spec_peak"])
        areas.append(ch["area"])
        active_masks.append(ch["active_mask"])
        centroid_dx.append(ch["centroid_dx"])
        centroid_dy.append(ch["centroid_dy"])
        valid_centroids.append(ch["valid_centroid"])
        contrast_values.append(ch["contrast"])
        background_std_values.append(ch["background_std"])
        source_mean_values.append(ch["source_mean"])
        source_max_values.append(ch["source_max"])

    spec_sum_arr = np.asarray(spec_sum, dtype=float)
    valid_channels = int(np.count_nonzero(np.isfinite(spec_sum_arr)))

    # ---------------------------------------------------------------------
    # Aggregate all feature groups
    # ---------------------------------------------------------------------
    result.update(aggregate_flux_features(spec_sum, spec_peak, z0, z1, z))
    result.update(aggregate_area_features(areas))
    result.update(aggregate_centroid_features(centroid_dx, centroid_dy, valid_centroids))
    result.update(aggregate_continuity_features(active_masks))
    result.update(
        aggregate_local_contrast_features(
            contrast_values,
            background_std_values,
            source_mean_values,
            source_max_values,
        )
    )

    # Extraction-control features.
    result.update(
        {
            "feature_extraction_ok": True,
            "edge_clipped": bool(edge_clipped),
            "n_valid_channels": valid_channels,
        }
    )

    return result


# -----------------------------------------------------------------------------
# 6. CLI: procesar un catálogo completo
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--catalog-key", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    cfg = resolve_catalog_config(load_config(args.config), args.catalog_key)
    cube_path = Path(cfg["cube_path"])
    catalog_path = Path(cfg["candidate_catalog"])
    output_path = Path(cfg["output_features"])

    missing = [str(p) for p in [cube_path, catalog_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Required input path(s) not found: " + ", ".join(missing))

    candidates = pd.read_csv(catalog_path)
    required = {"x", "y", "z"}
    missing_cols = sorted(required - set(candidates.columns))
    if missing_cols:
        raise ValueError(f"Candidate catalogue missing required columns: {missing_cols}")

    if args.max_candidates is not None:
        candidates = candidates.head(args.max_candidates).copy()

    try:
        from tqdm import tqdm

        iterator = tqdm(candidates.iterrows(), total=len(candidates), desc="Extracting spectral features")
    except Exception:
        iterator = candidates.iterrows()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(cube_path, memmap=True) as hdul:
        cube = first_data_hdu(hdul)
        rows: list[dict[str, Any]] = []

        for idx, row in iterator:
            # Keep candidate identity/labels from the original SoFiA catalogue.
            base = {"candidate_index": int(idx)}
            for col in IDENTIFIER_COLUMNS:
                if col in candidates.columns:
                    base[col] = row[col]

            # Add the new spectral/local features.
            base.update(extract_one(row, cube, cfg))
            rows.append(base)

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)

    print(f"Wrote spectral features: {output_path}")
    print(f"Catalog key: {cfg.get('catalog_key', 'default')}")
    print(f"Rows: {len(out)}")
    print(f"feature_extraction_ok: {int(out['feature_extraction_ok'].sum())}/{len(out)}")


if __name__ == "__main__":
    main()
