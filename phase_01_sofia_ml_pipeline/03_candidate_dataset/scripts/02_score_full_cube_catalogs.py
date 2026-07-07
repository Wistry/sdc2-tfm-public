from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


DEMO_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = DEMO_DIR / "outputs"

FITS_PATH = PROJECT_ROOT / "data" / "sky_dev_v2.fits"
TRUTH_PATH = PROJECT_ROOT / "data" / "sky_dev_truthcat_v2.txt"
SCORING_CSV = OUTPUTS_DIR / "scoring_full_cube.csv"

MATCH_DX = 20.0
MATCH_DY = 20.0
MATCH_DZ = 50.0
CLEAN_FP_MIN_DIST = 100.0
Z_OFFSET = 0.0

CATALOGS = {
    "baseline_current_full": OUTPUTS_DIR / "baseline_current_full" / "baseline_current_full_cat.txt",
    "sdc2_team_sofia_like_full": OUTPUTS_DIR / "sdc2_team_sofia_like_full" / "sdc2_team_sofia_like_full_cat.txt",
}

SOFIA_COLUMNS = [
    "name", "id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max",
    "z_min", "z_max", "n_pix", "f_min", "f_max", "f_sum", "rel", "flag",
    "rms", "w20", "w50", "wm50", "z_w20", "z_w50", "z_wm50", "ell_maj",
    "ell_min", "ell_pa", "ell3s_maj", "ell3s_min", "ell3s_pa", "kin_pa",
    "err_x", "err_y", "err_z", "err_f_sum", "snr", "snr_max", "ra", "dec",
    "freq", "x_peak", "y_peak", "z_peak", "ra_peak", "dec_peak", "freq_peak",
]


def require_inputs() -> None:
    missing = [str(path) for path in [FITS_PATH, TRUTH_PATH, *CATALOGS.values()] if not path.exists()]
    if missing:
        raise SystemExit("Faltan inputs:\n" + "\n".join(missing))


def read_sofia_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="c",
        quotechar='"',
    )

    if df.empty:
        return pd.DataFrame(columns=SOFIA_COLUMNS)

    if df.shape[1] != len(SOFIA_COLUMNS):
        raise ValueError(
            f"{path} tiene {df.shape[1]} columnas, pero se esperaban "
            f"{len(SOFIA_COLUMNS)}. Revisa el parseo del catálogo SoFiA."
        )

    df.columns = SOFIA_COLUMNS

    for col in df.columns:
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} no contiene columnas SoFiA esperadas: {sorted(missing)}")

    return df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)


def read_truth_pixels() -> pd.DataFrame:
    truth_df = pd.read_csv(TRUTH_PATH, sep=r"\s+", comment="#", engine="python")
    required = {"ra", "dec", "central_freq"}
    missing = required.difference(truth_df.columns)
    if missing:
        raise SystemExit(f"Truth catalogue sin columnas esperadas: {sorted(missing)}")

    header = fits.getheader(FITS_PATH)
    wcs = WCS(header)
    truth_x, truth_y, truth_z = wcs.world_to_pixel_values(
        truth_df["ra"].to_numpy(),
        truth_df["dec"].to_numpy(),
        truth_df["central_freq"].to_numpy(),
    )
    truth_df = truth_df.copy()
    truth_df["truth_x"] = truth_x
    truth_df["truth_y"] = truth_y
    truth_df["truth_z"] = truth_z
    in_cube = (
        truth_df["truth_x"].between(0, 642)
        & truth_df["truth_y"].between(0, 642)
        & truth_df["truth_z"].between(0, 6667)
    )
    return truth_df[in_cube].reset_index(drop=True)


def f_beta(reliability: float, completeness: float, beta: float) -> float:
    if reliability <= 0.0 and completeness <= 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * reliability * completeness / (beta2 * reliability + completeness)


def score_catalog(config: str, catalog: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int | str]:
    n_candidates = len(catalog)
    n_truth = len(truth)
    if n_candidates == 0:
        return {
            "config": config,
            "n_candidates": 0,
            "tp": 0,
            "fp": 0,
            "ambiguous": 0,
            "fn": n_truth,
            "reliability": 0.0,
            "completeness": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "fp_per_tp": np.nan,
            "matched_truth_unique": 0,
            "median_abs_dz_tp": np.nan,
            "candidate_z_min": np.nan,
            "candidate_z_max": np.nan,
        }

    truth_xyz = truth[["truth_x", "truth_y", "truth_z"]].to_numpy(dtype=float)
    matched_truth = set()
    tp = 0
    fp = 0
    ambiguous = 0
    tp_abs_dz = []

    for _, row in catalog.iterrows():
        candidate = np.asarray([row["x"], row["y"], row["z"] + Z_OFFSET], dtype=float)
        delta = np.abs(truth_xyz - candidate)
        dist = np.sqrt((delta * delta).sum(axis=1))
        nearest = int(np.nanargmin(dist))
        is_tp = (
            delta[nearest, 0] <= MATCH_DX
            and delta[nearest, 1] <= MATCH_DY
            and delta[nearest, 2] <= MATCH_DZ
        )
        if is_tp:
            tp += 1
            matched_truth.add(nearest)
            tp_abs_dz.append(float(delta[nearest, 2]))
        elif float(dist[nearest]) > CLEAN_FP_MIN_DIST:
            fp += 1
        else:
            ambiguous += 1

    reliability = tp / n_candidates if n_candidates else 0.0
    completeness = len(matched_truth) / n_truth if n_truth else 0.0
    return {
        "config": config,
        "n_candidates": n_candidates,
        "tp": tp,
        "fp": fp,
        "ambiguous": ambiguous,
        "fn": max(n_truth - len(matched_truth), 0),
        "reliability": reliability,
        "completeness": completeness,
        "f1": f_beta(reliability, completeness, beta=1.0),
        "f2": f_beta(reliability, completeness, beta=2.0),
        "fp_per_tp": fp / tp if tp else np.nan,
        "matched_truth_unique": len(matched_truth),
        "median_abs_dz_tp": float(np.median(tp_abs_dz)) if tp_abs_dz else np.nan,
        "candidate_z_min": float(catalog["z"].min()),
        "candidate_z_max": float(catalog["z"].max()),
    }


def label_catalog(catalog: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Return one row per SoFiA candidate with clean TP/FP/ambiguous labels."""
    labelled = catalog.copy()
    for column in [
        "clean_label",
        "label",
        "is_ambiguous",
        "matched_truth_id",
        "truth_row",
        "truth_x",
        "truth_y",
        "truth_z",
        "min_abs_dx",
        "min_abs_dy",
        "min_abs_dz",
        "min_dist_3d",
        "matching_mode",
    ]:
        labelled[column] = np.nan

    if catalog.empty or truth.empty:
        labelled["clean_label"] = 0
        labelled["label"] = "FP"
        labelled["is_ambiguous"] = False
        labelled["matching_mode"] = "no_truth"
        return labelled

    truth_xyz = truth[["truth_x", "truth_y", "truth_z"]].to_numpy(dtype=float)
    truth_ids = truth["id"].to_numpy() if "id" in truth.columns else np.arange(len(truth))

    for index, row in catalog.iterrows():
        candidate = np.asarray([row["x"], row["y"], row["z"] + Z_OFFSET], dtype=float)
        delta = np.abs(truth_xyz - candidate)
        dist = np.sqrt((delta * delta).sum(axis=1))
        nearest = int(np.nanargmin(dist))
        is_tp = (
            delta[nearest, 0] <= MATCH_DX
            and delta[nearest, 1] <= MATCH_DY
            and delta[nearest, 2] <= MATCH_DZ
        )
        is_clean_fp = float(dist[nearest]) > CLEAN_FP_MIN_DIST

        if is_tp:
            clean_label = 1
            label = "TP"
            matching_mode = "tp_match"
            is_ambiguous = False
        elif is_clean_fp:
            clean_label = 0
            label = "FP"
            matching_mode = "clean_fp"
            is_ambiguous = False
        else:
            clean_label = -1
            label = "ambiguous"
            matching_mode = "ambiguous_near_truth"
            is_ambiguous = True

        labelled.loc[index, "clean_label"] = clean_label
        labelled.loc[index, "label"] = label
        labelled.loc[index, "is_ambiguous"] = is_ambiguous
        labelled.loc[index, "matched_truth_id"] = truth_ids[nearest]
        labelled.loc[index, "truth_row"] = nearest
        labelled.loc[index, "truth_x"] = truth.iloc[nearest]["truth_x"]
        labelled.loc[index, "truth_y"] = truth.iloc[nearest]["truth_y"]
        labelled.loc[index, "truth_z"] = truth.iloc[nearest]["truth_z"]
        labelled.loc[index, "min_abs_dx"] = float(delta[nearest, 0])
        labelled.loc[index, "min_abs_dy"] = float(delta[nearest, 1])
        labelled.loc[index, "min_abs_dz"] = float(delta[nearest, 2])
        labelled.loc[index, "min_dist_3d"] = float(dist[nearest])
        labelled.loc[index, "matching_mode"] = matching_mode

    labelled["clean_label"] = labelled["clean_label"].astype(int)
    labelled["is_ambiguous"] = labelled["is_ambiguous"].astype(bool)
    return labelled


def write_labelled_catalogs(config: str, labelled: pd.DataFrame) -> None:
    output_dir = OUTPUTS_DIR / config
    output_dir.mkdir(parents=True, exist_ok=True)
    sofia_only_path = output_dir / "candidates_sofia_only.csv"
    labelled.to_csv(sofia_only_path, index=False)


def main() -> None:
    require_inputs()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    truth = read_truth_pixels()
    rows = []
    for config, path in CATALOGS.items():
        catalog = read_sofia_catalog(path)
        write_labelled_catalogs(config, label_catalog(catalog, truth))
        rows.append(score_catalog(config, catalog, truth))
    scores = pd.DataFrame(rows).sort_values(["f2", "reliability"], ascending=False)
    scores.to_csv(SCORING_CSV, index=False)
    print(f"Scoring guardado en: {SCORING_CSV}")
    print(scores[["config", "n_candidates", "tp", "fp", "ambiguous", "reliability", "completeness", "f1", "f2", "fp_per_tp"]].to_string(index=False))


if __name__ == "__main__":
    main()
