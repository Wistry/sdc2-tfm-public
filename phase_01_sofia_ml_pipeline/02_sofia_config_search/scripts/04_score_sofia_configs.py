"""Calcula el score local de cada catálogo SoFiA contra el truth preparado.

El score no es un score interno de SoFiA. Es un matching local contra las
fuentes truth proyectadas al cubo, usando tolerancias de distancia en x/y/z.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DEMO_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = DEMO_DIR / "outputs"
CONFIG_DIR = DEMO_DIR / "configs"
CATALOGS_DIR = OUTPUTS_DIR / "catalogs"
TRUTH_PIXELS_CSV = OUTPUTS_DIR / "truth_pixels.csv"
SCORES_CSV = OUTPUTS_DIR / "scores.csv"

MATCH_DX = 20.0
MATCH_DY = 20.0
MATCH_DZ = 50.0
REGION_OFFSET_X = 0.0
REGION_OFFSET_Y = 0.0
REGION_OFFSET_Z = 5000.0

SOFIA_COLUMNS = [
    "name", "id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max",
    "z_min", "z_max", "n_pix", "f_min", "f_max", "f_sum", "rel", "flag",
    "rms", "w20", "w50", "wm50", "z_w20", "z_w50", "z_wm50", "ell_maj",
    "ell_min", "ell_pa", "ell3s_maj", "ell3s_min", "ell3s_pa", "kin_pa",
    "err_x", "err_y", "err_z", "err_f_sum", "snr", "snr_max", "ra", "dec",
    "freq", "x_peak", "y_peak", "z_peak", "ra_peak", "dec_peak", "freq_peak",
]


def require_inputs() -> None:
    if not CONFIG_DIR.exists():
        raise SystemExit(f"Falta la carpeta de configs: {CONFIG_DIR}")
    if not CATALOGS_DIR.exists():
        raise SystemExit(f"Falta carpeta de catalogos: {CATALOGS_DIR}")
    if not TRUTH_PIXELS_CSV.exists():
        raise SystemExit(
            f"Falta {TRUTH_PIXELS_CSV}. Crea un CSV pequeno con truth_x, truth_y, truth_z."
        )


def read_truth() -> pd.DataFrame:
    truth = pd.read_csv(TRUTH_PIXELS_CSV)
    required = {"truth_x", "truth_y", "truth_z"}
    missing = required.difference(truth.columns)
    if missing:
        raise SystemExit(f"Truth sin columnas requeridas: {sorted(missing)}")
    return truth


def read_sofia_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if df.empty:
        return pd.DataFrame(columns=SOFIA_COLUMNS)
    if df.shape[1] > len(SOFIA_COLUMNS):
        df = df.iloc[:, :len(SOFIA_COLUMNS)]
    df.columns = SOFIA_COLUMNS[:df.shape[1]]
    required = {"x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} no contiene columnas SoFiA esperadas: {sorted(missing)}")
    return df


def f_beta(reliability: float, completeness: float, beta: float) -> float:
    if reliability <= 0.0 and completeness <= 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * reliability * completeness / (beta2 * reliability + completeness)


def score_catalog(config_name: str, catalog: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | str]:
    n_candidates = len(catalog)
    n_truth = len(truth)
    if n_candidates == 0:
        return {
            "config": config_name,
            "n_candidates": 0,
            "tp": 0,
            "fp": 0,
            "fn": n_truth,
            "reliability": 0.0,
            "completeness": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "fp_per_tp": np.nan,
        }

    truth_xyz = truth[["truth_x", "truth_y", "truth_z"]].to_numpy(dtype=float)
    matched_truth = set()
    tp_candidates = 0

    for _, row in catalog.iterrows():
        candidate = np.asarray([
            row["x"] + REGION_OFFSET_X,
            row["y"] + REGION_OFFSET_Y,
            row["z"] + REGION_OFFSET_Z,
        ], dtype=float)
        delta = np.abs(truth_xyz - candidate)
        nearest = int(np.nanargmin(np.sqrt((delta * delta).sum(axis=1))))
        is_match = (
            delta[nearest, 0] <= MATCH_DX
            and delta[nearest, 1] <= MATCH_DY
            and delta[nearest, 2] <= MATCH_DZ
        )
        if is_match:
            tp_candidates += 1
            matched_truth.add(nearest)

    fp = n_candidates - tp_candidates
    fn = max(n_truth - len(matched_truth), 0)
    reliability = tp_candidates / n_candidates if n_candidates else 0.0
    completeness = len(matched_truth) / n_truth if n_truth else 0.0
    return {
        "config": config_name,
        "n_candidates": n_candidates,
        "tp": tp_candidates,
        "fp": fp,
        "fn": fn,
        "reliability": reliability,
        "completeness": completeness,
        "f1": f_beta(reliability, completeness, beta=1.0),
        "f2": f_beta(reliability, completeness, beta=2.0),
        "fp_per_tp": fp / tp_candidates if tp_candidates else np.nan,
    }


def main() -> None:
    require_inputs()
    truth = read_truth()
    active_configs = {path.stem for path in CONFIG_DIR.glob("*.par")}
    catalog_paths = [
        path for path in sorted(CATALOGS_DIR.glob("*/*_cat.txt"))
        if path.parent.name in active_configs
    ]
    if not catalog_paths:
        raise SystemExit(
            f"No hay catalogos *_cat.txt válidos en {CATALOGS_DIR}. "
            "Ejecuta SoFiA manualmente primero."
        )

    rows = []
    errors = []
    for path in catalog_paths:
        config_name = path.name.removesuffix("_cat.txt")
        try:
            catalog = read_sofia_catalog(path)
            rows.append(score_catalog(config_name, catalog, truth))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not rows:
        raise SystemExit("No se pudo puntuar ningun catalogo:\n" + "\n".join(errors))

    scores = pd.DataFrame(rows).sort_values(["f2", "reliability"], ascending=False)
    scores.to_csv(SCORES_CSV, index=False)
    print(f"Scores guardados en: {SCORES_CSV}")
    if errors:
        print("Catalogos omitidos:")
        print("\n".join(errors))


if __name__ == "__main__":
    main()
