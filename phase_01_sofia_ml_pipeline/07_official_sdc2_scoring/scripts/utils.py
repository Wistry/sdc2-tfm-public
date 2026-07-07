from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE_DIR / "config.yaml"

SOFIA_COLUMNS = [
    "name", "id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max",
    "z_min", "z_max", "n_pix", "f_min", "f_max", "f_sum", "rel", "flag",
    "rms", "w20", "w50", "wm50", "z_w20", "z_w50", "z_wm50", "ell_maj",
    "ell_min", "ell_pa", "ell3s_maj", "ell3s_min", "ell3s_pa", "kin_pa",
    "err_x", "err_y", "err_z", "err_f_sum", "snr", "snr_max", "ra", "dec",
    "freq", "x_peak", "y_peak", "z_peak", "ra_peak", "dec_peak", "freq_peak",
]

SUBMISSION_COLUMNS = [
    "id",
    "ra",
    "dec",
    "hi_size",
    "line_flux_integral",
    "central_freq",
    "pa",
    "i",
    "w20",
]

KEY_SOURCE_COLUMNS = ["f_sum", "ell_maj", "ell_min", "kin_pa", "w20", "ra", "dec", "freq"]


def _coerce_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            config[current_section] = {}
            current_list_key = None
            continue
        if current_section is None:
            continue
        text = line.strip()
        if text.startswith("- ") and current_list_key:
            config[current_section][current_list_key].append(_coerce_scalar(text[2:]))
            continue
        key, _, value = text.partition(":")
        key = key.strip()
        value = value.strip()
        if value:
            config[current_section][key] = _coerce_scalar(value)
            current_list_key = None
        else:
            config[current_section][key] = []
            current_list_key = key
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    try:
        import yaml  # type: ignore

        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except ImportError:
        return _simple_yaml(config_path)


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (BASE_DIR / value).resolve()


def ensure_dirs(config: dict[str, Any]) -> dict[str, Path]:
    out = config["outputs"]
    paths = {
        "catalog_versions_dir": resolve_path(out["catalog_versions_dir"]),
        "submissions_dir": resolve_path(out["submissions_dir"]),
        "official_scores_dir": resolve_path(out["official_scores_dir"]),
        "report_figures_dir": resolve_path(out["report_figures_dir"]),
        "reports_dir": resolve_path(out["reports_dir"]),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_sofia_catalog(path: str | Path, config_name: str | None = None) -> pd.DataFrame:
    catalog_path = resolve_path(path)
    df = pd.read_csv(
        catalog_path,
        sep=r"\s+",
        comment="#",
        names=SOFIA_COLUMNS,
        quotechar='"',
        engine="python",
    )
    if config_name:
        df.insert(0, "config_name", config_name)
    return df


def read_catalog_any(path: str | Path) -> pd.DataFrame:
    catalog_path = resolve_path(path)
    suffix = catalog_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(catalog_path)
    if suffix in {".txt", ".tsv"}:
        try:
            return pd.read_csv(catalog_path, sep=None, engine="python", comment="#")
        except Exception:
            pass
        for sep in ["\t", ",", r"\s+"]:
            try:
                return pd.read_csv(catalog_path, sep=sep, engine="python", comment="#")
            except Exception:
                continue
    raise ValueError(f"No se pudo leer catalogo: {catalog_path}")


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_path = resolve_path(path)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML es necesario para leer selected_for_08.yaml") from exc
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return None


def _numeric(series_or_value: pd.Series | float | int, length: int) -> pd.Series:
    if isinstance(series_or_value, pd.Series):
        return pd.to_numeric(series_or_value, errors="coerce")
    return pd.Series([series_or_value] * length, dtype="float64")


def convert_to_sdc2_submission(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    n_input = len(df)
    missing_key = [column for column in KEY_SOURCE_COLUMNS if column not in df.columns]
    if missing_key:
        print(f"WARNING: faltan columnas fisicas clave antes de convertir a SDC2: {missing_key}")
    out = pd.DataFrame(index=df.index)
    out["ra"] = _numeric(_first_existing(df, ["ra"]), n_input)
    out["dec"] = _numeric(_first_existing(df, ["dec"]), n_input)
    out["central_freq"] = _numeric(_first_existing(df, ["freq", "central_freq"]), n_input)
    out["line_flux_integral"] = _numeric(_first_existing(df, ["f_sum", "line_flux_integral"]), n_input)
    if "f_sum" not in df.columns and "line_flux_integral" not in df.columns:
        print("WARNING: falta `f_sum`/`line_flux_integral`; `line_flux_integral` quedara en 0 si no hay datos.")

    hi_size = _first_existing(df, ["hi_size"])
    if "ell_maj" in df.columns:
        out["hi_size"] = pd.to_numeric(df["ell_maj"], errors="coerce") * 4.0
    else:
        if hi_size is None:
            print("WARNING: falta `ell_maj`/`hi_size`; `hi_size` se rellenara con 0.")
        out["hi_size"] = _numeric(hi_size if hi_size is not None else 0.0, n_input)

    pa = _first_existing(df, ["kin_pa", "pa"])
    if pa is None:
        print("WARNING: falta `kin_pa`/`pa`; `pa` se rellenara con 0.")
    out["pa"] = _numeric(pa if pa is not None else 0.0, n_input)
    inclination = _first_existing(df, ["ell_min", "i"])
    if inclination is None:
        print("WARNING: falta `ell_min`/`i`; `i` se rellenara con 0.")
    out["i"] = _numeric(inclination if inclination is not None else 0.0, n_input)
    w20 = _first_existing(df, ["w20"])
    if w20 is None:
        print("WARNING: falta `w20`; se rellenara con 0.")
    out["w20"] = _numeric(w20 if w20 is not None else 0.0, n_input)

    required = ["ra", "dec", "central_freq"]
    before_required = len(out)
    out = out.dropna(subset=required).copy()
    n_dropped_rows = before_required - len(out)
    out = out.fillna(0.0)
    out.insert(0, "id", range(1, len(out) + 1))
    out = out[SUBMISSION_COLUMNS]
    diagnostics = {
        "n_input_rows": int(n_input),
        "n_submission_rows": int(len(out)),
        "n_dropped_rows": int(n_dropped_rows),
        "missing_key_columns": ",".join(missing_key),
    }
    return out, diagnostics


def deduplicate_xyz(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"x", "y", "z"}.issubset(out.columns):
        keys = out[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").round(3).astype(str)
    elif {"ra", "dec", "freq"}.issubset(out.columns):
        keys = out[["ra", "dec", "freq"]].apply(pd.to_numeric, errors="coerce").round({"ra": 6, "dec": 6, "freq": 3}).astype(str)
    elif {"ra", "dec", "central_freq"}.issubset(out.columns):
        keys = out[["ra", "dec", "central_freq"]].apply(pd.to_numeric, errors="coerce").round({"ra": 6, "dec": 6, "central_freq": 3}).astype(str)
    else:
        print("WARNING: no hay columnas suficientes para deduplicar; se conserva el catalogo completo.")
        return out
    out["_dedup_key"] = keys.agg("|".join, axis=1)
    out = out.drop_duplicates("_dedup_key", keep="first").drop(columns=["_dedup_key"])
    return out


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def short_strategy_name(name: str) -> str:
    text = name.removeprefix("ML_").removeprefix("ENS_sdc2_plus_")
    text = text.replace("balanced_accuracy", "bal")
    text = text.replace("conservative_fp", "cons")
    text = text.replace("baseline_current_full_raw", "raw_permissive")
    text = text.replace("sdc2_team_sofia_like_full_raw", "raw_conservative")
    return text


def scalar_attrs(obj: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            attrs[attr] = value
    return attrs


def write_json(obj: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")
