from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    KEY_SOURCE_COLUMNS,
    convert_to_sdc2_submission,
    deduplicate_xyz,
    ensure_dirs,
    load_config,
    load_yaml_file,
    read_catalog_any,
    read_sofia_catalog,
    resolve_path,
    write_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye catalogos y submissions para scoring SDC2.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def source_warning(df: pd.DataFrame) -> str:
    missing = [column for column in KEY_SOURCE_COLUMNS if column not in df.columns]
    if missing:
        return "faltan columnas clave: " + ",".join(missing)
    zero_like = []
    for column in ["f_sum", "ell_maj", "w20"]:
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").fillna(0).eq(0).all():
            zero_like.append(column)
    if zero_like:
        return "columnas fisicas todo cero o nulas: " + ",".join(zero_like)
    return ""


def add_catalog(
    rows: list[dict],
    name: str,
    source_type: str,
    selection_role: str,
    optional: bool,
    df: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    warning = source_warning(df)
    if warning:
        print(f"WARNING fuerte en {name}: {warning}")
    catalog_path = paths["catalog_versions_dir"] / f"{name}.csv"
    submission_path = paths["submissions_dir"] / f"{name}_submission.csv"
    write_dataframe(df, catalog_path)
    submission, diagnostics = convert_to_sdc2_submission(df)
    write_dataframe(submission, submission_path)
    rows.append(
        {
            "catalog_name": name,
            "source_type": source_type,
            "selection_role": selection_role,
            "optional": bool(optional),
            "n_rows": int(len(df)),
            "catalog_path": str(catalog_path),
            "submission_path": str(submission_path),
            "n_submission_rows": diagnostics["n_submission_rows"],
            "n_dropped_rows": diagnostics["n_dropped_rows"],
            "has_f_sum": "f_sum" in df.columns,
            "has_ell_maj": "ell_maj" in df.columns,
            "has_w20": "w20" in df.columns,
            "warning": warning,
        }
    )


def load_selected_entries(config: dict) -> list[dict]:
    selected_path = config["data"].get("selected_strategies_path")
    if not selected_path:
        raise SystemExit("Falta data.selected_strategies_path en config.yaml")
    path = resolve_path(selected_path)
    if not path.exists():
        raise SystemExit(f"Falta seleccion del paso 06: {path}. Ejecuta 06/scripts/05_select_strategies_for_scoring.py")
    payload = load_yaml_file(path)
    entries = list(payload.get("selected_strategies", []))
    entries.extend(config.get("strategies", {}).get("selected_strategies_extra", []))
    return entries


def read_entry_catalog(entry: dict, config: dict) -> pd.DataFrame | None:
    data_cfg = config["data"]
    strategy_name = entry["strategy_name"]
    source_type = entry["source_type"]
    if source_type == "raw":
        if strategy_name == "A_baseline_current_full_raw":
            return read_sofia_catalog(data_cfg["baseline_raw_catalog"], "baseline_current_full")
        if strategy_name == "B_sdc2_team_sofia_like_full_raw":
            return read_sofia_catalog(data_cfg["sdc2_raw_catalog"], "sdc2_team_sofia_like_full")
        print(f"WARNING: raw strategy no reconocida: {strategy_name}")
        return None
    if source_type == "accepted":
        accepted_file = entry.get("accepted_file")
        if not accepted_file:
            print(f"WARNING: {strategy_name} no define accepted_file.")
            return None
        accepted_path = resolve_path(accepted_file)
        if not accepted_path.exists():
            print(f"WARNING: no existe accepted_file para {strategy_name}: {accepted_path}")
            return None
        return read_catalog_any(accepted_path)
    print(f"WARNING: source_type no reconocido para {strategy_name}: {source_type}")
    return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    strat_cfg = config["strategies"]
    rows: list[dict] = []
    loaded_for_ensemble: dict[str, pd.DataFrame] = {}

    if strat_cfg.get("use_selected_from_06", True):
        entries = load_selected_entries(config)
        include_optional = bool(strat_cfg.get("include_optional", True))
        for entry in entries:
            if bool(entry.get("optional", False)) and not include_optional:
                print(f"Saltando estrategia opcional: {entry['strategy_name']}")
                continue
            df = read_entry_catalog(entry, config)
            if df is None:
                continue
            name = entry["strategy_name"]
            loaded_for_ensemble[name] = df
            add_catalog(
                rows,
                name,
                entry["source_type"],
                entry.get("selection_role", ""),
                bool(entry.get("optional", False)),
                df,
                paths,
            )
    else:
        raise SystemExit("Este refactor espera strategies.use_selected_from_06: true")

    if strat_cfg.get("build_ensembles_with_sdc2_raw", False):
        sdc2 = loaded_for_ensemble.get("B_sdc2_team_sofia_like_full_raw")
        if sdc2 is None:
            sdc2 = read_sofia_catalog(config["data"]["sdc2_raw_catalog"], "sdc2_team_sofia_like_full")
        for name, df in loaded_for_ensemble.items():
            if not name.startswith("ML_"):
                continue
            ensemble = deduplicate_xyz(pd.concat([sdc2, df], ignore_index=True, sort=False))
            add_catalog(
                rows,
                f"ENS_sdc2_plus_{name.removeprefix('ML_')}",
                "ensemble",
                f"ensemble_{name}",
                False,
                ensemble,
                paths,
            )

    manifest = pd.DataFrame(rows)
    manifest_path = paths["catalog_versions_dir"] / "catalog_versions_manifest.csv"
    write_dataframe(manifest, manifest_path)
    print(f"Manifest: {manifest_path}")
    if not manifest.empty:
        print(manifest[["catalog_name", "source_type", "selection_role", "optional", "n_rows", "n_submission_rows", "n_dropped_rows", "warning"]].to_string(index=False))


if __name__ == "__main__":
    main()
