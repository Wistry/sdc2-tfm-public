from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils import ensure_dirs, load_config, resolve_path, scalar_attrs, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el scorer oficial SDC2 sobre submissions generadas.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--truth-file", type=Path, default=None)
    return parser.parse_args()


def import_scorer():
    try:
        from ska_sdc.sdc2.sdc2_scorer import Sdc2Scorer
    except ImportError as exc:
        raise SystemExit(
            "ERROR: no se pudo importar `ska_sdc.sdc2.sdc2_scorer.Sdc2Scorer`. "
            "Instala/activa el entorno con el paquete oficial de scoring SDC2."
        ) from exc
    return Sdc2Scorer


def score_value_from(score: Any) -> float | None:
    if isinstance(score, (int, float)):
        return float(score)
    for attr in ["value", "score", "score_value"]:
        if hasattr(score, attr):
            value = getattr(score, attr)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    manifest_path = paths["catalog_versions_dir"] / "catalog_versions_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Falta {manifest_path}. Ejecuta primero 01_build_submission_catalogs.py")

    truth_path = Path(args.truth_file) if args.truth_file else resolve_path(config["data"]["truth_file"])
    if not truth_path.exists():
        raise SystemExit(f"ERROR: no existe truth catalogue esperado: {truth_path}")

    Sdc2Scorer = import_scorer()
    manifest = pd.read_csv(manifest_path)
    df_truth = pd.read_csv(truth_path, sep=r"\s+", comment="#", engine="python")
    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        catalog_name = str(item["catalog_name"])
        submission_path = Path(str(item["submission_path"]))
        row: dict[str, Any] = {
            "catalog_name": catalog_name,
            "source_type": item.get("source_type", item.get("catalog_type", "")),
            "selection_role": item.get("selection_role", ""),
            "optional": bool(item.get("optional", False)),
            "status": "ERROR",
            "n_rows": int(item.get("n_rows", 0)),
            "n_submission": int(item.get("n_submission_rows", 0)),
            "score_value": None,
            "n_match": None,
            "n_false": None,
            "score_per_candidate": None,
            "submission_path": str(submission_path),
            "error": "",
        }
        try:
            submission = pd.read_csv(submission_path)
            score = Sdc2Scorer(submission, df_truth).run()
            attrs = scalar_attrs(score)
            score_value = score_value_from(score)
            n_match = attrs.get("n_match")
            n_false = attrs.get("n_false")
            row.update(
                {
                    "status": "OK",
                    "n_submission": int(len(submission)),
                    "score_value": score_value,
                    "n_match": n_match,
                    "n_false": n_false,
                    "score_per_candidate": score_value / len(submission) if score_value is not None and len(submission) else None,
                    "score_attrs_json": json.dumps(attrs, ensure_ascii=True, sort_keys=True),
                }
            )
        except Exception as exc:  # noqa: BLE001 - scoring must continue for remaining catalogues.
            row["error"] = str(exc)
            row["score_attrs_json"] = "{}"
        rows.append(row)

    scores = pd.DataFrame(rows)
    csv_path = paths["official_scores_dir"] / "official_scores.csv"
    write_dataframe(scores, csv_path)
    print(f"Scores: {csv_path}")


if __name__ == "__main__":
    main()
