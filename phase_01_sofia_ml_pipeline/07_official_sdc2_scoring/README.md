# 07 - Scoring oficial SDC2

## Objetivo

Convertir estrategias seleccionadas al formato SDC2 y evaluarlas con el scorer
oficial.

## Entradas

- `../06_catalog_strategy_comparison/outputs/selected_for_scoring/selected_for_08.yaml`
- Catálogos raw y post-filtrados.
- Truth catalogue indicado en `config.yaml`.

## Scripts

1. `01_build_submission_catalogs.py`: genera submissions y su manifest CSV.
2. `02_score_with_official_sdc2.py`: ejecuta el scorer y guarda resultados CSV.
3. `03_make_final_scoring_report.py`: utilidad opcional para figuras.

`utils.py` contiene conversión de columnas, lectura de selección y persistencia.

## Salidas

- `outputs/catalog_versions/catalog_versions_manifest.csv`
- `outputs/submissions/`
- `outputs/official_sdc2_scores/official_scores.csv`
- figuras en `outputs/report_figures/`

## Notas

Esta es la única subfase que produce score oficial SDC2. No entrena modelos ni
recalcula thresholds.
