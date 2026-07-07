# 04 - EDA y benchmark

## Objetivo

Analizar el catálogo permisivo y comparar clasificadores antes de la
optimización final.

## Entradas

- `../03_candidate_dataset/outputs/baseline_current_full/candidates_sofia_only.csv`
- `config.yaml`

## Scripts

1. `01_eda_candidates.py`: genera tablas CSV, metadata JSON y figuras del EDA.
2. `02_benchmark_classifiers.py`: genera métricas, rankings, predicciones,
   metadata y figuras para `full` y `no_position`.
3. `make_report_official_figures.py`: utilidad opcional que genera figuras
   adicionales.

`utils.py` contiene carga de configuración, control de leakage y funciones de
persistencia compartidas por los dos scripts principales.

## Salidas

- `outputs/eda/*.csv`
- `outputs/eda/eda_summary.json`
- `outputs/benchmark/full/`
- `outputs/benchmark/no_position/`
- figuras dentro de `outputs/`

## Notas

Los ambiguos se incluyen en el EDA y se excluyen del entrenamiento. Las
métricas del benchmark son internas y no son score oficial SDC2.
