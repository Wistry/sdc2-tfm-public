# Scoring de catálogos SoFiA raw

## Objetivo

Normalizar las coordenadas de los catálogos por tile, fusionar la región externa y calcular el score oficial de los catálogos raw.

## Entrada

- Catálogos SoFiA generados por tiles.
- Truth catalogue externo.
- Scorer oficial SDC2.

## Scripts

1. `01_inspect_tile_catalog_coordinates.py`
2. `02_merge_tile_catalogs_external.py`
3. `03_score_merged_external_tile_catalogs.py`

## Salida

- Auditoría de coordenadas en CSV.
- Catálogos fusionados y resumen CSV.
- Submissions y scores oficiales en CSV.

La inspección de coordenadas debe completarse antes de fusionar catálogos locales y globales.
