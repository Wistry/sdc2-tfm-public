# 03 - Dataset de candidatos

## Objetivo

Generar y etiquetar los catálogos completos usados por las etapas tabulares.

## Entradas

- Cubo FITS de desarrollo.
- Truth catalogue.
- Configuraciones `baseline_current_full` y `sdc2_team_sofia_like_full`.

## Scripts

1. `run_full_cube_sofia_2h.sh`: ejecuta SoFiA sobre las configuraciones completas.
2. `scripts/02_score_full_cube_catalogs.py`: construye etiquetas y métricas locales.

## Salidas

- `outputs/baseline_current_full/candidates_sofia_only.csv`
- `outputs/sdc2_team_sofia_like_full/candidates_sofia_only.csv`
- `outputs/scoring_full_cube.csv`

## Notas

`clean_label` se construye mediante matching local. Los valores `1` y `0` se
usan como TP/FP limpios; `-1` identifica candidatos ambiguos.
