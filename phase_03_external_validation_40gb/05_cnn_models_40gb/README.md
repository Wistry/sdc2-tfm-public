# Modelos CNN sobre 40 GB

## Objetivo

Extraer patches locales, aplicar CNN congeladas y evaluar sus catálogos sobre la región externa.

## Entrada

- Cubo FITS de 40 GB y catálogos externos fusionados.
- Checkpoints, metadatos y thresholds congelados de Fase 2.
- Truth externo y scorer oficial.

## Scripts

1. `01_extract_cnn_patches_external_40gb.py`
2. `02_apply_frozen_cnn_external_40gb.py`
3. `03_score_cnn_external_40gb.py`
4. `04_apply_valid_frozen_cnns_external_40gb.py`
5. `05_score_valid_frozen_cnns_external_40gb.py`
6. `06_make_valid_cnn_external_report.py`

El sexto script conserva su nombre por compatibilidad con el runner, pero genera únicamente `valid_cnn_external_40gb_comparison.csv`.

## Runner

- `run_phase3_external_cnn.sh`

## Salida

- Patches `21 x 32 x 32` y metadatos.
- Predicciones y catálogos filtrados.
- Submissions y scores oficiales.
- Comparación estructurada de resultados CNN en CSV.

Los patches y resultados se almacenan en `../outputs/` y no se versionan. No se reentrenan redes ni se recalibran thresholds con 40 GB.
