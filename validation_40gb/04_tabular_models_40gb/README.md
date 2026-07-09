# Modelos tabulares sobre 40 GB

## Objetivo

Extraer las features de Fase 2 sobre candidatos de la región ampliada, aplicar modelos tabulares congelados y evaluar los catálogos filtrados.

## Entrada

- Catálogos fusionados de la región ampliada.
- Cubo FITS de 40 GB.
- Modelos, columnas y thresholds congelados de Fase 2.
- Truth de la región ampliada y scorer oficial.

## Scripts

1. `01_extract_phase2_features_external_tiles_40gb.py`
2. `02_apply_frozen_phase2_models_external_tiles_40gb.py`
3. `03_score_filtered_phase2_external_tiles_40gb.py`
4. `04_compare_sofia_only_vs_extended_40gb.py`

## Runner

- `run_validation_40gb_models.sh`

## Salida

- Datasets externos con features extendidas.
- Predicciones y catálogos filtrados.
- Submissions y scores oficiales.
- Comparación `sofia_only_full` frente a `extended_full` en CSV.

No se reentrenan modelos ni se seleccionan thresholds usando los scores de 40 GB.
