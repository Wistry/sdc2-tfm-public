# Evaluación agrupada final

## Objetivo

Aplicar un protocolo agrupado común a los modelos tabulares y CNN seleccionados y consolidar sus artefactos estructurados.

## Entrada

- Dataset extendido limpio.
- Patches y grupos de candidatos.
- Configuraciones seleccionadas en los pasos anteriores.

## Scripts

1. `scripts/01_final_grouped_tabular_evaluation.py`
2. `scripts/02_final_grouped_cnn_evaluation.py`
3. `scripts/03_apply_best_grouped_models.py`
4. `scripts/04_score_best_grouped_catalogs.py`
5. `scripts/05_final_grouped_summary.py`

## Salida

- Métricas y predicciones tabulares/CNN en CSV/JSON.
- Modelos congelados.
- Catálogos filtrados y scores oficiales.
- `final_grouped_comparison.csv`.

La tabla de comparación conserva separadas las métricas internas y las métricas del scorer oficial.
