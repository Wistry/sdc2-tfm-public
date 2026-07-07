# Validación agrupada

## Objetivo

Evaluar modelos tabulares y CNN con particiones agrupadas para reducir el solapamiento entre candidatos asociados a una misma fuente.

## Entrada

- Dataset extendido limpio.
- Patches, etiquetas y metadatos de `08_cnn_candidate_classifier`.

## Scripts

1. `scripts/01_grouped_tabular_validation.py`
2. `scripts/02_grouped_cnn_validation.py`
3. `scripts/03_apply_grouped_cnn_to_conservative.py`
4. `scripts/04_score_grouped_cnn_catalogs.py`
5. `scripts/05_compare_grouped_vs_original.py`

## Salida

- Métricas, folds, predicciones y comparaciones en CSV/JSON.
- Modelo CNN agrupado.
- Catálogos filtrados y scores oficiales en CSV.

Las métricas internas agrupadas y el score oficial se almacenan como resultados distintos.
