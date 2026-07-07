# CNN agrupada robusta

## Objetivo

Entrenar una variante CNN con validación agrupada y controles adicionales de estabilidad.

## Entrada

- Patches y metadatos de candidatos.
- Grupos definidos para evitar solapamiento entre particiones.

## Scripts

1. `scripts/01_train_robust_grouped_cnn.py`
2. `scripts/02_apply_robust_grouped_cnn.py`
3. `scripts/03_score_robust_grouped_cnn.py`
4. `scripts/04_compare_robust_cnn_results.py`

## Salida

- Modelo, configuración y métricas estructuradas.
- Predicciones y catálogos filtrados.
- Scores oficiales y comparación en CSV.

El scorer oficial se usa después de congelar el modelo y el threshold.
