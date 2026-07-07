# Comparación final de pipelines CNN

## Objetivo

Comparar las variantes CNN seleccionadas bajo un protocolo común de datos, partición y threshold.

## Entrada

- Patches, etiquetas y metadatos.
- Configuraciones y artefactos de las CNN evaluadas en los pasos anteriores.

## Script

- `scripts/01_compare_cnn15_vs_cnn16.py`

## Salida

- Resultados por fold y predicciones en CSV.
- Barrido común de thresholds en CSV.
- Tabla final de comparación en CSV.
- Modelos y catálogos filtrados cuando corresponda.

La comparación no sustituye la evaluación externa de `phase_03_external_validation_40gb`.
