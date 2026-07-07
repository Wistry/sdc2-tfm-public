# Auditoría de features extendidas

## Objetivo

Comprobar tipo, valores ausentes, cardinalidad, constancia, separación TP/FP y posibles columnas de identificación o leakage.

## Entrada

- Datasets extendidos generados en `02_build_extended_datasets`.

## Script

- `scripts/03_audit_spectral_features.py`

## Salida

- `outputs/reports/spectral_feature_audit_<catalogo>.csv`

Cada fila describe una feature y contiene los controles numéricos utilizados por la auditoría. Este paso no entrena modelos ni modifica datasets.
