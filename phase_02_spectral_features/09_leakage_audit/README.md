# Auditoría de leakage

## Objetivo

Comprobar duplicados, solapamientos entre particiones, reutilización de truth IDs y uso del scorer durante la selección.

## Entrada

- Datasets, predicciones y metadatos generados por los pipelines tabular y CNN.

## Script

- `scripts/01_audit_leakage_risks.py`

## Salida

- `duplicate_truth_groups.csv`
- `cnn_train_test_group_overlap.csv`
- `scorer_usage_audit.csv`

La auditoría es estática sobre artefactos existentes y no modifica modelos ni resultados.
