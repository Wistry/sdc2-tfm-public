# Scoring oficial SDC2

## Objetivo

Convertir los catálogos filtrados al esquema requerido y evaluarlos con el scorer oficial SDC2.

## Entrada

- Catálogos producidos por `06_apply_to_conservative_catalog`.
- Truth catalogue y scorer oficial configurados externamente.

## Script

- `scripts/07_score_phase2_filtered_catalogs.py`

## Salida

- `phase2_official_scores.csv`
- Catálogos de submission y artefactos técnicos del scorer.

La disponibilidad del scorer y sus dependencias se valida antes de ejecutar este paso.
