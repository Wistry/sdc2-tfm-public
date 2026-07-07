# Aplicación al catálogo conservador

## Objetivo

Aplicar modelos tabulares congelados al catálogo `sdc2_team_sofia_like_full` sin reentrenamiento.

## Entrada

- Dataset extendido del catálogo conservador.
- Modelos y thresholds producidos en `05_focused_winners_comparison`.

## Scripts

- `scripts/06_apply_extended_models_to_conservative_catalog.py`

## Salida

- Catálogos filtrados.
- Probabilidades y decisiones por candidato en CSV.
- Resumen local estructurado en CSV.

Este paso prepara catálogos; no calcula el score oficial.
