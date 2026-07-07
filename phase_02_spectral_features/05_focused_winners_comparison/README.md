# Comparación focalizada de modelos tabulares

## Objetivo

Entrenar los modelos seleccionados en Fase 1 con los feature sets de Fase 2 y fijar sus thresholds de operación.

## Entrada

- Dataset extendido limpio del catálogo permisivo.
- Configuraciones y modelos candidatos de Fase 1.

## Scripts

- `scripts/05_train_phase1_winners_with_extended_features.py`

## Salida

- Resultados y thresholds en CSV.
- Modelos serializados.
- Comparación local de estrategias en CSV.
- Selección de estrategias en JSON.

Los modelos se comparan con el mismo esquema de partición. La selección oficial se realiza posteriormente con el scorer SDC2.
