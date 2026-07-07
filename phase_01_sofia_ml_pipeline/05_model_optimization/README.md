# 05 - Optimización de modelos

## Objetivo

Optimizar los modelos seleccionados, validar su estabilidad y guardar modelos
y thresholds finales.

## Entradas

- Dataset limpio de `baseline_current_full`.
- Features `full` y `no_position` de la subfase 04.
- `config.yaml`.

## Scripts

1. `01_optimize_models.py`: optimización Optuna y barrido de thresholds.
2. `02_validate_optimized_models.py`: validación repetida y permutation test.
3. `03_train_final_models.py`: entrenamiento y persistencia de modelos finales.
4. `04_make_optimization_report.py`: utilidad opcional para figuras.

`utils.py` contiene carga de datos, construcción de modelos, métricas y
persistencia compartida.

## Salidas

- `outputs/optuna/optimization_summary.csv`
- `outputs/validation/validation_summary.csv`
- `outputs/final_models/final_models_manifest.csv`
- modelos y metadata en `outputs/final_models/`
- figuras en `outputs/report_figures/`

## Notas

El entrenamiento usa solo TP/FP limpios. Las políticas disponibles incluyen
`f0_5`, `f1`, `f2`, `balanced_accuracy` y `conservative_fp`.
