# 06 - Comparación de estrategias de catálogo

## Objetivo

Aplicar modelos finales a catálogos completos y calcular métricas locales para
seleccionar estrategias de scoring.

## Entradas

- Catálogos `baseline_current_full` y `sdc2_team_sofia_like_full`.
- Modelos y metadata de `../05_model_optimization/outputs/final_models/`.
- `config.yaml`.

## Scripts

1. `01_apply_final_models_to_catalog.py`: aplica modelos al catálogo permisivo.
2. `02_score_filtered_catalogs.py`: calcula métricas locales en CSV.
3. `03_compare_catalog_strategies.py`: genera rankings estructurados en CSV.
4. `04_make_catalog_strategy_report.py`: utilidad opcional para figuras.
5. `05_select_strategies_for_scoring.py`: exporta selección en CSV/YAML.
6. `06_apply_models_to_sdc2_raw_catalog.py`: aplica modelos al catálogo conservador.

## Salidas

- `outputs/scores/catalog_strategy_scores.csv`
- `outputs/scores/catalog_strategy_rankings.csv`
- `outputs/selected_for_scoring/selected_strategies.csv`
- `outputs/selected_for_scoring/selected_for_08.yaml`
- `outputs/sdc2_postfilter/`
- figuras en `outputs/report_figures/`

## Notas

Esta subfase no entrena. Las métricas son locales y los ambiguos se mantienen
separados de TP y FP limpios.
