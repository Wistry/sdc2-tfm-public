# Fase 1 - Pipeline SoFiA y ML tabular

## Objetivo

Generar candidatos HI con SoFiA, construir etiquetas locales mediante matching,
entrenar un post-filtro tabular y evaluar los catálogos seleccionados con el
scorer oficial SDC2.

## Estructura

| Carpeta | Función |
| --- | --- |
| `01_fits_cube_understanding/` | Comprobación de ejes, WCS y coordenadas del cubo. |
| `02_sofia_config_search/` | Preparación, ejecución y revisión inicial de configuraciones SoFiA. |
| `03_candidate_dataset/` | Construcción y etiquetado local de catálogos completos. |
| `04_eda_and_benchmark/` | EDA y benchmark interno con variantes `full` y `no_position`. |
| `05_model_optimization/` | Optimización con Optuna, validación, thresholds y modelos finales. |
| `06_catalog_strategy_comparison/` | Aplicación de modelos y comparación de estrategias locales. |
| `07_official_sdc2_scoring/` | Conversión a formato SDC2 y scoring oficial. |

## Datos y dependencias

Los cubos FITS y otros datos voluminosos no se versionan. SoFiA se necesita en
las subfases 02 y 03. El paquete oficial de scoring SDC2 se necesita en la
subfase 07. Las rutas locales se configuran mediante variables de entorno o
copias de trabajo de las configuraciones `.par`.

## Artefactos

Los scripts generan CSV, JSON, YAML, modelos y figuras. Los directorios
`outputs/` están ignorados y conservan los resultados científicos locales. La
interpretación de resultados se mantiene en la memoria del TFM.

Las métricas basadas en `clean_label` son locales. Solo la subfase 07 produce
score oficial SDC2. 