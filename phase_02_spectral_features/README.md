# Fase 2: features espectro-espaciales y CNN

## Objetivo

Extender los candidatos SoFiA con 30 descriptores extraídos del cubo, evaluar modelos tabulares y comparar clasificadores CNN sobre patches locales.

## Estructura

| Carpeta | Función |
|---|---|
| `01_extract_spectral_features` | Extrae descriptores locales del cubo. |
| `02_build_extended_datasets` | Une catálogo SoFiA y features extraídas. |
| `03_feature_audit` | Comprueba calidad, variabilidad y separación de features. |
| `04_extended_benchmark` | Evalúa los feature sets extendidos. |
| `05_focused_winners_comparison` | Ajusta y compara los modelos tabulares seleccionados. |
| `06_apply_to_conservative_catalog` | Aplica modelos congelados al catálogo conservador. |
| `07_official_scoring` | Ejecuta el scorer SDC2 sobre catálogos preparados. |
| `08_cnn_candidate_classifier` | Construye patches y entrena la CNN inicial. |
| `09_leakage_audit` | Audita solapamientos y riesgos de leakage. |
| `10_grouped_validation` | Evalúa modelos con particiones agrupadas. |
| `11_robust_grouped_cnn` | Entrena y evalúa la CNN agrupada robusta. |
| `12_final_grouped_evaluation` | Consolida la evaluación agrupada tabular y CNN. |
| `13_cnn_pipeline_comparison` | Compara las variantes CNN con un protocolo común. |
| `configs` | Configuración compartida. |

## Orden de ejecución

Pipeline tabular: `01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07`.

Pipeline CNN: `08 -> 09 -> 10 -> 11 -> 12 -> 13`.

Cada carpeta documenta sus entradas, scripts y artefactos estructurados. Los pasos de scoring requieren el scorer oficial instalado por separado.

## Datos no incluidos

Los cubos FITS de SDC2, patches NumPy, modelos entrenados y otros artefactos voluminosos no se versionan. Las rutas se configuran mediante variables de entorno y los ficheros de `configs/`.

## Salidas

Los scripts generan CSV, JSON, modelos y figuras dentro de `outputs/`. Esas carpetas se mantienen fuera del control de versiones. Los resultados existentes no deben regenerarse sin disponer de los mismos datos y entorno experimental.

## Relación con Fase 3

`phase_03_external_validation_40gb/` reutiliza modelos y configuraciones congelados de esta fase para la validación externa sobre 40 GB.
