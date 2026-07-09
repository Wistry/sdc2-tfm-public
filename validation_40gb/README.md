# Validación ampliada en 40 GB

## Objetivo

Aplicar sobre una región ampliada del cubo de 40 GB las configuraciones SoFiA, modelos y thresholds obtenidos en las fases anteriores. Los artefactos se utilizan congelados: no se reentrenan modelos ni se recalibran thresholds con los datos de esta validación.

## Estructura

| Carpeta | Función |
|---|---|
| `01_region_definition` | Comprueba el solape WCS, audita configuraciones y genera el truth de la región ampliada. |
| `02_sofia_tile_generation` | Prepara configuraciones y runners SoFiA por tiles. |
| `03_raw_catalog_scoring` | Inspecciona, fusiona y evalúa los catálogos SoFiA raw. |
| `04_tabular_models_40gb` | Extrae features y aplica modelos tabulares congelados. |
| `05_cnn_models_40gb` | Extrae patches y aplica modelos CNN congelados. |
| `outputs` | Contiene artefactos locales no versionados. |
| `sofia_tile_runs` | Contiene configuraciones y resultados locales por tile. |
| `validation40gb_utils.py` | Define rutas, regiones y utilidades compartidas. |

## Orden de ejecución

1. Ejecutar los controles y el filtrado de `01_region_definition/`.
2. Preparar los tiles con `02_sofia_tile_generation/`.
3. Ejecutar SoFiA mediante los runners generados.
4. Fusionar y puntuar los catálogos raw con `03_raw_catalog_scoring/`.
5. Aplicar y puntuar modelos tabulares con `04_tabular_models_40gb/`.
6. Extraer patches, aplicar CNN y comparar resultados con `05_cnn_models_40gb/`.

Cada bloque incluye un README con sus scripts, entradas y salidas.

## Dependencias

- Cubo SDC2 de 40 GB y truth catalogue correspondiente.
- Ejecutable SoFiA.
- Scorer oficial SDC2.
- Modelos y thresholds congelados de Fase 2.
- Entorno definido en `environment.yml`.
- Variables de entorno documentadas en `.env.example`.

## Datos no incluidos

No se versionan cubos FITS, truth catalogues completos, catálogos generados, patches CNN, predicciones, submissions, modelos, logs ni resultados voluminosos.

## Salidas

Los scripts generan CSV, JSON, YAML, catálogos, submissions, scores y arrays NumPy. `outputs/` está ignorado por Git y conserva los artefactos locales de ejecución.

La región de 10 GB usada durante el desarrollo se excluye espacialmente. Esta validación ampliada se realiza dentro de los datos SDC2, no sobre un conjunto completamente independiente.
