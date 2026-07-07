# 02 - Búsqueda de configuraciones SoFiA

## Objetivo

Preparar configuraciones SoFiA permisivas y conservadoras y compararlas con
métricas locales sobre una región reducida.

## Entradas

- Cubo FITS de desarrollo.
- Truth catalogue proyectado a píxeles.
- Plantillas `.par` de `configs/`.

## Scripts

1. `01_prepare_truth_pixels.py`: proyecta el truth a píxeles.
2. `02_generate_sofia_configs.py`: genera variantes `.par`.
3. `03_prepare_sofia_runs.py`: prepara comandos de ejecución.
4. `run_sofia_all.sh`: ejecuta las configuraciones activas.
5. `04_score_sofia_configs.py`: calcula métricas locales sobre catálogos existentes.

## Salidas

Los catálogos, métricas y ficheros de ejecución se guardan en `outputs/`. Esta
subfase no produce informes Markdown desde sus scripts.

## Notas

SoFiA solo debe ejecutarse con rutas de entrada y salida verificadas. Las
métricas obtenidas aquí son locales y no equivalen al score oficial SDC2.
