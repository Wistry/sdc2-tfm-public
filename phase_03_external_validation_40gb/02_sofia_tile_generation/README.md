# Generación de ejecuciones SoFiA por tiles

## Objetivo

Preparar las configuraciones y runners necesarios para procesar la región externa mediante tiles espaciales.

## Entrada

- Cubo FITS de 40 GB.
- Plantillas SoFiA de las configuraciones evaluadas.
- Definición de tiles de `phase03_utils.py`.

## Script

- `01_prepare_sofia_40gb_tile_runs.py`

## Runners

- `run_all_sofia_40gb_tiles.sh`
- `run_all_sofia_40gb_tiles_baseline.sh`
- `run_all_sofia_40gb_tiles_sdc2_like.sh`

## Salida

- Configuraciones `.par` por tile.
- Runners `run_sofia.sh`.
- Metadatos de tiles en JSON.
- Catálogos y diagnósticos bajo `../sofia_tile_runs/`.

La preparación no ejecuta SoFiA. Los runners requieren revisar previamente rutas, binario y parámetros locales.
