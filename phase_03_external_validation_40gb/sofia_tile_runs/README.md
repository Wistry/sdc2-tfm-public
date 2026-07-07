# Ejecuciones SoFiA locales

## Objetivo

Almacenar las configuraciones, runners y artefactos producidos para cada tile de 40 GB.

## Generación

La estructura se crea mediante:

```text
../02_sofia_tile_generation/01_prepare_sofia_40gb_tile_runs.py
```

## Contenido

- Configuraciones `.par` por tile.
- Runners `run_sofia.sh`.
- Catálogos y diagnósticos SoFiA.

El contenido depende del entorno local y está ignorado por Git. Antes de ejecutar un runner se deben revisar las rutas del cubo, el binario SoFiA y el directorio de salida.
