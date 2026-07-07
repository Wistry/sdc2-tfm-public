# Extracción de features espectro-espaciales

## Objetivo

Extraer 30 descriptores locales para cada candidato SoFiA a partir del cubo FITS.

## Entrada

- Catálogo de candidatos con coordenadas `x`, `y`, `z`.
- Cubo FITS configurado en el entorno.
- Parámetros de ventana y regiones fuente/fondo.

## Script

- `scripts/01_extract_candidate_z_features.py`

## Salida

- CSV con las 30 features espectro-espaciales por candidato.
- Indicadores técnicos `feature_extraction_ok` y `edge_clipped`.

## Notas de reproducibilidad

La configuración principal usa una ventana `17 x 17 x 41`, región fuente `r <= 4 px` y anillo de fondo `6 <= r <= 8 px`. `edge_clipped` registra ventanas recortadas por los límites del cubo; no implica por sí solo un fallo de extracción.
