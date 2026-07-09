# Definición de la región ampliada

## Objetivo

Definir la región de validación ampliada en 40 GB, comprobar su relación espacial con el subcubo de 10 GB y preparar el truth catalogue correspondiente.

## Entrada

- Cubos FITS de 10 GB y 40 GB.
- Truth catalogue completo.
- Configuraciones SoFiA de referencia y por tiles.

## Scripts

1. `01_check_overlap_10gb_40gb.py`
2. `02_audit_sofia_config_equivalence.py`
3. `03_filter_truth_external_region.py`

## Salida

- Informe técnico de solape en JSON.
- Diferencias de configuración en texto y resumen JSON.
- Truth catalogue de la región ampliada y resumen CSV.

Los ficheros FITS y el truth catalogue completo son datos locales no versionados. Las rutas se obtienen de `validation40gb_utils.py` y de las variables de entorno.
