# Construcción de datasets extendidos

## Objetivo

Unir los catálogos SoFiA con las 30 features espectro-espaciales conservando la trazabilidad por candidato.

## Entrada

- Catálogos SoFiA etiquetados.
- CSV generados por `01_extract_spectral_features`.

## Script

- `scripts/02_build_extended_dataset.py`

## Salida

- `baseline_current_full_extended_raw.csv`
- `baseline_current_full_extended_clean.csv`
- `sdc2_team_sofia_like_full_extended_raw.csv`
- `sdc2_team_sofia_like_full_extended_clean.csv`

Las versiones `clean` exigen extracción correcta y ausencia de valores perdidos en las features añadidas.
