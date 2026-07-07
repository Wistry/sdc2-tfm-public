# Clasificador CNN de candidatos

## Objetivo

Construir patches locales del cubo, entrenar una CNN binaria y aplicar el modelo al catálogo conservador.

## Entrada

- Cubo FITS.
- Catálogos de candidatos y etiquetas limpias.
- Configuración de extracción de patches.

## Scripts

1. `scripts/01_build_candidate_patches.py`
2. `scripts/02_train_small_cnn.py`
3. `scripts/03_apply_cnn_to_conservative_catalog.py`
4. `scripts/04_score_cnn_catalogs.py`

## Salida

- Patches y etiquetas en formatos NumPy.
- Metadatos, métricas y predicciones en CSV/JSON.
- Modelo entrenado.
- Catálogos filtrados y scores oficiales en CSV.

Los patches y modelos son artefactos locales no versionados. El entrenamiento excluye candidatos ambiguos.
