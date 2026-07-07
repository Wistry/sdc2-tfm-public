# TFM SDC2: detección y post-filtrado de fuentes H I

Repositorio experimental del Trabajo de Fin de Máster dedicado a la detección y clasificación de candidatos de galaxias H I en cubos espectrales del **SKA Science Data Challenge 2 (SDC2)**.

El trabajo combina:

- detección inicial con SoFiA;
- post-filtrado con modelos tabulares;
- features espectro-espaciales extraídas del cubo;
- clasificación CNN sobre patches locales;
- evaluación con el scorer oficial SDC2;
- validación ampliada sobre una región de 40 GB.

Las métricas calculadas sobre `clean_label` son métricas internas. Solo los resultados obtenidos mediante el scorer SDC2 se consideran scores oficiales.

## Estructura del repositorio

```text
sdc2/
├── phase_01_sofia_ml_pipeline/          # SoFiA y post-filtro tabular en 10 GB
├── phase_02_spectral_features/          # Features espectro-espaciales y CNN
├── phase_03_external_validation_40gb/   # Validación ampliada en 40 GB
├── data/                                # Referencias a datos locales no versionados
├── repos/                               # Dependencias externas como submódulos
├── environment.yml                      # Entorno Conda de referencia
├── .env.example                         # Plantilla de rutas y ejecutables
└── README.md
```

Cada fase dispone de documentación específica:

- [Fase 1: SoFiA y ML tabular](phase_01_sofia_ml_pipeline/README.md)
- [Fase 2: features espectro-espaciales y CNN](phase_02_spectral_features/README.md)
- [Fase 3: validación ampliada en 40 GB](phase_03_external_validation_40gb/README.md)

## Naturaleza del repositorio

Este repositorio debe entenderse como un cuaderno técnico de investigación asociado al TFM, no como una librería de software cerrada ni como un paquete instalable de propósito general.

El objetivo principal es conservar la trazabilidad experimental del trabajo: configuraciones probadas, scripts usados en cada fase, variantes de entrenamiento, criterios de filtrado, procesos de scoring y documentación técnica suficiente para entender cómo se obtuvieron los resultados descritos en la memoria.

Por este motivo, el repositorio puede contener scripts específicos de fase, variantes experimentales y cierta duplicación controlada. Esta organización responde a la necesidad de documentar la evolución del pipeline y no a la intención de presentar una arquitectura software optimizada para producción.

La prioridad del repositorio es la reproducibilidad y la interpretación del proceso experimental, no la abstracción máxima del código.

## Fases del trabajo

### Fase 1: SoFiA y ML tabular

Genera candidatos con SoFiA, construye etiquetas locales mediante matching, realiza el EDA y benchmark tabular, optimiza modelos y evalúa catálogos con el scorer oficial sobre el subcubo de desarrollo de 10 GB.

Se mantienen separados:

- las métricas internas basadas en `clean_label`;
- los scores oficiales de catálogos en formato SDC2.

### Fase 2: features espectro-espaciales y CNN

Añade 30 variables locales calculadas desde el cubo FITS y desarrolla dos ramas:

- pipeline tabular con datasets extendidos;
- clasificación CNN sobre patches de candidatos.

Incluye controles de leakage, validación agrupada, aplicación sobre el catálogo conservador y scoring oficial.

### Fase 3: validación ampliada en 40 GB

Aplica artefactos congelados de las fases anteriores sobre una región de 40 GB, excluyendo el subcubo de 10 GB empleado durante el desarrollo.

Incluye procesamiento por tiles, scoring raw y aplicación de modelos tabulares y CNN congelados. Es una validación ampliada dentro de los datos SDC2, no un conjunto externo completamente independiente.

## Requisitos

### Entorno Conda

El entorno de referencia está definido en [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate sdc2-tfm
```

Incluye Python 3.10, el stack científico, scikit-learn, XGBoost, Optuna y PyTorch. SoFiA y algunos componentes del scorer se proporcionan mediante repositorios externos.

### Submódulos

Clonado inicial:

```bash
git clone --recurse-submodules <URL_DEL_REPOSITORIO>
```

Para inicializarlos en un clon existente:

```bash
git submodule update --init --recursive
```

Los submódulos declarados en [`.gitmodules`](.gitmodules) contienen SoFiA, el scorer SDC2 y material externo utilizado durante el desarrollo. No todos intervienen en cada ejecución.

## Configuración de rutas

Crear la configuración local a partir de la plantilla:

```bash
cp .env.example .env
```

Variables disponibles:

| Variable | Uso |
|---|---|
| `SDC2_DATA_ROOT` | Directorio base de datos SDC2. |
| `SDC2_10GB_CUBE` | Cubo de desarrollo de 10 GB. |
| `SDC2_40GB_CUBE` | Cubo usado en la validación ampliada. |
| `SDC2_TRUTH` | Truth catalogue correspondiente a la ejecución. |
| `SOFIA_BIN` | Ejecutable de SoFiA. |
| `SCORER_ROOT` | Ruta al repositorio del scorer oficial. |

Los scripts no cargan `.env` automáticamente. Para exportar sus variables:

```bash
set -a
source .env
set +a
```

Algunas configuraciones `.par` conservan rutas del entorno experimental original como referencia. Para nuevas ejecuciones deben usarse copias locales adaptadas o los lanzadores que sustituyen las rutas de entrada y salida.

## Datos no incluidos

El repositorio no redistribuye:

- cubos FITS del SDC2;
- truth catalogues completos;
- catálogos SoFiA e intermedios;
- patches CNN;
- modelos entrenados;
- predicciones y submissions completas;
- logs y outputs voluminosos.

Para reproducir el pipeline completo se necesitan los cubos de 10 GB y 40 GB, el truth catalogue correspondiente, SoFiA y el scorer oficial. Los datos deben obtenerse por los canales autorizados del challenge.

## Orden de ejecución

La reproducción debe realizarse de forma incremental:

1. `phase_01_sofia_ml_pipeline/`: detección, etiquetado, benchmark, optimización y scoring.
2. `phase_02_spectral_features/`: extracción de features, datasets extendidos, modelos tabulares y CNN.
3. `phase_03_external_validation_40gb/`: aplicación de artefactos congelados sobre la región ampliada.

Antes de ejecutar una fase:

1. revisar su `README.md`;
2. configurar las rutas locales;
3. comprobar que están disponibles sus entradas;
4. evitar sobrescribir outputs congelados usados para trazabilidad.

## Outputs no versionados

El archivo [`.gitignore`](.gitignore) excluye, entre otros:

- cubos `*.fits` y `*.fit`;
- arrays `*.npy` y `*.npz`;
- modelos `*.pt`, `*.pth`, `*.joblib` y `*.pkl`;
- carpetas `outputs/` y logs;
- archivos `.env` con rutas locales;
- notebooks exploratorios y PDFs finales del TFM.

Los scripts generan artefactos técnicos estructurados como CSV, JSON, YAML, modelos y figuras. La interpretación científica de los resultados se encuentra en la memoria del TFM.

## Alcance y reproducibilidad

Este repositorio documenta un proceso experimental, no una librería instalable de propósito general. Conserva scripts específicos y variantes necesarias para reconstruir la trazabilidad del trabajo.

La reproducción completa no es autocontenida debido al tamaño y las restricciones de redistribución de los datos. Las ejecuciones de SoFiA, Optuna, CNN y scoring pueden requerir tiempos prolongados, memoria suficiente y una configuración local adecuada.

## Licencia

El código se distribuye con fines académicos y de reproducibilidad bajo los términos indicados en [`LICENSE`](LICENSE). Los datos SDC2 están sujetos a sus propias condiciones de acceso y uso.
