# Configuraciones SoFiA de referencia

## Tabla comparativa

| Nombre                    | Origen                                                          | scfind.threshold | kernelsXY | kernelsZ       | radio linker XY/Z | minSize linker XY/Z | reliability                                    | Interpretación                                                       |
| ------------------------- | --------------------------------------------------------------- | ---------------: | --------- | -------------- | ----------------- | ------------------- | ---------------------------------------------- | -------------------------------------------------------------------- |
| `baseline_current`        | `configs/sofia/test_dev_medium.par`                             |            `3.5` | `0,3,6`   | `0,3,7`        | `2/2`             | `3/3`               | desactivada                                    | Baseline local actual                                                |
| `sofia2_default_template` | `repos/SoFiA-2-src/template_par_file.par`                       |            `5.0` | `0,3,6`   | `0,3,7,15`     | `1/1`             | `5/5`               | desactivada, threshold `0.9`                   | Conservadora                                                         |
| `sdc2_team_sofia_like`    | `repos/SoFiA-2/sofia/sofia_001.par` ... `sofia_080.par`         |            `3.8` | `0,3,6`   | `0,3,7,15,31`  | `2/2`             | `3/3`               | activada, threshold `0.1`, minSNR `1.5`        | Referencia SDC2 equilibrada                                          |
| `hi_friends_dev12_like`   | `repos/hi-firends_analysis/config/dev12.par`                    |            `3.5` | `0,4,8`   | `0,5,11,21,41` | `4/5`             | `5/3`               | activada, threshold `0.4`, filter.minSNR `6.0` | Permisiva / adecuada para fuentes extendidas, con corte SNR estricto |
| `hi_friends_yaml_like`    | `repos/hi-firends_analysis/config/sofia_12.par` + `config.yaml` |            `4.0` | `0,4,8`   | `0,5,11,21,41` | `4/5`             | `5/3`               | activada, threshold `0.4`, filter.minSNR `6.0` | Plantilla tipo HI-FRIENDS, algo más estricta                         |
| `loose_recall`            | Variante exploratoria local                                     |            `2.8` | `0,3,6`   | `0,3,7,15,31`  | `3/3`             | `2/2`               | desactivada                                    | Orientada a recall                                                   |
| `strict_reliability`      | Variante exploratoria local                                     |            `4.5` | `0,3,6`   | `0,3,7,15`     | `1/1`             | `5/5`               | activada, threshold `0.6`, minSNR `2.0`        | Orientada a reliability                                              |

## Interpretación

* **Conservadora:** genera menos candidatos, reduce la presión de falsos positivos, pero aumenta el riesgo de perder fuentes reales.
* **Equilibrada:** punto de partida razonable para generación de candidatos en un contexto similar a SDC2.
* **Permisiva:** usa suavizados y enlazados más amplios; útil para fuentes extendidas o fragmentadas.
* **Orientada a recall:** acepta intencionadamente más candidatos para reducir el riesgo de perder verdaderos positivos.
* **Orientada a reliability:** aplica filtrado más estricto antes de la etapa de clasificación.

## Nota de compatibilidad con SoFiA 2.7

`reliability.fmin = 6.0` aparece en configuraciones históricas de HI-FRIENDS, pero SoFiA-2 v2.7 no reconoce ese parámetro. En esta demo se mantiene el equivalente moderno `filter.minSNR` cuando se necesita una explicación de compatibilidad.
