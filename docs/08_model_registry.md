# 08 — Registro de modelos y promoción

`promote.py` decide qué versión de `fire-detector` está en producción, con reglas versionadas (el gate) y dejando
constancia en el registro de MLflow. `deploy` y `rollback` además la sirven en TF Serving y la verifican
([09_serving.md](09_serving.md)).

Cómo llegan las versiones al registro: [07_experiment_tracking.md](07_experiment_tracking.md). Alias, tags y su
almacenamiento en MLflow, en detalle: [learning/mlflow.md](learning/mlflow.md#model-registry).

- Un **alias** (`@production`) apunta a **una** versión y se mueve: es lo que usa el resto del sistema.
- Un **tag de versión** (`gate.result=passed`) es una nota que se queda: es el historial, porque MLflow no guarda
  el de los alias.

## Por qué

Cada entrenamiento registra una versión nueva (v1, v2…), pero el resto del sistema no debe saber números de versión.
La API pide **"la de producción"** y alguien decide cuál es, con reglas y dejando constancia. Desplegar o volver atrás
es mover un puntero, sin tocar código ni copiar ficheros.

## Flujo

```
train.py / import_results.py ──► nueva versión (sin alias)
                                     │
                          promote.py stage: ¿pasa el gate?
                                     │ sí
                                     ▼
                                 @staging      candidata aprobada, pendiente de desplegar
                                     │
                          promote.py deploy (vuelve a pasar el gate)
                                     ▼
                                 @production   la que sirve TF Serving ── la saliente pasa a @previous
                                     │             (exportar → activar → esperar → verificar)
                                     │
                          promote.py rollback: @production vuelve a @previous
```

## El gate

Reglas en [`src/registry/promotion.json`](../src/registry/promotion.json), versionadas en git:

```json
{
    "registered_model": "fire-detector",
    "compare":  {"metric": "val.false_positive_rate", "lower_is_better": true, "min_improvement": 0.001},
    "minimums": {"test.recall": 0.92},
    "maximums": {"test.false_positive_rate": 0.03}
}
```

**Se compara con val, no con test.** `train.py` elige el umbral en val para que el recall sea ≥ 95 %, así que en val
todos los modelos tienen recall ~0.95 y lo que los distingue son las **falsas alarmas** con ese recall. Elegir por test
ajustaría la elección al conjunto de test: con test recall ganaría la v2, que en val es la peor (casi el doble de falsas
alarmas que la v3). Test se usa solo como suelo de seguridad.

| v | run | val.recall | val.FPR | test.recall | test.FPR | img/s (GPU) |
|---|---|---:|---:|---:|---:|---:|
| 1 | mobilenetv3large_original | 0.951 | 1,67 % | 0.926 | 1,60 % | 406 |
| 2 | mobilenetv3large_clahe | 0.951 | 2,84 % | 0.949 | 2,00 % | 374 |
| 3 | efficientnetb4_clahe | 0.950 | **1,24 %** | 0.932 | **0,85 %** | 25 |

Comprobaciones:

| comprobación | por qué |
|---|---|
| run `FINISHED` y no debug | solo entrenamientos completos |
| no está ya en `@production` | no tiene sentido promoverla |
| `minimums` / `maximums` | requisitos absolutos; si falta la métrica, falla |
| `compare` (solo si hay `@production`) | mejora = signo × (candidato − producción), con signo −1 si menor es mejor; debe ser ≥ `min_improvement` (evita cambiar de modelo por ruido) |

La **velocidad** no está en el gate: la que guarda `train.py` se mide en la GPU de entrenamiento, no donde se sirve.
Servido en CPU, B4 (v3) tarda ~290–320 ms por imagen y MobileNet (v1) ~72 ms ([04](04_results.md#coste)).

## `promote.py`

En `src/registry/`, separado de `src/train/`: no entrena ni importa TensorFlow; lee métricas y mueve alias (~2 s).
Para servir una versión lanza [`export.py`](../src/registry/export.py) como proceso aparte (ver
[09_serving.md](09_serving.md#despliegue)).

| comando | qué hace |
|---|---|
| `status` | versiones, métrica de comparación, `test.recall`, resultado del gate y alias; además `@production = v3   served = v3   OK` o `DRIFT`, preguntando a la API qué versión sirve |
| `stage [--version N]` | gate sobre la versión (por defecto, la última). Graba el resultado en tags **también si falla** y, si pasa, mueve `@staging`. Si falla, sale con código 1 (útil en CI) |
| `deploy` | exige `@staging`; **repite el gate** contra el `@production` actual (puede haber cambiado); exporta la versión (antes de tocar alias); la saliente pasa a `@previous`; mueve `@production`; quita `@staging`; **activa la versión en TF Serving**; espera a que se sirva y la **verifica** |
| `rollback` | exige `@previous`; `@production` vuelve a esa versión y **TF Serving la sirve** (ya exportada: solo cambia la config); quita `@previous` (no hay "anterior del anterior"); espera y verifica |

Opciones:

| opción | qué hace |
|---|---|
| `--env bcn` | alias `*_bcn` y tags con sufijo `_bcn`, para no mezclar historiales |
| `--dry-run` | imprime `[dry-run] would …` sin cambiar nada; todas las escrituras pasan por un único método |
| `--rules <fichero>` | otras reglas del gate; [`promotion_drill.json`](../src/registry/promotion_drill.json) no exige mejora, para ensayar `deploy`/`rollback` con una versión peor. El fichero usado queda en `gate.rules` |
| `--verify N` | imágenes del test que se comprueban a través de la API tras `deploy`/`rollback` (por defecto 200, `0` lo salta); si falla, sale con código 1 y aconseja `rollback` ([09](09_serving.md#verificación-tras-el-despliegue)) |

Tags que deja:

| cuándo | versión | tags |
|---|---|---|
| `stage` / gate | candidata | `gate.result`, `gate.reason`, `gate.compared_to`, `gate.rules`, `gate.checked_at` |
| `deploy` | entrante | `promoted_at`, `replaced` |
| `deploy` | saliente | `retired_at`, `replaced_by` (+ alias `@previous`) |
| `rollback` | la que sale / la que vuelve | `rolled_back_at` / `restored_at` |

Las versiones de `search_model_versions` no traen sus alias; `status` los lee del modelo registrado
(`get_registered_model(...).aliases`, `{alias: versión}`).

## Uso

```bash
P=".venv/bin/python src/registry/promote.py"
$P status                  # tabla de versiones y versión servida
$P stage --version 4       # gate → @staging (sale con código 1 si no pasa)
$P deploy                  # @staging → @production, exporta, sirve y verifica
$P rollback                # @production vuelve a @previous, sirve y verifica
$P deploy --dry-run        # qué haría, sin cambiar nada
```

Salida de `status`:

```
fire-detector  (aliases: @production @staging @previous)

v   run                            val.false_positive_rate  test.recall  gate    aliases
1   mobilenetv3large_original                       0.0167       0.9261  passed
2   mobilenetv3large_clahe                          0.0284       0.9487  -
3   efficientnetb4_clahe                            0.0124       0.9318  passed  @production

@production = v3   served = v3   OK
```

`DRIFT` en la última línea indica que TF Serving no sirve lo que dice `@production` (por ejemplo, si se movió el alias
desde la UI): se arregla con `deploy`, `rollback` o `export.py --version <v>`.

Con las reglas reales, ni la v1 ni la v2 pueden sustituir a la v3: tienen más falsas alarmas en val.

### Simulacro de despliegue

Para ensayar `deploy` y `rollback` sin una versión mejor que la de producción,
[`promotion_drill.json`](../src/registry/promotion_drill.json) mantiene los suelos de test pero no exige mejora
(`min_improvement: -1`):

```bash
$P stage --version 1 --rules src/registry/promotion_drill.json
$P deploy --rules src/registry/promotion_drill.json    # @production = v1, @previous = v3
$P rollback                                            # @production = v3
```

La versión queda con `gate.rules=promotion_drill.json`, así que el historial distingue un simulacro de una promoción
real. Las etiquetas de despliegue (`promoted_at`, `rolled_back_at`…) se quedan en las versiones.

## Limitaciones conocidas

- **Sin regla de latencia en el gate.** Haría falta medir la velocidad donde se sirve (CPU) y guardarla en el run; con
  ella, el gate podría exigir, por ejemplo, ≤ 500 ms por imagen.
- **El gate usa métricas registradas, no vuelve a evaluar.** Confía en las del run; la comprobación independiente es
  la verificación tras el despliegue.
