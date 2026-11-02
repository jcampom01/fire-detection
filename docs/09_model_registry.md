# 09 — Registro de modelos y promoción

> Estado: `fire-detector` con 3 versiones y **v3 (`efficientnetb4_clahe`) en `@production`**, promovida con
> [`src/registry/promote.py`](../src/registry/promote.py). Pendiente: la API de inferencia, que cargará
> `models:/fire-detector@production`.
>
> Cómo llegan las versiones al registro: [08_experiment_tracking.md](08_experiment_tracking.md).

## Por qué

Cada entrenamiento registra una versión nueva (v1, v2…), pero el resto del sistema no debe saber números de versión.
La API pide **"la de producción"** y alguien decide cuál es, con reglas y dejando constancia. Desplegar o volver atrás
es mover un puntero, sin tocar código ni copiar ficheros.

## Conceptos

### Alias, tags y descripción

| | alias | tag de versión | descripción |
|---|---|---|---|
| qué representa | un **puesto**: "la que está en producción" | una **nota**: "pasó el gate el día X" | texto libre |
| ¿cuántas versiones a la vez? | **una**: al ponerlo en otra, se quita de la anterior | todas las que se quiera | — |
| forma | nombre | clave = valor | texto |
| ¿sirve para cargar el modelo? | ✅ `models:/fire-detector@production` | ❌ solo para buscar (`tags.gate.result='passed'`) | ❌ |
| ¿cambia? | se **mueve** | se pone y se deja, como registro | — |

Regla práctica: si el resto del sistema necesita **encontrar** un modelo por su papel, alias; si es información
**sobre** un modelo, tag.

- El nombre del alias es libre, salvo `latest` y nombres tipo `v1`, `v2` (reservados para versiones).
- Los antiguos **stages** de MLflow (`Staging`, `Production`, `Archived`) están obsoletos: los sustituyen alias y tags.

### Dónde se guardan

Postgres, tabla `registered_model_aliases`, una fila por alias (`name`, `alias`, `version`). Mover un alias es un
`UPDATE`: instantáneo, sin copiar ficheros. Los tags van en `model_version_tags`.

**MLflow no guarda el historial de alias:** al mover `@production`, no queda registro de la versión anterior. Por eso
`promote.py` deja tags en las versiones y usa el alias `@previous`.

### Qué pasa con la versión que sale de producción

Nada, salvo que pierde el alias. Sigue registrada, con su run, métricas y modelo, y se puede cargar por número
(`models:/fire-detector/2`). Solo desaparece del registro con `delete_model_version`, y ni eso borra el run ni el modelo
del bucket.

### Varios entornos

Un alias por entorno: `@production_mad`, `@production_bcn`. Cada API usa la misma imagen y solo cambia la variable
`MODEL_URI`. Permite desplegar primero en un sitio (*canary*) y hacer rollback por sitio.

| situación | solución |
|---|---|
| mismo modelo, desplegado en varios sitios a distinto ritmo | un modelo, un alias por sitio |
| modelos distintos por sitio (otros datos o arquitectura) | un modelo registrado por sitio |

## Flujo

```
train.py / import_results.py ──► nueva versión (sin alias)
                                     │
                          promote.py stage: ¿pasa el gate?
                                     │ sí
                                     ▼
                                 @staging      candidata (se probará con la API en un entorno de pruebas)
                                     │
                          promote.py deploy (vuelve a pasar el gate)
                                     ▼
                                 @production   la que sirve la API ──── la saliente pasa a @previous
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

La **velocidad** no está en el gate: la que guarda `train.py` se mide en la GPU de entrenamiento, no donde correrá la
API. Se decidirá al medir la API (B4 es ~15 veces más lenta que MobileNet).

## `promote.py`

En `src/registry/`, separado de `src/train/`: no entrena ni importa TensorFlow; lee métricas y mueve alias (~2 s).
Para servir una versión lanza [`export.py`](../src/registry/export.py) como proceso aparte (ver
[10_serving.md](10_serving.md#s3--despliegue-automático)).

| comando | qué hace |
|---|---|
| `status` | versiones, métrica de comparación, `test.recall`, resultado del gate y alias; además `@production = v3   served = v3   OK` o `DRIFT`, preguntando a la API qué versión sirve |
| `stage [--version N]` | gate sobre la versión (por defecto, la última). Graba el resultado en tags **también si falla** y, si pasa, mueve `@staging`. Si falla, sale con código 1 (útil en CI) |
| `deploy` | exige `@staging`; **repite el gate** contra el `@production` actual (puede haber cambiado); exporta la versión (antes de tocar alias); la saliente pasa a `@previous`; mueve `@production`; quita `@staging`; **activa la versión en TF Serving** |
| `rollback` | exige `@previous`; `@production` vuelve a esa versión y **TF Serving la sirve** (ya exportada: solo cambia la config); quita `@previous` (no hay "anterior del anterior") |

Opciones: `--env bcn` (alias `*_bcn` y tags con sufijo `_bcn`, para no mezclar historiales) y `--dry-run` (imprime
`[dry-run] would …` sin cambiar nada; todas las escrituras pasan por un único método).

Tags que deja:

| cuándo | versión | tags |
|---|---|---|
| `stage` / gate | candidata | `gate.result`, `gate.reason`, `gate.compared_to`, `gate.checked_at` |
| `deploy` | entrante | `promoted_at`, `replaced` |
| `deploy` | saliente | `retired_at`, `replaced_by` (+ alias `@previous`) |
| `rollback` | la que sale / la que vuelve | `rolled_back_at` / `restored_at` |

Las versiones de `search_model_versions` no traen sus alias; `status` los lee del modelo registrado
(`get_registered_model(...).aliases`, `{alias: versión}`).

## Primera promoción

```bash
.venv/bin/python src/registry/promote.py stage --version 3   # sin @production: solo requisitos absolutos
.venv/bin/python src/registry/promote.py deploy
.venv/bin/python src/registry/promote.py status
```

```
v   run                        val.false_positive_rate  test.recall  gate    aliases
1   mobilenetv3large_original                   0.0167       0.9261  -
2   mobilenetv3large_clahe                      0.0284       0.9487  -
3   efficientnetb4_clahe                        0.0124       0.9318  passed  @production
```

Tags de la v3: `gate.result=passed`, `gate.compared_to=none`, `promoted_at=2026-11-01 09:49:25 UTC`, `replaced=none`.
Con estas reglas, ni la v1 ni la v2 pueden sustituirla: tienen más falsas alarmas en val.

## Referencia

| API | qué hace |
|---|---|
| `client.set_registered_model_alias(name, alias, version)` | poner o mover un alias |
| `client.get_model_version_by_alias(name, alias)` | qué versión tiene un alias (excepción si no existe) |
| `client.delete_registered_model_alias(name, alias)` | quitar un alias |
| `client.get_registered_model(name).aliases` | todos los alias: `{alias: versión}` |
| `client.set_model_version_tag(name, version, key, value)` | tag de versión |
| `mlflow.keras.load_model("models:/fire-detector@production")` | cargar el modelo de producción |

## Siguientes pasos

- ~~API de inferencia que sirva `@production`~~ → hecho: [10_serving.md](10_serving.md); `deploy` y `rollback`
  llegan solos a TF Serving.
- Un ciclo real de `deploy` / `rollback` con `promote.py` (S.5 de [10](10_serving.md)).
- Regla de latencia en el gate: en CPU, B4 (v3) ~290 ms por imagen y MobileNet (v1) ~72 ms.
