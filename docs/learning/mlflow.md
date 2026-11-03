# MLflow en detalle

Cómo funcionan el tracking y el Model Registry de MLflow, con ejemplos de este proyecto. Uso en el proyecto: [07_experiment_tracking.md](../07_experiment_tracking.md) y [08_model_registry.md](../08_model_registry.md).

## Por qué

Una carpeta `results/<exp>/` por entrenamiento no responde a:

- ¿con qué config, código y datos exactos se entrenó este modelo?
- ¿cómo se comparan las curvas de todos los runs a la vez?
- ¿qué modelo está en producción y cuál había antes?

**MLflow** es un registro de experimentos, un cuaderno de laboratorio: **apunta lo que pasó** en cada entrenamiento y
**guarda lo que produjo**. No entrena, no versiona datos y no sirve modelos.

## Conceptos

### Cliente y servidor

```
  .venv (host)                          Docker (deploy/mlops)
┌──────────────────┐   HTTP :5000   ┌──────────────────────┐
│ train.py         │ ─────────────► │ MLflow server        │
│  └ mlflow-skinny │                │  (API REST + UI web) │
│    (cliente)     │                └───┬──────────────┬───┘
└──────────────────┘                    ▼              ▼
                                   Postgres         RustFS
                                   texto y números  ficheros
```

- **Cliente:** la librería que usa `train.py`. Cada `mlflow.log_*` es una petición HTTP al servidor.
- **Servidor:** decide dónde guardar cada cosa. **No tiene estado**: todo vive en Postgres y RustFS.
- **El cliente nunca habla con Postgres ni con RustFS.** Con `--serve-artifacts`, el servidor hace de intermediario,
  así que el cliente no necesita credenciales de S3.

⚠️ Sin `MLFLOW_TRACKING_URI` (o `mlflow.set_tracking_uri`), el cliente **no da error**: guarda en local, en el
directorio actual. Es la causa típica de "he entrenado y no aparece en la UI". `tracking.py` siempre fija la URI.

### Jerarquía

```
Experimento  "fire-detection"                       agrupa runs comparables
 ├── Run  a1b2…                                     UNA ejecución de train.py
 │    ├── params     phase1.lr=0.001, data.image_size=380…   la config
 │    ├── metrics    val_auc: [0.81, 0.87, 0.90…]   por epoch, y las finales
 │    ├── tags       mlflow.source.git.commit, data.source_md5…
 │    └── artifacts  png, csv, meta.json…
 │         └── modelo (LoggedModel m-…)
 └── Run  c3d4…

Model Registry  "fire-detector"                     catálogo aparte
 ├── versión 1  → modelo del run …
 ├── versión 2  → modelo del run …
 └── versión 3  → modelo del run …   ◄── @production
```

| concepto | qué es | ¿mutable? | dónde |
|---|---|---|---|
| **Experimento** | carpeta lógica: nombre + ID numérico (`0` = `Default`) | se renombra o se borra | Postgres |
| **Run** | una ejecución. ID de 32 caracteres hex. Estado `RUNNING` → `FINISHED` / `FAILED` / `KILLED` | — | Postgres |
| **Param** | clave = valor, como texto. Lo que **se decide** antes de entrenar | ❌ registrar otro valor da error | Postgres |
| **Metric** | clave + número + `step` + hora. Lo que **se mide**; guarda toda la historia | se añaden valores | Postgres |
| **Tag** | etiqueta libre; las `mlflow.*` las pone MLflow (usuario, script, commit…) | ✅ | Postgres |
| **Artifact** | cualquier fichero | ❌ | RustFS |
| **LoggedModel** | modelo empaquetado en formato MLflow, ID `m-…` (MLflow 3) | ❌ | metadatos en Postgres, ficheros en RustFS |
| **Registered model / versión** | nombre estable con versiones numeradas, cada una apuntando a un modelo | — | Postgres |
| **Alias** | etiqueta móvil hacia una versión: `@production`, `@staging` | ✅ se mueve | Postgres |

El alias es lo que permite desplegar sin números de versión: se pide `models:/fire-detector@production`.

### Dónde se guarda cada cosa

**Postgres** (de las 65 tablas, las que se usan):

```
experiments ─┬─ experiment_tags
             └─ runs ─┬─ params          (run_uuid, key, value)
                      ├─ metrics         (run_uuid, key, value, step, timestamp)  historia completa
                      ├─ latest_metrics  último valor de cada métrica, para listar rápido
                      └─ tags
registered_models ─┬─ model_versions ── model_version_tags
                   └─ registered_model_aliases
```

**RustFS, bucket `mlflow`.** A diferencia de DVC, los ficheros **conservan su nombre** y se organizan por IDs:

```
s3://mlflow/<experiment_id>/
├── <run_id>/artifacts/
│   ├── config.json  history.csv  metrics.json  meta.json  (git_diff.patch si el repo estaba sucio)
│   ├── data/original_dataset.dvc
│   ├── plots/          confusion_matrix.png  curves.png  metrics.png
│   └── predictions/    predictions_test.csv  predictions_val.csv
└── models/m-<id>/artifacts/          el modelo es una entidad aparte del run
    ├── MLmodel                       flavors keras + python_function, signature, metadata
    ├── data/model.keras
    ├── requirements.txt              tensorflow==2.21.0, keras==3.15.1
    └── python_env.yaml  conda.yaml
```

El fichero **`MLmodel`** convierte una carpeta en un modelo MLflow: indica con qué librería se carga (*flavor*:
`keras`, más el genérico `python_function`), qué entrada espera (*signature*) y qué dependencias necesita. Así se
puede cargar con `mlflow.keras.load_model("models:/fire-detector@production")` sin saber cómo se entrenó.

### Cómo viaja cada llamada

```
mlflow.log_param("lr", 0.001)
  └► POST :5000/api/2.0/mlflow/runs/log-parameter ──► servidor ──► INSERT en Postgres

mlflow.log_artifact("curves.png")
  └► PUT :5000/api/2.0/mlflow-artifacts/artifacts/<exp>/<run>/artifacts/curves.png
        └► servidor (--serve-artifacts) ──► boto3 ──► RustFS
```

| llamada | Postgres | RustFS |
|---|---|---|
| `mlflow.set_experiment(nombre)` | `experiments` (lo crea si no existe) | — |
| `mlflow.start_run()` | `runs`, `tags` | — |
| `mlflow.log_param` | `params` | — |
| `mlflow.log_metric(k, v, step)` | `metrics` + `latest_metrics` | — |
| `mlflow.log_artifact` / `log_text` | — | `<exp>/<run>/artifacts/…` |

### Commit de git

MLflow pone solo el tag `mlflow.source.git.commit` (vía GitPython), con estos matices:

- Es el **HEAD** del repo, no el código ejecutado: con cambios sin commitear, el tag apunta a un código distinto del
  que entrenó, y MLflow no avisa. Por eso `tracking.py` añade `git_dirty` y `git_diff.patch`.
- Se lee del repo **donde está el script**, no del directorio desde el que se lanza: un script fuera del repo no
  tiene tag de commit.
- Como el `.dvc` está en git, el commit fija también la versión de los datos, **si** el repo está limpio y `data/`
  coincide con el ticket.

### MLflow frente a DVC

| | DVC | MLflow |
|---|---|---|
| pregunta | ¿con qué datos? | ¿qué pasó al entrenar y qué salió? |
| versiona | entradas (datasets) | ejecuciones y salidas (runs, métricas, modelos) |
| identifica por | hash del contenido (MD5) | IDs (experimento, run, modelo) |
| ficheros en el bucket | por hash, sin nombre, deduplicados | con su nombre, por carpetas de IDs |
| metadatos | ticket `.dvc` en git | tablas en Postgres |
| caché local | sí (`.dvc/cache`) | no |
| acceso al bucket | directo, con credenciales en `config.local` | a través del servidor, sin credenciales |
| bucket | `datasets` | `mlflow` |

Unión de las tres herramientas en un run:

```
run a1b2…
├── tag mlflow.source.git.commit = 3d2770c   ──► git checkout 3d2770c   (código)
├── tag data.source_md5 = fd2b8029….dir      ──► dvc pull               (datos)
└── model m-… → fire-detector v1
```

## Model Registry

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

Un alias por entorno: `@production_mad`, `@production_bcn` (`promote.py --env bcn`). Cada sitio despliega su alias con
el mismo código. Permite desplegar primero en un sitio (*canary*) y hacer rollback por sitio. En este proyecto el
TF Serving local es el entorno por defecto (sin sufijo): con `--env` solo se mueven los alias.

| situación | solución |
|---|---|
| mismo modelo, desplegado en varios sitios a distinto ritmo | un modelo, un alias por sitio |
| modelos distintos por sitio (otros datos o arquitectura) | un modelo registrado por sitio |

## Borrado en dos fases

El borrado es en dos fases:

1. **Borrado lógico** (UI o API): `lifecycle_stage` pasa a `deleted`. No se borra nada, se puede restaurar y el nombre
   del experimento sigue reservado.
2. **`mlflow gc`**: borra de verdad filas de Postgres y ficheros de RustFS. Solo actúa sobre lo que ya está en
   `deleted`. Irreversible.

Nunca se borran tablas ni objetos a mano: Postgres seguiría apuntando a ficheros inexistentes.

## Referencia de la API

| API / comando | qué hace |
|---|---|
| `mlflow.set_tracking_uri(uri)` / `MLFLOW_TRACKING_URI` | a qué servidor hablar |
| `mlflow.set_experiment(nombre)` | activa el experimento (lo crea si no existe) |
| `mlflow.start_run(run_name=…)` | abre un run; con `with` se cierra solo |
| `mlflow.log_param(s)` | params (inmutables) |
| `mlflow.log_metric(k, v, step=…)` | métrica con historia |
| `mlflow.set_tag(k, v)` | tag |
| `mlflow.log_artifact(ruta)` / `log_text` / `log_dict` | ficheros |
| `mlflow.keras.log_model(...)` | modelo con flavor, signature y metadata |
| `mlflow.get_run(run_id)` | leer un run |
| `MlflowClient().set_terminated(run_id, status)` | cerrar un run |
| `mlflow gc …` (en el contenedor) | ⚠️ borrado definitivo de lo que está en `deleted` |

| API | qué hace |
|---|---|
| `client.set_registered_model_alias(name, alias, version)` | poner o mover un alias |
| `client.get_model_version_by_alias(name, alias)` | qué versión tiene un alias (excepción si no existe) |
| `client.delete_registered_model_alias(name, alias)` | quitar un alias |
| `client.get_registered_model(name).aliases` | todos los alias: `{alias: versión}` |
| `client.set_model_version_tag(name, version, key, value)` | tag de versión |
| `mlflow.keras.load_model("models:/fire-detector@production")` | cargar el modelo de producción |
