# 08 — Tracking de experimentos con MLflow

> Estado: cliente `mlflow-skinny` en el `.venv`, cadena comprobada con una prueba de humo y **`train.py` integrado**
> (cada entrenamiento es un run y registra una versión de `fire-detector`), validado con un run de debug.
> Pendiente: entrenar los modelos `_original`, importar los `_clahe` ya entrenados y los alias con `promote.py`.
>
> Infraestructura (contenedores, puertos, arranque): [06_mlops.md](06_mlops.md).

## Por qué

Cada entrenamiento deja una carpeta `results/<exp>/`. Con pocos experimentos basta, pero no responde a:

- ¿con qué config, código y datos exactos se entrenó este modelo?
- ¿cómo comparo las curvas de todos los runs a la vez?
- ¿qué modelo está en producción y cuál había antes?

**MLflow** es un registro de experimentos: un cuaderno de laboratorio. No entrena, no versiona datos y no sirve
modelos. **Apunta lo que pasó** en cada entrenamiento y **guarda lo que produjo**.

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
  así que el cliente no necesita credenciales de S3. El puerto 9000 abierto en el host es para DVC, no para MLflow.

⚠️ Sin `MLFLOW_TRACKING_URI` (o `mlflow.set_tracking_uri`), el cliente **no da error**: guarda en local, en el
directorio actual. Es la causa típica de "he entrenado y no aparece en la UI".

### Jerarquía

```
Experimento  "fire-detection"                 agrupa runs comparables
 ├── Run  a1b2…                               UNA ejecución de train.py
 │    ├── params     lr=0.001, img_size=224…   la config
 │    ├── metrics    val_auc: [0.81, 0.87, 0.90…]  por epoch
 │    ├── tags       mlflow.source.git.commit=…
 │    ├── inputs     dataset (nombre + hash)
 │    └── artifacts  png, csv…
 │         └── modelo (LoggedModel m-…)
 └── Run  c3d4…

Model Registry  "fire-detector"               catálogo aparte
 ├── versión 1  → modelo del run a1b2…
 ├── versión 2  → modelo del run c3d4…   ◄── @production
 └── versión 3  → …                      ◄── @staging
```

| concepto | qué es | ¿mutable? | dónde |
|---|---|---|---|
| **Experimento** | carpeta lógica: nombre + ID numérico (`0` = `Default`) | se renombra o se borra | Postgres |
| **Run** | una ejecución. ID de 32 caracteres hex. Estado `RUNNING` → `FINISHED` / `FAILED` / `KILLED` | — | Postgres |
| **Param** | clave = valor, como texto. Lo que **se decide** antes de entrenar | ❌ registrar otro valor da error | Postgres |
| **Metric** | clave + número + `step` + hora. Lo que **se mide**; guarda toda la historia | se añaden valores | Postgres |
| **Tag** | etiqueta libre; las `mlflow.*` las pone MLflow (usuario, script, commit…) | ✅ | Postgres |
| **Input / dataset** | referencia a los datos: nombre + digest. **No sube los datos** | — | Postgres |
| **Artifact** | cualquier fichero | ❌ | RustFS |
| **LoggedModel** | modelo empaquetado en formato MLflow, ID `m-…` (MLflow 3) | ❌ | metadatos en Postgres, ficheros en RustFS |
| **Registered model / versión** | nombre estable con versiones numeradas, cada una apuntando a un modelo | — | Postgres |
| **Alias** | etiqueta móvil hacia una versión: `@production`, `@staging` | ✅ se mueve | Postgres |

El alias es lo que permite desplegar: la inferencia pide `models:/fire-detector@production`, nunca un número de
versión. Desplegar o hacer rollback es mover el alias, sin tocar código ni copiar ficheros.

### Dónde se guarda cada cosa

**Postgres** (de las 65 tablas, las que se usan):

```
experiments ─┬─ experiment_tags
             └─ runs ─┬─ params          (run_uuid, key, value)
                      ├─ metrics         (run_uuid, key, value, step, timestamp)  historia completa
                      ├─ latest_metrics  último valor de cada métrica, para listar rápido
                      ├─ tags
                      └─ inputs / datasets
registered_models ─┬─ model_versions
                   └─ registered_model_aliases
```

**RustFS, bucket `mlflow`.** A diferencia de DVC, los ficheros **conservan su nombre** y se organizan por IDs:

```
s3://mlflow/
└── <experiment_id>/
    ├── <run_id>/artifacts/           ficheros del run (comprobado en la prueba de humo)
    │   └── training_curves.png
    └── models/<model_id>/artifacts/  modelos (se comprobará al registrar el primero)
        ├── MLmodel                   ficha técnica: flavor, signature, dependencias
        ├── data/                     el modelo
        └── requirements.txt
```

El fichero **`MLmodel`** convierte una carpeta en un modelo MLflow: indica con qué librería se carga (*flavor*:
`keras`, más el genérico `python_function`), qué entrada espera (*signature*) y qué dependencias necesita. Así se
puede cargar con `mlflow.keras.load_model("models:/fire-detector@production")` sin saber cómo se entrenó.

### Cómo viaja cada llamada

```
mlflow.log_param("lr", 0.001)
  └► POST :5000/api/2.0/mlflow/runs/log-parameter ──► servidor ──► INSERT en Postgres

mlflow.log_artifact("training_curves.png")
  └► PUT :5000/api/2.0/mlflow-artifacts/artifacts/<exp>/<run>/artifacts/training_curves.png
        └► servidor (--serve-artifacts) ──► boto3 ──► RustFS
```

### Qué se registra solo y qué no

| | ¿automático? | cómo lo resuelve `train.py` (M.4) |
|---|---|---|
| commit de git | ✅ tag `mlflow.source.git.commit` (vía GitPython) | tag `git_dirty` + `git_diff.patch` si hay cambios sin commitear |
| `.dvc` / hash de los datos | ❌ MLflow no conoce DVC | tag `data.source_md5` + el `.dvc` como artefacto |
| config JSON | ❌ | `mlflow.log_params` con la config aplanada |
| métricas por epoch | solo con `mlflow.keras.autolog()` | callback propio (`autolog` no sirve con dos fases) |

Matices del commit:

- Es el **HEAD** del repo, no el código ejecutado: con cambios sin commitear, el tag apunta a un código distinto del
  que entrenó. MLflow no avisa.
- Se lee del repo **donde está el script**, no del directorio desde el que se lanza. Un script fuera del repo no
  tiene tag de commit (pasó en la prueba de humo).
- Como el `.dvc` está en git, el commit fija también la versión de los datos, **si** el repo está limpio y `data/`
  coincide con el ticket. Solo `original_dataset` está en DVC; los datasets procesados con los que se entrena
  (`*_224`, `*_380`) se deducen de él y de las configs de preproceso del commit (decisión en M.4).

### Borrado en dos fases

1. **Borrado lógico** (UI o API): `lifecycle_stage` pasa a `deleted`. No se borra nada, se puede restaurar y el nombre
   del experimento sigue reservado.
2. **`mlflow gc`**: borra de verdad filas de Postgres y ficheros de RustFS. Solo actúa sobre lo que ya está en
   `deleted`. Irreversible.

Es el equivalente de `dvc gc --cloud`. Un run borrado desde la UI sigue ocupando espacio en el bucket hasta el `gc`.
Nunca se borran tablas ni objetos a mano: Postgres seguiría apuntando a ficheros inexistentes.

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

Unión de las tres herramientas en un run (objetivo de D.6):

```
run a1b2…
├── tag   mlflow.source.git.commit = 9ae3895   ──► git checkout 9ae3895   (código)
├── input dataset digest = fd2b8029…           ──► dvc pull               (datos)
└── model m-… → fire-detector v2 @production
```

### Lo que MLflow no hace

- **No entrena**: lo hace `train.py`; MLflow solo recibe lo que se le envía.
- **No versiona datos**: solo guarda la referencia.
- **No sirve el modelo** en esta arquitectura: lo hará TF Serving. MLflow es el sitio del que se saca el modelo.
- **No decide qué modelo es mejor**: lo hará `promote.py` moviendo el alias.

## Pasos realizados

### M.1 — Cliente en el `.venv`

El servidor ya corre en Docker, así que en el `.venv` solo hace falta el cliente. Comparación con `--dry-run`:

| | `mlflow` | `mlflow-skinny` |
|---|---|---|
| paquetes nuevos | 38 | 19 |
| cliente (log, registro, carga de modelos) | ✅ | ✅ |
| servidor + UI, SQLAlchemy, Flask, pyarrow… | ✅ | ❌ |

Ninguna de las dos cambiaba paquetes existentes. Elegido **`mlflow-skinny`**, la misma versión que el servidor:

```bash
.venv/bin/pip install mlflow-skinny==3.17.2
.venv/bin/pip check          # No broken requirements found
```

Añadido a `requirements.txt` (sección `MLOps`) y lock regenerado (191 paquetes). No incluye `mlflow ui` ni
`mlflow gc`: esos comandos se ejecutan dentro del contenedor.

Al importar aparece `INFO mlflow.agent.hint ... tracing`: publicidad de una función de IA generativa. Se silencia con
`MLFLOW_DISABLE_AGENT_HINT=1`.

### M.2 — Prueba de humo

Script temporal (fuera del repo):

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
exp = mlflow.set_experiment("smoke-test")           # crea el experimento si no existe

with mlflow.start_run(run_name="hola") as run:      # RUNNING → FINISHED al salir (FAILED si hay excepción)
    mlflow.log_param("lr", 0.001)
    for epoch, acc in enumerate([0.70, 0.82, 0.90]):
        mlflow.log_metric("acc", acc, step=epoch)
    mlflow.log_text("hola desde el host", "prueba.txt")
```

| llamada | Postgres | RustFS |
|---|---|---|
| `set_experiment` | `experiments` | — |
| `start_run` | `runs`, `tags` | — |
| `log_param` | `params` | — |
| `log_metric` ×3 | `metrics` ×3, `latest_metrics` | — |
| `log_text` | — | `1/<run_id>/artifacts/prueba.txt` |

Resultado comprobado en los tres sitios:

- **Cliente:** experimento `1`, `artifact_location = mlflow-artifacts:/1`, run `FINISHED`.
- **Postgres** (`psql`, solo lectura): `params` 1 fila; `metrics` 3 filas (0.7, 0.82, 0.9 en los steps 0–2);
  `latest_metrics` 1 fila (0.9); `tags` 4 (`runName`, `source.name`, `source.type`, `user`). Sin tag de commit,
  porque el script estaba fuera del repo. Las horas se guardan en UTC.
- **RustFS:** `s3://mlflow/1/c5798e1de9c34353a1b6abb7b654758a/artifacts/prueba.txt`, 18 bytes, con su nombre original.
- **UI:** el run, el param, la curva de `acc` y el artefacto.

**La UI abre en modo GenAI.** MLflow 3 tiene dos modos: *GenAI* (trazas, tokens; la página
`overview/usage?workflowType=genai` sale vacía) y *Machine learning* (runs, métricas, modelos), que es el que se usa
aquí. Enlaces directos: `http://localhost:5000/#/experiments/<id>/runs` y `…/runs/<run_id>`.

### M.3 — Borrado

**Fase 1:** experimento borrado desde la UI. En Postgres, el experimento y el run pasan a `deleted` y sus filas
siguen ahí; `prueba.txt` sigue en el bucket.

**Fase 2:** `mlflow gc` dentro del contenedor:

```bash
cd deploy/mlops
docker compose exec -T mlflow sh -c '
  URI=$(tr "\0" "\n" < /proc/1/cmdline | sed -n "s/^--backend-store-uri=//p")
  MLFLOW_TRACKING_URI=http://localhost:5000 mlflow gc --backend-store-uri "$URI" --experiment-ids 1
'
```

- `gc` se conecta directamente a Postgres y necesita la URI con la contraseña. No está en las variables del
  contenedor, pero sí en los argumentos del servidor (proceso 1, `/proc/1/cmdline`). Así la contraseña no sale del
  contenedor ni aparece en pantalla, en el historial o en el `ps` del host.
- `MLFLOW_TRACKING_URI` es obligatorio con *artifact proxying*: los runs guardan `mlflow-artifacts:/…`, que solo el
  servidor sabe resolver.
- `--experiment-ids 1` limita el borrado a ese experimento.

Resultado: `Run … has been permanently deleted` y `Experiment with ID 1 has been permanently deleted`. Postgres
queda solo con `Default` y `runs`/`params`/`metrics`/`tags` a 0; el bucket `mlflow`, vacío.

### M.4 — Integración en `train.py`

Todo lo de MLflow está en [`src/train/tracking.py`](../src/train/tracking.py). `train.py` ejecuta el entrenamiento
dentro de `with start_run(...)`: el run acaba `FINISHED`, o `FAILED` si algo falla a mitad. `trainer.py` solo
cambia para aceptar callbacks en las dos fases. `results/<exp>/` se sigue escribiendo igual.

**Diseño:**

| decisión | elegido | por qué |
|---|---|---|
| organización | un experimento `fire-detection`; run con el nombre de la config | los 4 experimentos son comparables (misma tarea y test) |
| params | la config JSON aplanada (`phase1.lr`, `data.image_root`…) + `n_train/val/test` | la config define el experimento |
| métricas por epoch | callback `EpochLogger`, step global 1…N | `autolog` registra el param `epochs` en cada `fit()` y la fase 2 daría error (params inmutables); además cada fase empezaría en 0. Con callback se ven las curvas en directo |
| métricas finales | todo `metrics.json` aplanado (`test.recall`, `test.confusion_matrix.fp`, `test_subtype_alarm_rate.smoke_only`…) | ordenar y comparar runs por cualquiera en la UI |
| tags | `model`, `dataset`, `image_size`, `git_dirty`, `debug`, `data.source`, `data.source_md5` | filtrar en la UI |
| repo sucio | aviso + `git_dirty=true` + `git_diff.patch` | práctico mientras se experimenta; bloquear tendría sentido en CI |
| datos | hash DVC de `original_dataset` (tag) + su `.dvc` como artefacto | ver abajo |
| modelo | `mlflow.keras.log_model` con firma, metadata y versiones de TF/Keras | ver abajo |
| registro | cada run → nueva versión de `fire-detector`, **sin alias**; los runs de debug no se registran | los alias los mueve `promote.py` tras comparar |
| servidor caído | `train.py` para antes de entrenar | no perder una hora de entrenamiento sin registrar |
| dirección | `MLFLOW_TRACKING_URI`, por defecto `http://localhost:5000` | nunca guarda en local sin avisar |

**Datos: solo el origen.** `train.py` lee `data/<dataset>_<size>`, que no está en DVC. Se registra el hash de
`original_dataset` porque el resto se deduce de él y del run: las configs de preproceso están en el commit registrado,
y `dataset`, `image_size` y `data.image_root` son params o tags. Para reproducir los datos de un run:
`git checkout <commit>` → `dvc pull` → scripts de preproceso. El pipeline `dvc.yaml` (D.5) cerraría el hueco de que
alguien regenere los datos derivados sin commitear, y queda como mejora.

**`meta.json`: lo que la inferencia necesita además de los pesos.** Va en el campo `metadata` del `MLmodel` (viaja
con el modelo, también al registrarlo) y como fichero en el run y en `results/<exp>/`. Los parámetros de CLAHE se leen
de `src/preprocess/clahe_config.json`.

```json
{
  "threshold": 0.728, "target_recall": 0.95, "image_size": 224, "dataset": "clahe",
  "preprocessing": {
    "clahe":  {"clip_limit": 2.0, "tile_grid": 8, "channel": "L of CIELAB", "applied": "at full resolution, before resizing"},
    "resize": {"size": 224, "interpolation": "cv2.INTER_AREA", "keep_aspect_ratio": false},
    "input":  "RGB, float32 in 0-255 (the model normalizes internally)"
  },
  "output": "probability of alarm (fire and/or smoke); alarm if prob >= threshold"
}
```

En los datasets `original`, `clahe` es `null`.

**Prueba con un run de debug** (config temporal: MobileNet + CLAHE 224, `debug_max_images: 200`, 1 epoch por fase;
1 min 9 s). El experimento `fire-detection` se creó con ID 2 (el 1 lo usó `smoke-test`; Postgres no reutiliza IDs).

- Tags: los propios más los de MLflow: `mlflow.source.git.commit`, `branch` y `repoURL`. Esta vez sí aparece el
  commit, porque `train.py` está dentro del repo. `git_dirty=True` con su `git_diff.patch`, porque los cambios aún no
  estaban commiteados.
- 26 params, 63 métricas (13 por epoch + 50 finales). `val_pr_auc` con step 1 (fase 1) y 2 (fase 2).
- Modelo no registrado (`debug=True`), como se esperaba.

Estructura real en el bucket:

```
s3://mlflow/2/
├── <run_id>/artifacts/
│   ├── config.json  history.csv  metrics.json  meta.json  git_diff.patch
│   ├── data/original_dataset.dvc
│   ├── plots/          confusion_matrix.png  curves.png  metrics.png
│   └── predictions/    predictions_test.csv  predictions_val.csv
└── models/m-<id>/artifacts/          el modelo es una entidad aparte del run
    ├── MLmodel                       flavors keras + python_function, signature, metadata
    ├── data/model.keras              52 MB
    ├── requirements.txt              tensorflow==2.21.0, keras==3.15.1
    └── python_env.yaml  conda.yaml
```

Signature del `MLmodel`: entrada `image` float32 `[-1, 224, 224, 3]`, salida `alarm` float32 `[-1, 1]`.

**Borrado: el run y el modelo son entidades distintas.** Borrar el run en la UI deja el modelo `active`, ocupando
espacio en el bucket. Hay que borrar los dos y luego hacer `gc` de ambos:

```bash
.venv/bin/python -c "from mlflow import MlflowClient; MlflowClient('http://localhost:5000').delete_logged_model('m-<id>')"
cd deploy/mlops
docker compose exec -T mlflow sh -c '
  URI=$(tr "\0" "\n" < /proc/1/cmdline | sed -n "s/^--backend-store-uri=//p")
  MLFLOW_TRACKING_URI=http://localhost:5000 mlflow gc --backend-store-uri "$URI" \
    --run-ids <run_id> --logged-model-ids m-<id>
'
```

**Pendiente de investigar:** descargar un artefacto desde Python (`mlflow.artifacts.load_text("runs:/…")`) se quedó
colgado más de 2 minutos. Subir funciona y la UI muestra los artefactos, pero cargar el modelo de `@production` desde
`promote.py` o la API depende de la descarga.

**Mejora posible:** `model.keras` incluye el estado del optimizador Adam (~75 % del tamaño, ver
[03_training.md](03_training.md)). No hace falta para inferencia; se podría subir el modelo sin él.

## Referencia

| API / comando | qué hace |
|---|---|
| `mlflow.set_tracking_uri(uri)` / `MLFLOW_TRACKING_URI` | a qué servidor hablar |
| `mlflow.set_experiment(nombre)` | activa el experimento (lo crea si no existe) |
| `mlflow.start_run(run_name=…)` | abre un run; con `with` se cierra solo |
| `mlflow.log_param(s)` | params (inmutables) |
| `mlflow.log_metric(k, v, step=…)` | métrica con historia |
| `mlflow.set_tag(k, v)` | tag |
| `mlflow.log_artifact(ruta)` / `log_text` / `log_dict` | ficheros |
| `mlflow.log_input(dataset)` | referencia a los datos usados |
| `mlflow.get_run(run_id)` | leer un run |
| `mlflow gc …` (en el contenedor) | ⚠️ borrado definitivo de lo que está en `deleted` |

## Siguientes pasos

1. Entrenar `mobilenetv3large_original` y `efficientnetb4_original` con el repo limpio (`git_dirty=False`).
   Con MLflow levantado durante el entrenamiento, para B4 conviene bajar a `--workers=1` por la RAM.
2. Importar `mobilenetv3large_clahe` y `efficientnetb4_clahe` con la misma estructura.
3. Investigar la descarga de artefactos colgada.
4. Alias y `promote.py`.
