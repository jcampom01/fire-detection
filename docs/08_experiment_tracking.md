# 08 — Tracking de experimentos con MLflow

> Estado: cliente `mlflow-skinny` instalado en el `.venv` y cadena completa comprobada con una prueba de humo
> (escritura, lectura y borrado definitivo). Pendiente: diseñar qué registra `train.py`, integrarlo, importar los
> modelos ya entrenados y el registro de modelos con alias.
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

| | ¿automático? | qué hará falta en `train.py` |
|---|---|---|
| commit de git | ✅ tag `mlflow.source.git.commit` (vía GitPython) | comprobar que el repo no tiene cambios sin commitear |
| `.dvc` / hash de los datos | ❌ MLflow no conoce DVC | `mlflow.log_input` o artefacto + comprobar `dvc status` |
| config JSON | ❌ | `mlflow.log_params(config)` |
| métricas por epoch | solo con `mlflow.keras.autolog()` | autolog o callback |

Matices del commit:

- Es el **HEAD** del repo, no el código ejecutado: con cambios sin commitear, el tag apunta a un código distinto del
  que entrenó. MLflow no avisa.
- Se lee del repo **donde está el script**, no del directorio desde el que se lanza. Un script fuera del repo no
  tiene tag de commit (pasó en la prueba de humo).
- Como el `.dvc` está en git, el commit fija también la versión de los datos, **si** el repo está limpio y `data/`
  coincide con el ticket. Ahora solo `original_dataset` está en DVC; los datasets procesados con los que se entrena
  (`*_224`, `*_380`) necesitan el pipeline D.5 de [07_data_versioning.md](07_data_versioning.md).

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

1. Diseñar qué registra `train.py`: nombres de experimento y modelo, params, métricas, artefactos, tags y
   comprobaciones (repo limpio, `dvc status`).
2. Integrarlo en `train.py` antes de entrenar `mobilenetv3large_original` y `efficientnetb4_original`.
3. Importar `mobilenetv3large_clahe` y `efficientnetb4_clahe` con la misma estructura.
4. Registro de modelos, alias y `promote.py`.
