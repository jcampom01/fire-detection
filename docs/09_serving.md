# 09 — Servir el modelo

El modelo en `@production` se sirve con **TF Serving** detrás de una **API FastAPI**, que también sirve una página web
para probarlo. Los dos van en [`deploy/serving/docker-compose.yml`](../deploy/serving/docker-compose.yml) (proyecto
`fire-serving`). `promote.py deploy/rollback` exporta la versión, la activa y la verifica contra el entrenamiento.

Conceptos (TF Serving, SavedModel, cambio de versión sin cortes, alternativas y patrones de la industria):
[learning/serving.md](learning/serving.md).

## Arquitectura

La inferencia va **separada del modelo**: la API no carga el modelo, se lo pide a un servidor de modelos.

```
                       ┌──────── stack MLOps (deploy/mlops) ──────┐
                       │  MLflow ── Postgres     RustFS           │
                       └────▲─────────────────────────────────────┘
                            │ ¿@production? · descarga
 promote.py deploy/rollback ─► export.py (host): .keras → SavedModel (bytes JPEG) + warmup
                            │                     + models.config ("sirve la versión <v>")
                            ▼
                    ./models/fire_detector/<versión>/  (+ meta.json)     ./models/models.config
                       │                     │                                 │
                 lee meta.json         carga la versión fijada ◄───────────────┘
                       ▼                     ▼
cliente ─POST /predict─► [ API :8080 ] ──REST (red interna)──► [ TF Serving ]
web       foto original   preparación: CLAHE → JPEG → resize → JPEG    decodifica el JPEG
                          umbral                                         y predice
       ◄─ {prob, alarm, threshold, model_version, latency_ms}
```

| pieza | hace | no hace |
|---|---|---|
| **API** ([`src/api/app.py`](../src/api/app.py)) | validar, preparar la imagen como el dataset, llamar al modelo, aplicar el **umbral**, responder; servir la web | no lleva TensorFlow |
| **TF Serving** | decodificar el JPEG y predecir; cargar la versión fijada en la config; cambiar sin cortes | no preprocesa, no aplica el umbral, no sabe qué es MLflow |
| **[`export.py`](../src/registry/export.py)** | descargar la versión, convertir a SavedModel con *warmup*, escribir la carpeta, fijar la versión | no sirve peticiones |
| **[`verify_serving.py`](../src/registry/verify_serving.py)** | mandar fotos del test a la API y compararlas con las predicciones del run en MLflow | no cambia nada: aprueba o suspende |

La API no lleva TensorFlow y cambiar de modelo no la reinicia: TF Serving carga la versión nueva en segundo plano.

## Modelo servido

Cada versión se exporta a `deploy/serving/models/fire_detector/<v>/` como **SavedModel** (el formato de TF Serving),
con *warmup* y su `meta.json`:

```
inputs['image_bytes']  string   (-1)          un JPEG por imagen
outputs['alarm']       float32  (-1, 1)
grafo: decode_image → resize(antialias) → clip(0, 255) → modelo
```

| dentro del modelo | fuera (en la API, según `meta.json`) |
|---|---|
| decodificar el JPEG, normalización, pesos | CLAHE, resize, JPEG intermedios, **umbral** |

La entrada son bytes JPEG, no píxeles, para que el modelo decodifique igual que en el entrenamiento (ver
[Preparación fiel al entrenamiento](#preparación-fiel-al-entrenamiento)).

**Versión fijada:** `models.config` dice qué versión servir (no "la más alta", para que el rollback a una versión
menor funcione). TF Serving lo relee cada 5 s y cambia sin cortes. La API pide siempre una versión explícita y usa su
`meta.json`, así preparación, modelo y umbral son de la misma versión también durante un cambio.

**Despliegue *push*:** lo lanza `promote.py`; mover `@production` desde la UI de MLflow no despliega nada, y
`promote.py status` lo detecta (`DRIFT`).

## TF Serving en Docker

| pieza | valor | por qué |
|---|---|---|
| imagen | `tensorflow/serving:2.21.0` (856 MB) | la misma versión que el TensorFlow del export |
| arranque | `entrypoint: tensorflow_model_server` con `--model_config_file=/models/models.config` y `--model_config_file_poll_wait_seconds=5` | versión fijada por fichero, releído cada 5 s (sustituye al script de la imagen, que usa `MODEL_NAME`) |
| volumen | `./models:/models:ro` | solo lectura: el servidor nunca escribe |
| puertos | ninguno publicado | solo la API habla con él, por la red interna `fire-serving_default` |
| healthcheck | `bash` + `/dev/tcp`: `GET /v1/models/fire_detector` y busca `AVAILABLE` | la imagen no trae `curl`, `wget` ni Python; comprueba que el modelo está cargado, no solo el puerto |

**`models.config`** (lo escribe `export.py`):

```
model_config_list {
  config {
    name: "fire_detector"
    base_path: "/models/fire_detector"
    model_platform: "tensorflow"
    model_version_policy { specific { versions: 3 } }
  }
}
```

***Warmup*:** sin él, la primera predicción tras cargar una versión tarda ~2,5 s (TF Serving prepara el grafo y
reserva memoria al primer uso). `export.py` escribe `assets.extra/tf_serving_warmup_requests`: 5 peticiones con un
JPEG gris (formato TFRecord de `PredictionLog`, con `tensorflow-serving-api`). TF Serving las ejecuta al cargar la
versión, **antes** de marcarla disponible (~3,5 s), y la primera petición real tarda ~300 ms.

Recursos: carga la versión en ~3,4 s; RAM de 580 MB en reposo a 1,7 GB bajo carga (B4).

## API

[`src/api/app.py`](../src/api/app.py), dependencias en [`src/api/requirements.txt`](../src/api/requirements.txt) (sin
TensorFlow).

### Endpoints

| endpoint | para quién | respuesta |
|---|---|---|
| `GET /` | personas | página web: arrastrar una imagen y ver el resultado |
| `POST /predict` | aplicaciones | `{prob, alarm, threshold, model_version, latency_ms: {preprocess, model}}` |
| `GET /model` | aplicaciones, la web, `promote.py status` | versión servida + su `meta.json` |
| `GET /health` | Docker / Kubernetes (**liveness**) | `{"status": "ok"}`; no llama a TF Serving |
| `GET /ready` | Docker / Kubernetes (**readiness**) | 200 con la versión, o 503 |
| `GET /docs` | personas | documentación interactiva |

`/health` es la sonda de *liveness* (el proceso vive) y `/ready` la de *readiness* (puede recibir tráfico).

| código | cuándo |
|---|---|
| 400 | el fichero no es una imagen |
| 422 | falta el campo `file` (lo genera FastAPI) |
| 503 | TF Serving no responde, no hay versión cargada o falta su `meta.json` (reintentar más tarde) |
| 502 | TF Serving responde con error |

Con TF Serving parado, `/health` responde 200 y el resto 503 con el motivo; al volver TF Serving, la API se recupera
sola.

### Diseño

- **multipart** (`curl -F "file=@foto.jpg"`), una imagen por petición.
- **`def`, no `async def`:** FastAPI ejecuta cada petición en un grupo de hilos; una lenta no bloquea las demás.
- **Configuración por variables de entorno** con valores por defecto para el host: `TF_SERVING_URL`, `MODEL_NAME`,
  `MODELS_DIR`. El mismo código funciona en el host y en Docker.
- **La versión se pregunta a TF Serving** (la que está `AVAILABLE`) y se lee su `meta.json`: cada versión puede
  necesitar otro tamaño, CLAHE o umbral, y la API se adapta sola. Con la v1 servida usa 224 px, sin CLAHE y umbral
  0,8788; con la v3, 380 px, CLAHE y 0,8228.
- **Reintento:** si la versión deja de existir entre preguntar y predecir (TF Serving acaba de cambiar), TF Serving
  responde 404 y `/predict` repite una vez con la versión nueva.
- **El umbral se aplica en la API:** TF Serving solo devuelve la probabilidad.
- **Log** de una línea por predicción (`file`, `version`, `prob`, `alarm`, `latency_ms`).

### Página web

[`src/api/static/index.html`](../src/api/static/index.html), servida por la propia API en `GET /`. Muestra la decisión,
la probabilidad frente al umbral, la versión y los tiempos de preparación y de modelo. La cabecera indica el modelo
servido (`GET /model`) y se refresca tras cada predicción, así que se ve un `deploy` o `rollback`.

| decisión | por qué |
|---|---|
| la sirve la API | mismo origen que `/predict`: sin CORS ni otro contenedor |
| HTML + JS sin librerías ni CDN | funciona sin internet y no hay nada que compilar |
| fuera de `/docs` (`include_in_schema=False`) | es la web, no parte de la API |
| solo llama a la API | TF Serving sigue sin puerto publicado; preparación y umbral están en un solo sitio |

### Imagen Docker

| fichero | qué |
|---|---|
| [`api.Dockerfile`](../deploy/serving/api.Dockerfile) | `python:3.12.16-slim`; primero `requirements.txt` (capa reutilizable), después el código con la misma estructura que `src/` (`api/`, `preprocess/clahe.py`); usuario no root; `uvicorn --host 0.0.0.0 --port 8080`. Imagen `fire-serving/api:1.0.0`, 373 MB |
| [`api.Dockerfile.dockerignore`](../deploy/serving/api.Dockerfile.dockerignore) | el contexto de construcción es el repo (`data/` pesa ~11 GB): deja pasar solo `src/api/` y `src/preprocess/clahe.py`, un contexto de 7 KB. Al llamarse como el Dockerfile, solo afecta a esta imagen |
| [`docker-compose.yml`](../deploy/serving/docker-compose.yml) | servicio `api`: `depends_on` TF Serving `healthy`, `./models:/models:ro` (los `meta.json`), puerto `127.0.0.1:8080`, healthcheck sobre `/ready` |

Dentro de la red de compose, la API llega a TF Serving por su nombre (`http://tf-serving:8501`).

Para desarrollar sin Docker (con TF Serving accesible en `127.0.0.1:8501`):

```bash
.venv/bin/uvicorn api.app:app --app-dir src --port 8080 --reload    # desde la raíz del repo
```

## Preparación fiel al entrenamiento

El modelo es sensible a diferencias de píxeles invisibles, así que la API prepara cada imagen **exactamente** como se
preparó el dataset, en `prepare()`:

```
foto original → CLAHE (si el modelo lo usa) → JPEG q95 → resize (INTER_AREA) → JPEG q95 → bytes a TF Serving
```

- Los scripts del dataset guardaron un JPEG tras cada paso, y el modelo aprendió con esos artefactos de compresión.
- TensorFlow decodifica JPEG con una DCT aproximada distinta de la de OpenCV: por eso decodifica el propio modelo.
- CLAHE tiene una única implementación ([`src/preprocess/clahe.py`](../src/preprocess/clahe.py)), compartida por los
  scripts del dataset y la API.

Sin esto, 14 de 228 imágenes cambiaban de decisión; con esto, diferencia media 0,0004 frente al entrenamiento y solo 2
cambios, ambos a menos de 0,001 del umbral. Investigación completa:
[learning/training_serving_skew.md](learning/training_serving_skew.md).

## Exportar una versión

[`src/registry/export.py`](../src/registry/export.py) (por defecto, la versión en `@production`):

1. comprueba que MLflow responde;
2. si la versión no está exportada (o con `--force`): la descarga, la exporta con la signature de bytes
   (`keras.export.ExportArchive`), añade el *warmup* y el `meta.json` en `.tmp-<v>/` y renombra a `<v>/`. El renombrado
   es atómico: TF Serving nunca ve una carpeta a medio escribir;
3. salvo `--no-activate`: escribe `models.config` fijando esa versión (fichero temporal + renombrado).

```bash
.venv/bin/python src/registry/export.py                       # la versión en @production
.venv/bin/python src/registry/export.py --version 1           # exportar (si hace falta) y servir la v1
.venv/bin/python src/registry/export.py --version 3 --force   # volver a exportar
.venv/bin/python src/registry/export.py --version 2 --no-activate
```

Exporta en CPU (`CUDA_VISIBLE_DEVICES=""`): ~35 s, frente a 4,7 min con la GPU, cuyo arranque domina. La conversión no
cambia el modelo: el `.keras` y el SavedModel dan exactamente las mismas probabilidades.

`deploy/serving/models/` está en `.gitignore`: son ficheros generados, y la fuente de verdad es MLflow.

## Despliegue

`promote.py` ([08](08_model_registry.md)) llama a `export.py` como proceso aparte, así que nunca carga TensorFlow:

```
promote.py deploy
  1. gate contra el @production actual
  2. export.py --no-activate        descarga y convierte: lo lento y lo que puede fallar, antes de tocar alias
  3. MLflow: @previous ← saliente, @production ← entrante, se quita @staging, tags de historial
  4. export.py                      models.config → "sirve la versión <v>"
  5. espera a que la API sirva <v> y verifica (ver abajo)
TF Serving (relee la config cada 5 s): carga <v>, ejecuta el warmup, cambia, descarga la anterior
```

`rollback` hace lo mismo hacia `@previous`. Esa versión ya está exportada, así que el paso 2 no hace nada.

Con `--env <sitio>` solo se mueven los alias: el TF Serving local es el entorno por defecto. En un CI real, este
proceso sería un *job* del pipeline.

**Cambio de versión con tráfico** (una petición cada ~0,3 s mientras se cambia v3 → v1 → v3):

| | resultado |
|---|---|
| peticiones | 328, **todas 200** |
| v3 → v1 (exportar + activar) | 24 s hasta la primera respuesta de la v1 |
| v1 → v3 (ya exportada) | 18 s |
| latencia máxima del modelo | 367 ms: sin *cold start* |
| `promote.py status` | con la v1 servida fuera del registro: `DRIFT`; tras volver: `OK` |

## Verificación tras el despliegue

Desplegar no basta: hay que comprobar que TF Serving ha cambiado de versión y que responde como en el entrenamiento.
[`src/registry/verify_serving.py`](../src/registry/verify_serving.py) prueba el servicio como lo usaría una cámara:

1. pregunta a la API qué versión sirve y con qué umbral (`GET /model`): se comprueba **lo servido**, no el alias;
2. descarga de MLflow las predicciones de test del run de esa versión (`predictions/predictions_test.csv`), así la
   referencia vale para cualquier versión;
3. manda las **fotos originales** a `POST /predict` (4 peticiones a la vez, `--workers`); `--limit N` toma N imágenes
   repartidas por todo el test;
4. aplica los criterios y sale con **código 1 si alguno falla**.

| criterio | regla por defecto | detecta |
|---|---|---|
| media de \|diferencia\| | ≤ 0,001 (`--max-mean`) | desajuste sistemático (por ejemplo, olvidar el CLAHE) |
| percentil 95 | ≤ 0,005 (`--max-p95`) | desajuste en una parte de las imágenes |
| cambios de decisión | solo si la probabilidad del entrenamiento está a menos de 0,01 del umbral (`--near`) | lejos del umbral es un error; pegado a él, ruido CPU/GPU |
| versión | todas las respuestas de la misma | un despliegue a mitad de la prueba invalida el resultado |

El máximo se imprime solo como información: una imagen aislada puede llegar a 0,01 sin que nada esté mal. La
decisión de referencia es `prob ≥ umbral`. La columna `alarm` del CSV es la etiqueta real, no la decisión del modelo.

```bash
.venv/bin/python src/registry/verify_serving.py --limit 200           # ~40 s
.venv/bin/python src/registry/verify_serving.py --out /tmp/check.csv  # todo el test, ~14 min con B4
```

**En `promote.py`:** al final de `deploy` y `rollback`, espera a que `GET /model` diga la versión nueva (cada 2 s,
máximo 180 s) y lanza `verify_serving.py --limit 200` (`--verify N` para cambiarlo, `0` para saltarlo). Si falla,
sale con código 1 y aconseja `rollback`; **no vuelve atrás solo**, porque quitar un modelo de producción es una
decisión de una persona o de CI. No se ejecuta con `--dry-run` ni con `--env`.

Resultado con la v3 y 200 imágenes: diferencia media 0,00022, p95 0,00136, máxima 0,00336; ningún cambio de decisión.

## Rendimiento en CPU

| | B4 (v3) | MobileNet (v1) |
|---|---:|---:|
| modelo, 1 petición | ~290–320 ms | ~72 ms |
| preparación en la API (CLAHE + JPEG) | ~30 ms | menos (sin CLAHE, 224 px) |
| rendimiento, 1 petición a la vez | ~3 img/s | — |
| rendimiento, 4 a la vez | 5,3 img/s (cada petición ~720 ms) | — |

- En paralelo, cada petición tarda más (se reparten la CPU), pero salen más imágenes por segundo.
- Con B4, una cámara a una imagen por segundo va sobrada; varias cámaras o vídeo, no.
- gRPC, *batching* o un modelo más ligero (MobileNet, TFLite/ONNX) mejorarían el rendimiento.

## Operación

```bash
docker compose -f deploy/serving/docker-compose.yml up -d            # arrancar (requiere deploy/serving/models/)
docker compose -f deploy/serving/docker-compose.yml up -d --build    # tras cambiar el código de la API
docker compose -f deploy/serving/docker-compose.yml ps               # ambos "healthy"
docker compose -f deploy/serving/docker-compose.yml logs -f api
docker compose -f deploy/serving/docker-compose.yml stop
.venv/bin/python src/registry/promote.py status                     # ¿sirve lo que dice @production?
```

## Limitaciones conocidas

- **Sensibilidad a la compresión JPEG:** que ~1,5 niveles de gris muevan la probabilidad 0,15 indica fragilidad. Las
  cámaras envían JPEG de calidades distintas: convendría reentrenar sin JPEG intermedios y con variaciones de calidad
  (*data augmentation*).
- **Pesos del SavedModel más grandes de lo esperado:** 141 MB para 17,7 M parámetros, frente a ~71 MB en float32.
- **Sin regla de latencia en el gate** ([08](08_model_registry.md#limitaciones-conocidas)).
- **Sin autenticación:** la API solo escucha en `127.0.0.1`.
