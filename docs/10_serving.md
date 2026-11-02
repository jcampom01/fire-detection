# 10 — Servir el modelo con TF Serving

> Estado: **S.1–S.3 hechos**. TF Serving + API (FastAPI) en
> [`deploy/serving/docker-compose.yml`](../deploy/serving/docker-compose.yml); `promote.py deploy/rollback` exporta y
> activa la versión automáticamente. La API reproduce las predicciones del entrenamiento desde la foto original y el
> cambio de versión se hace sin errores ni *cold start*. Pendiente: prueba de extremo a extremo con todo el test y un
> ciclo real de `deploy`/`rollback` (S.5).

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
        foto original     preparación: CLAHE → JPEG → resize → JPEG    decodifica el JPEG
                          umbral                                         y predice
       ◄─ {prob, alarm, threshold, model_version, latency_ms}
```

| pieza | hace | no hace |
|---|---|---|
| **API** (FastAPI + uvicorn, S.2) | validar, preparar la imagen como el dataset, llamar al modelo, **umbral**, responder | no lleva TensorFlow |
| **TF Serving** | decodificar el JPEG y predecir; cargar la versión fijada en la config; cambiar sin cortes | no preprocesa, no aplica el umbral, no sabe qué es MLflow |
| **`export.py`** (S.3) | descargar la versión, convertir a SavedModel con *warmup*, escribir la carpeta, fijar la versión | no sirve peticiones |

Por qué separado (frente a una API que cargue el modelo):

- **API ligera** (sin TensorFlow) y servidor optimizado (C++, imagen oficial).
- **Cambiar de modelo no reinicia la API**: TF Serving carga la versión nueva en segundo plano y cambia sin cortes.
- **Escalan por separado**: la API por peticiones, el servidor por cómputo.

Alternativas valoradas:

| opción | por qué no (de momento) |
|---|---|
| MLflow Model Serving (`mlflow models serve`) | carga el modelo de MLflow sin convertir, pero resuelve el alias solo al arrancar: cambiar de modelo = reiniciar, sin servicio mientras carga |
| BentoML | carga `.keras` directo, pero tiene su propio registro: solapa con MLflow |
| Triton (NVIDIA) | imagen >10 GB, orientado a GPU: excesivo |
| ONNX Runtime | el más rápido en CPU, pero otra conversión que validar; opción futura para optimizar |
| servidor propio | reinventar versiones, recarga y batching |

## Conceptos

### TF Serving

Programa de Google (C++) que carga modelos de TensorFlow y responde predicciones. Se configura con una carpeta y un
nombre; no se programa. Imagen oficial `tensorflow/serving`.

| puerto | protocolo | uso |
|---|---|---|
| 8501 | REST (JSON) | fácil de probar con `curl` |
| 8500 | gRPC (binario) | más rápido y compacto; para clientes en producción |

| petición | respuesta |
|---|---|
| `GET /v1/models/fire_detector` | versiones cargadas y su estado (`AVAILABLE`) |
| `GET /v1/models/fire_detector/metadata` | la signature: entradas y salidas |
| `POST /v1/models/fire_detector/versions/3:predict` | `{"instances": [{"b64": "<jpeg>"}]}` → `{"predictions": [[0.012]]}` |

Otras funciones: *batching* (agrupa peticiones simultáneas), *warmup* (predicciones de prueba al cargar) y métricas
para Prometheus.

### SavedModel

TF Serving no lee `.keras` (formato de Keras, necesita Python y Keras para reconstruir el modelo). Lee **SavedModel**:
el modelo traducido a un grafo de TensorFlow puro, autónomo. Lo genera `model.export(ruta)`:

1. **traza** el modelo con una entrada simbólica `[N, 380, 380, 3]` y anota cada operación;
2. guarda el grafo en `saved_model.pb` y los pesos en `variables/`, **sin el optimizador**;
3. crea la signature `serving_default`, el punto de entrada que llama TF Serving.

```
fire_detector/3/                 "3" = versión 3 de MLflow
├── saved_model.pb       4,8 MB  grafo
├── variables/
│   ├── variables.data-… 141 MB  pesos
│   └── variables.index   71 KB  índice de los pesos
├── fingerprint.pb        98 B   huella del modelo
├── assets/                      vacío
└── meta.json            558 B   umbral y preproceso (añadido nuestro)
```

Signature de `model.export()` (`saved_model_cli show --dir … --tag_set serve --signature_def serving_default`):

```
inputs['input_layer_1']  float32  (-1, 380, 380, 3)     -1 = cualquier número de imágenes
outputs['output_0']      float32  (-1, 1)
method: tensorflow/serving/predict
```

La entrada es float32, pero por dentro el modelo sigue calculando en float16 (mixed precision se conserva en el grafo).

**Signature actual (desde S.2.2): bytes JPEG.** `export.py` no usa `model.export()` sino `keras.export.ExportArchive`
con una función propia que decodifica el JPEG dentro del grafo, igual que `src/train/data.py`:

```
inputs['image_bytes']  string   (-1)          un JPEG por imagen
outputs['alarm']       float32  (-1, 1)
grafo: decode_image → resize(antialias) → clip(0, 255) → modelo
```

Además lleva `assets.extra/tf_serving_warmup_requests` (S.3).

### Versiones y cambio sin cortes

TF Serving revisa la carpeta (o su fichero de configuración) cada pocos segundos. Al aparecer una versión nueva la carga
en segundo plano mientras la anterior sigue respondiendo, cambia y descarga la anterior: **cero tiempo sin servicio**.

Cómo llega un `@production` nuevo:

```
promote.py deploy
  1. export.py --no-activate: descarga la versión de MLflow, exporta a ./models/fire_detector/.tmp-<v>/
     (+ warmup + meta.json) y renombra .tmp-<v> → <v>   (atómico: TF Serving nunca ve una carpeta a medio escribir)
     Es lo lento y lo que puede fallar: va antes de tocar los alias
  2. MLflow: mueve @previous / @production, tags de historial
  3. export.py: models.config → "sirve la versión <v>"  (fichero temporal + renombrado)
TF Serving (relee la config cada 5 s): carga <v>, ejecuta el warmup, cambia, descarga la anterior
```

- **La conversión se hace una vez por versión**: la carpeta se queda, y un rollback a una versión ya convertida solo
  cambia la config (casi instantáneo).
- **Config en vez de "la versión más alta"** (comportamiento por defecto): con "la más alta", el rollback a una versión
  menor no funcionaría.
- **La API pide una versión explícita** (`/versions/<v>:predict`) y usa el `meta.json` de esa misma versión, para que
  preproceso, modelo y umbral sean siempre de la misma versión, también durante un cambio.

**Decisión: conversión al hacer `deploy` (push)**, no un servicio vigilando el registro (pull). Consecuencia: mover
`@production` desde la UI de MLflow no despliega nada. Se cubre con una regla (el alias solo se mueve con `promote.py`)
y detectándolo: `promote.py status` compara `@production` con la versión que sirve la API (`DRIFT`). No lo hace la
API: el servicio en producción no debe depender de la plataforma de entrenamiento (si MLflow cae, la API sigue).

El registro sigue decidiendo: `@production` es la orden y la carpeta es la orden cumplida. Se mantienen el gate, el
rollback, la trazabilidad (`model_version` en la respuesta → run → commit → datos) y el historial.

### Cómo se hace en la industria

| patrón | idea |
|---|---|
| **CI/CD (push)** — el nuestro | la promoción dispara un pipeline: descargar, convertir, tests, desplegar |
| **modelo dentro de la imagen** | el pipeline construye `fire-serving:model-v3`; lo desplegado es inmutable; rollback = imagen anterior |
| **GitOps** (Argo CD, Flux) | la versión está en un fichero de un repo; promover = pull request; un agente aplica el cambio |
| **plataformas** (KServe, Seldon, SageMaker, Vertex AI) | se declara la URI del modelo; la plataforma descarga, sirve, hace canary y rollback |

En todos: el registro es la fuente de verdad, a producción se llega por un proceso (no con clics), lo desplegado es
inmutable y versionado, y hay verificación antes de servir.

## Pasos realizados

### S.1.a / S.1.b — Descargar y exportar

```python
v = mlflow.MlflowClient().get_model_version_by_alias("fire-detector", "production")       # → v3
path = mlflow.artifacts.download_artifacts(f"models:/fire-detector/{v.version}", dst_path=…)
meta = mlflow.models.Model.load(path).metadata                                             # meta.json
model = keras.models.load_model(f"{path}/data/model.keras")
model.export("deploy/serving/models/fire_detector/3")
```

| paso | tiempo |
|---|---:|
| descarga | 3,0 s |
| carga del `.keras` | 5,9 s |
| export | 21 s |
| total del proceso (incluye arrancar/cerrar TensorFlow con GPU) | 4,7 min |

RAM máxima 2,0 GB. Tamaños: `results/` 290 MB → MLflow 174 MB → SavedModel 140 MB. Pendiente: los pesos ocupan 141 MB,
más del doble de lo esperable para 17,7 M parámetros en float32 (~71 MB).

`deploy/serving/models/` está en `.gitignore`: son ficheros generados; la fuente de verdad es MLflow.

### S.1.c — ¿Da lo mismo el SavedModel que el `.keras`?

Comprobación por tramos, para saber qué paso falla si algo no cuadra:

```
model.keras ─────────────────► A
   │ export()
SavedModel (Python, GPU) ────► B          S.1.c: ¿A = B?   aísla la conversión
   │ TF Serving
REST (CPU) ──────────────────► C          S.1.e: ¿B = C?   aísla servidor y transporte
                                          referencia: prob de predictions_test.csv
```

228 imágenes de `data/clahe_dataset_380/test` (las que vio la evaluación): 200 al azar + las 29 a ±0,02 del umbral
(0,8228), que son las únicas que una diferencia pequeña podría cambiar de decisión.

| comparación | diferencia máx. | media | decisiones que cambian |
|---|---:|---:|---:|
| A `.keras` vs B SavedModel | **0** | **0** | 0 |
| A / B vs CSV del entrenamiento | 0,0032 | 0,00013 | 0 |

La conversión no cambia nada. Frente al entrenamiento hay diferencias mínimas por float16 (dependen del tamaño de lote y
del algoritmo que elige la GPU).

### S.1.d — TF Serving con Docker Compose

[`deploy/serving/docker-compose.yml`](../deploy/serving/docker-compose.yml), proyecto `fire-serving`: el compose de la
"app" (en S.2.4 se añadió la API). Valores de S.1 y cómo han quedado:

| pieza | valor | por qué |
|---|---|---|
| imagen | `tensorflow/serving:2.21.0` | la misma versión que el TensorFlow del export (la última en Docker Hub); 856 MB |
| modelo | S.1: `MODEL_NAME=fire_detector` (sirve la versión más alta) → S.3: `--model_config_file=/models/models.config` (versión fijada) | ver S.3 |
| volumen | `./models:/models:ro` | solo lectura: el servidor nunca escribe |
| puerto | S.1: `127.0.0.1:8501` para pruebas → S.2.4: **ninguno** | solo la API habla con él, por la red interna. gRPC (8500) nunca se publicó |
| healthcheck | `bash` + `/dev/tcp` | la imagen no trae `curl`, `wget` ni Python. `bash` abre la conexión TCP, envía `GET /v1/models/fire_detector` y busca `AVAILABLE`: comprueba que el modelo está cargado, no solo el puerto |
| sin `restart` | — | como el stack de MLOps |

```bash
cd deploy/serving
docker compose up -d     # arrancar
docker compose ps        # healthy
docker compose logs -f
docker compose down      # parar; los modelos siguen en ./models
```

Al arrancar: `Successfully loaded servable version {name: fire_detector version: 3}` en 3,4 s;
`GET /v1/models/fire_detector` → versión 3 `AVAILABLE`. RAM: 580 MB en reposo, 1,7 GB tras procesar lotes.

### S.1.e — Predicciones por REST y rendimiento en CPU

Mismas 228 imágenes, pidiendo la versión explícita (`/versions/3:predict`):

| comparación | diferencia máx. | media | decisiones que cambian |
|---|---:|---:|---:|
| C TF Serving (CPU) vs B SavedModel (GPU) | 0,0039 | 0,0003 | 1 |
| C TF Serving (CPU) vs CSV del entrenamiento | 0,0048 | 0,0003 | 1 |

La CPU calcula float16 de forma distinta a la GPU, y las diferencias crecen un poco. La imagen que cambia estaba
prácticamente en el umbral: en todo el test solo 4 de 4.306 imágenes (0,09 %) están a menos de 0,004 de él. Es
inevitable con cualquier umbral, no un error.

**Criterio para la prueba automática (S.5):** media y percentil 95 de la diferencia pequeños, y cambios de decisión
solo en imágenes muy cerca del umbral. Solo el máximo no sirve: una imagen aislada puede pasar de 0,01 (S.2.2).

Rendimiento de B4 en CPU (16 núcleos):

| caso | tiempo | rendimiento | petición JSON |
|---|---:|---:|---:|
| 1 imagen | 320 ms (303–370) | ~3 img/s | 3,0 MB |
| lote de 16 | 4,1 s | 3,9 img/s | 48,7 MB |
| 228 imágenes en lotes de 16 | 86 s | 2,6 img/s | — |

- Una cámara a una imagen por segundo: sobra. Varias cámaras o vídeo: no llega.
- Parte del tiempo es el JSON (433.200 números en texto por imagen). gRPC o una entrada `uint8` lo reducirían.
- MobileNet sería del orden de 10 veces más rápido. Queda pendiente decidir si el gate incluye una regla de latencia,
  midiendo de punta a punta cuando exista la API.

### S.2.1 — API mínima

[`src/api/app.py`](../src/api/app.py), dependencias en [`src/api/requirements.txt`](../src/api/requirements.txt) (sin
TensorFlow; también instaladas en el `.venv` para desarrollar).

**FastAPI** define qué responder a cada ruta: lee las anotaciones de tipo (`file: UploadFile = File(...)`) para saber
qué trae la petición, valida (falta el fichero → 422), convierte el `dict` devuelto en JSON y genera la documentación
(`/docs`, `/openapi.json`: el contrato con la futura aplicación). **uvicorn** es el servidor: abre el puerto, entiende el
HTTP y pasa cada petición a FastAPI mediante el estándar **ASGI**.

```bash
.venv/bin/uvicorn api.app:app --app-dir src --port 8080    # desde la raíz del repo
```

`--app-dir src` pone `src/` en la ruta de Python (`api.app` y `preprocess.clahe`). Sin `--host`, solo escucha en
`127.0.0.1`; en Docker hace falta `--host 0.0.0.0`. `--reload` (reiniciar al cambiar el código) solo para desarrollar.

Decisiones de `/predict`:

- **multipart** (`curl -F "file=@foto.jpg"`), una imagen por petición.
- `def` y no `async def`: hace llamadas que esperan (TF Serving, disco), y FastAPI ejecuta las funciones `def` en un
  grupo de hilos, así una petición lenta no bloquea las demás.
- **Configuración por variables de entorno** con valores por defecto para el host: `TF_SERVING_URL`, `MODEL_NAME`,
  `MODELS_DIR`. El mismo código funciona en el host y en Docker.
- **La versión se pregunta a TF Serving** (`GET /v1/models/fire_detector`, la que está `AVAILABLE`) y se lee su
  `meta.json`: cada versión puede necesitar otro tamaño, CLAHE o umbral. Así la API se adapta sola al modelo servido.
- **Se pide la versión explícita** (`/versions/<v>:predict`): preparación, modelo y umbral siempre de la misma versión.
- El umbral se aplica en la API: TF Serving solo devuelve la probabilidad.

### S.2.2 — Preparación fiel al entrenamiento (*training-serving skew*)

CLAHE se movió a [`src/preprocess/clahe.py`](../src/preprocess/clahe.py) (solo depende de OpenCV), que importan
`clahe_preprocess.py` y la API: **una única implementación**. Importarla de `clahe_preprocess.py` habría obligado a la
API a instalar `tqdm`; copiarla, a mantener dos versiones que podrían divergir sin que nada falle.

Primera prueba con fotos originales del test (CLAHE → resize → píxeles en JSON): diferencias de hasta 0,04 con el
entrenamiento, y `PublicDataset01134` (humo, 0,8235 en el entrenamiento) pasó a 0,7839, **sin alarma**. En S.1.e,
enviando los ficheros del dataset, el máximo era 0,005: el problema estaba en la preparación.

**Investigación** (228 imágenes, todas por el mismo TF Serving):

| cadena | vs fichero del dataset |
|---|---|
| A: la API (CLAHE → resize, en memoria) | píxeles: diferencia media 1,54 · predicciones: máx 0,15, **14 decisiones cambian** |
| A2: la API + escribir y leer JPEG (calidad 95) tras el CLAHE y tras el resize | píxeles: **0,00** · predicciones: **idénticas** |

Y una segunda causa, comparando decodificadores sobre 200 ficheros del dataset:

| | diferencia media de píxel |
|---|---:|
| OpenCV vs `tf.io.decode_jpeg` por defecto (lo que usa `data.py`) | 1,52 |
| OpenCV vs `tf.io.decode_jpeg(dct_method="INTEGER_ACCURATE")` | 0,000 (200/200 idénticas) |

**Causas:**

1. **JPEG intermedios.** Los scripts del dataset guardan un JPEG (calidad 95) tras el CLAHE y otro tras el resize: el
   modelo se entrenó y evaluó con esos artefactos de compresión.
2. **Decodificador.** TensorFlow descomprime JPEG por defecto con una DCT rápida y aproximada; OpenCV, con la exacta.
   El modelo se entrenó con los píxeles de la rápida.

Cada una mueve los píxeles ~1,5 niveles de gris de media (invisible), y eso mueve la probabilidad hasta 0,15–0,19. El
umbral (elegido para recall 95 % con la preparación del entrenamiento) deja de estar calibrado: no hay errores, solo
un comportamiento distinto del que se validó.

**Solución (opción elegida entre: replicar / aceptar / reentrenar):**

- La API reproduce la cadena del dataset: CLAHE → **JPEG** → resize → **JPEG** (calidad de `meta.json`,
  `jpeg_quality`), en la función `prepare()`.
- Envía **los bytes del JPEG** (`{"instances": [{"b64": "..."}]}`, la forma de mandar datos binarios en JSON a TF
  Serving), y el modelo los decodifica **dentro del grafo** con `tf.io.decode_image`, como `data.py`.
- [`src/registry/export.py`](../src/registry/export.py) exporta con esa signature (`image_bytes` → `alarm`) y añade
  `jpeg_quality` al `meta.json`; `tracking.py` lo incluye en los entrenamientos futuros.

Resultado con 228 fotos originales a través de `prepare()` (el mismo código de la API) y TF Serving:

| | antes (píxeles) | ahora (bytes JPEG) |
|---|---:|---:|
| diferencia máx. vs CSV | 0,1458 | 0,0107 |
| media | 0,0116 | **0,0004** |
| p95 | 0,0630 | **0,0021** |
| decisiones que cambian | 14 | **2** (a menos de 0,001 del umbral) |
| tamaño de la petición | ~1.500 KB | **85 KB** |

Lo que queda es el ruido numérico de float16 en CPU (como en S.1.e). Exportar en CPU (`CUDA_VISIBLE_DEVICES=""`) bajó
el tiempo de 4,7 min a 35 s.

**Pendiente (modelo):** que ~1,5 niveles de gris muevan la probabilidad 0,15 indica fragilidad a la compresión. Las
cámaras reales envían JPEG de calidades distintas: conviene reentrenar sin JPEG intermedios y con variaciones de
calidad JPEG (*data augmentation*).

### S.2.3 — Endpoints y errores

| endpoint | para quién | respuesta |
|---|---|---|
| `POST /predict` | la aplicación | `{prob, alarm, threshold, model_version, latency_ms: {preprocess, model}}` |
| `GET /model` | la aplicación, personas, `promote.py status` | versión servida + su `meta.json` |
| `GET /health` | Docker / Kubernetes (**liveness**) | `{"status": "ok"}`; no llama a TF Serving |
| `GET /ready` | Docker / Kubernetes (**readiness**) | 200 con la versión, o 503 |
| `GET /docs` | personas | documentación interactiva |

Liveness y readiness separados: si falla el primero, el proceso está roto y se reinicia; si falla el segundo, está vivo
pero no se le manda tráfico (por ejemplo, TF Serving cargando). Mezclarlos provoca reinicios innecesarios.

| código | cuándo |
|---|---|
| 400 | el fichero no es una imagen |
| 422 | falta el campo `file` (lo genera FastAPI) |
| 503 | TF Serving no responde, no hay versión cargada o falta su `meta.json` (inténtalo luego) |
| 502 | TF Serving responde con error |

- **Reintento:** si la versión deja de existir entre preguntar y predecir (TF Serving acaba de cambiar), TF Serving
  responde 404 y `/predict` repite una vez con la versión nueva y su `meta.json`.
- **Log** de una línea por predicción (`file`, `version`, `prob`, `alarm`, `latency_ms`): base para la monitorización.

Comprobado con TF Serving parado: `/health` 200; `/ready`, `/model` y `/predict` 503 con el motivo; al volver TF
Serving, la API se recupera sola.

### S.2.4 — La API en Docker

| fichero | qué |
|---|---|
| [`api.Dockerfile`](../deploy/serving/api.Dockerfile) | `python:3.12.16-slim`; primero `requirements.txt` (capa reutilizable), después el código con la misma estructura que `src/` (`api/`, `preprocess/clahe.py`); usuario no-root; `uvicorn --host 0.0.0.0 --port 8080`. **373 MB** |
| [`api.Dockerfile.dockerignore`](../deploy/serving/api.Dockerfile.dockerignore) | el contexto de construcción es el repo (`data/` pesa ~11 GB): deja pasar solo `src/api/` y `src/preprocess/clahe.py` → contexto de **7 KB**. Al llamarse como el Dockerfile, solo afecta a esta imagen |
| [`docker-compose.yml`](../deploy/serving/docker-compose.yml) | servicio `api`: `build` con contexto `../..`, imagen `fire-serving/api:0.1.0`, `depends_on` TF Serving `healthy`, `./models:/models:ro` (los `meta.json`), puerto `127.0.0.1:8080`, healthcheck sobre `/ready` |

Dentro de la red de compose la API llega a TF Serving por su nombre (`http://tf-serving:8501`, en el Dockerfile).

```bash
cd deploy/serving
docker compose up -d --build     # --build tras cambiar el código de la API
docker compose logs -f api
```

### S.3 — Despliegue automático

**[`src/registry/export.py`](../src/registry/export.py)** despliega una versión (por defecto, la de `@production`):

1. comprueba MLflow (`/health`) con un mensaje claro si no responde;
2. si la versión no está exportada (o `--force`): descarga, exporta con la signature de bytes, añade el *warmup* y el
   `meta.json` en `.tmp-<v>/` y renombra a `<v>/`;
3. salvo `--no-activate`: escribe `models.config` fijando esa versión (fichero temporal + renombrado).

Las exportaciones se conservan: volver a una versión ya exportada solo reescribe la config.

**`models.config`** (TF Serving lo relee cada 5 s, `--model_config_file_poll_wait_seconds=5`):

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

Con la política por defecto (la versión más alta), un rollback a una versión menor no funcionaría. En el compose,
`entrypoint: tensorflow_model_server` con los argumentos explícitos sustituye al script de arranque de la imagen
(que usaba `MODEL_NAME`).

***Warmup*:** la primera predicción tras cargar una versión tardaba 2,5 s (TF Serving prepara el grafo y reserva
memoria al primer uso). `export.py` escribe `assets.extra/tf_serving_warmup_requests`: 5 peticiones (formato
TFRecord de `PredictionLog`, con `tensorflow-serving-api==2.21.0`, que solo añade ese paquete al `.venv`) con un JPEG
gris. TF Serving las ejecuta al cargar la versión, **antes** de marcarla disponible (unos 3,5 s en el log). Resultado:
la primera predicción real tarda ~300 ms.

**[`promote.py`](../src/registry/promote.py)** llama a `export.py` como proceso aparte (con el Python del `.venv`), así
`promote.py` sigue sin cargar TensorFlow:

| comando | pasos |
|---|---|
| `deploy` | gate → `export.py --no-activate` (lento, puede fallar: **antes** de tocar alias) → alias y tags → `export.py` (activa) |
| `rollback` | `export.py --no-activate` (normalmente ya exportada) → alias y tags → `export.py` (activa) |
| `status` | además de la tabla: `@production = v3   served = v3   OK` o `DRIFT`, preguntando a la API (`/model`) |
| `--env <sitio>` | solo alias: el TF Serving de aquí es el entorno por defecto |
| `--dry-run` | muestra también el comando de `export.py` que lanzaría |

Cambio respecto al diseño: el despliegue corre como proceso en el host (el `.venv` tiene TensorFlow y MLflow), no como
contenedor de un solo uso. En un CI real sería un *job* del pipeline.

**Prueba de cambio de versión con tráfico:** una petición cada ~0,3 s a `/predict` mientras se cambia v3 → v1 → v3
con `export.py` (sin tocar alias):

| | resultado |
|---|---|
| peticiones | 328, **todas 200** |
| v3 → v1 (exportar + activar) | lanzado 18:37:00 → primera respuesta de la v1 a las 18:37:24 |
| v1 → v3 (ya exportada) | lanzado 18:38:00 → de vuelta a las 18:38:18 |
| latencia máxima del modelo | 367 ms: sin *cold start* |
| la API con la v1 | umbral 0,8788 en vez de 0,8228, 224 px sin CLAHE, **72 ms** (MobileNet, ~4 veces más rápido que B4) |
| `promote.py status` | con la v1 servida: `DRIFT`; tras volver: `OK` |

Estado final: sirviendo la v3 (= `@production`); exportadas la v1 y la v3.

## Siguientes pasos

- **S.5** — Prueba de extremo a extremo con todo el test a través de la API (criterio: media/p95 de la diferencia y
  cambios de decisión solo junto al umbral) y un ciclo real de `promote.py deploy` / `rollback`.
- Regla de latencia en el gate (B4 ~290 ms frente a MobileNet ~72 ms en CPU).
- Robustez del modelo a la compresión JPEG (reentrenar con variaciones de calidad).
- Pesos del SavedModel mayores de lo esperado (141 MB frente a ~71 MB).
