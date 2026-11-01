# 10 — Servir el modelo con TF Serving

> Estado: **S.1 hecho**. La v3 (`@production`) exportada a SavedModel a mano y servida con TF Serving desde
> [`deploy/serving/docker-compose.yml`](../deploy/serving/docker-compose.yml); predicciones comprobadas frente al
> entrenamiento. Pendiente: la API (S.2), el despliegue automático desde MLflow (S.3), el compose completo (S.4) y la
> prueba de extremo a extremo (S.5).

## Arquitectura

La inferencia va **separada del modelo**: la API no carga el modelo, se lo pide a un servidor de modelos.

```
                       ┌──────── stack MLOps (deploy/mlops) ──────┐
                       │  MLflow ── Postgres     RustFS           │
                       └────▲─────────────────────────────────────┘
                            │ ¿@production? · descarga
           promote.py deploy ─► paso de despliegue: .keras → SavedModel
                            │
                            ▼
                    ./models/fire_detector/<versión>/  (+ meta.json)
                       │                     │
                 lee meta.json         carga la versión
                       ▼                     ▼
cliente ─POST /predict─► [ API ] ──REST──► [ TF Serving ]
        imagen original   preproceso        predice
                          umbral
       ◄─ {prob, alarm, model_version, latency_ms}
```

| pieza | hace | no hace |
|---|---|---|
| **API** (FastAPI, S.2) | decodificar, **CLAHE**, resize, llamar al modelo, **umbral**, validar, responder | no lleva TensorFlow |
| **TF Serving** | cargar SavedModel y predecir; cambiar de versión sin cortes | no preprocesa, no aplica el umbral, no sabe qué es MLflow |
| **paso de despliegue** (S.3) | descargar `@production`, convertir a SavedModel, escribir la carpeta | no sirve peticiones |

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
| `POST /v1/models/fire_detector/versions/3:predict` | `{"inputs": {"input_layer_1": [...]}}` → `{"outputs": [[0.012]]}` |

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

Signature (`saved_model_cli show --dir … --tag_set serve --signature_def serving_default`):

```
inputs['input_layer_1']  float32  (-1, 380, 380, 3)     -1 = cualquier número de imágenes
outputs['output_0']      float32  (-1, 1)
method: tensorflow/serving/predict
```

La entrada es float32, pero por dentro el modelo sigue calculando en float16 (mixed precision se conserva en el grafo).

### Versiones y cambio sin cortes

TF Serving revisa la carpeta (o su fichero de configuración) cada pocos segundos. Al aparecer una versión nueva la carga
en segundo plano mientras la anterior sigue respondiendo, cambia y descarga la anterior: **cero tiempo sin servicio**.

Cómo llega un `@production` nuevo:

```
promote.py deploy  → MLflow mueve el alias (instantáneo); TF Serving sigue con la versión anterior
  paso de despliegue:
    1. descarga la versión de MLflow
    2. carga el .keras y exporta a ./models/fire_detector/.tmp-<v>/  + meta.json
    3. renombra .tmp-<v> → <v>        (atómico: TF Serving nunca ve una carpeta a medio escribir)
    4. config de TF Serving: "sirve la versión <v>"
TF Serving: carga <v>, cambia, descarga la anterior
```

- **La conversión se hace una vez por versión**: la carpeta se queda, y un rollback a una versión ya convertida solo
  cambia la config (casi instantáneo).
- **Config en vez de "la versión más alta"** (comportamiento por defecto): con "la más alta", el rollback a una versión
  menor no funcionaría.
- **La API pide una versión explícita** (`/versions/<v>:predict`) y usa el `meta.json` de esa misma versión, para que
  preproceso, modelo y umbral sean siempre de la misma versión, también durante un cambio.

**Decisión: conversión al hacer `deploy` (push)**, no un servicio vigilando el registro (pull). Consecuencia: mover
`@production` desde la UI de MLflow no despliega nada. Se cubre con una regla (el alias solo se mueve con `promote.py`)
y detectándolo: el `/health` de la API comparará la versión servida con `@production` (`drift`).

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

[`deploy/serving/docker-compose.yml`](../deploy/serving/docker-compose.yml), proyecto `fire-serving`. Es el compose de la
"app": aquí se añadirán la API y el paso de despliegue.

| pieza | valor | por qué |
|---|---|---|
| imagen | `tensorflow/serving:2.21.0` | la misma versión que el TensorFlow del export (la última en Docker Hub); 856 MB |
| `MODEL_NAME` | `fire_detector` | el entrypoint de la imagen lo convierte en `--model_name` y `--model_base_path=/models/fire_detector` |
| volumen | `./models:/models:ro` | solo lectura: el servidor nunca escribe |
| puerto | `127.0.0.1:8501` | REST para pruebas; con la API delante quedará solo en la red interna. gRPC (8500) no se publica |
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

**Criterio para la prueba automática (S.5):** diferencia máxima < 0,01, y cambios de decisión solo en imágenes a menos
de esa distancia del umbral.

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

## Siguientes pasos

- **S.2** — API (FastAPI): imagen original → preproceso con `meta.json` (CLAHE, resize) → TF Serving (versión explícita)
  → umbral → `{prob, alarm, model_version, latency_ms}`. `/health` con detección de `drift` frente a `@production`.
- **S.3** — Paso de despliegue lanzado por `promote.py deploy` (contenedor de un solo uso), con `.tmp` + renombrado y la
  config de versiones de TF Serving.
- **S.4** — Compose completo; TF Serving solo en la red interna.
- **S.5** — Prueba de extremo a extremo con el test y prueba de `deploy` y `rollback`.
