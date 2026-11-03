# Servir modelos en detalle

Conceptos detrás del servicio: por qué un servidor de modelos, cómo funciona TF Serving y qué patrones se usan en la industria. Uso en el proyecto: [09_serving.md](../09_serving.md).

## Por qué separar la API del modelo

Por qué separado (frente a una API que cargue el modelo):

- **API ligera** (sin TensorFlow) y servidor optimizado (C++, imagen oficial).
- **Cambiar de modelo no reinicia la API**: TF Serving carga la versión nueva en segundo plano y cambia sin cortes.
- **Escalan por separado**: la API por peticiones, el servidor por cómputo.

Alternativas valoradas:

| opción | por qué no |
|---|---|
| MLflow Model Serving (`mlflow models serve`) | carga el modelo de MLflow sin convertir, pero resuelve el alias solo al arrancar: cambiar de modelo = reiniciar, sin servicio mientras carga |
| BentoML | carga `.keras` directo, pero tiene su propio registro: solapa con MLflow |
| Triton (NVIDIA) | imagen de más de 10 GB, orientado a GPU: excesivo |
| ONNX Runtime | el más rápido en CPU, pero otra conversión que validar; opción para optimizar |
| servidor propio | reinventar versiones, recarga y batching |

## Conceptos

### TF Serving

Programa de Google (C++) que carga modelos de TensorFlow y responde predicciones. Se configura con una carpeta y un
fichero de configuración; no se programa.

| puerto | protocolo | uso |
|---|---|---|
| 8501 | REST (JSON) | el que usa la API |
| 8500 | gRPC (binario) | más rápido y compacto; no se usa |

| petición | respuesta |
|---|---|
| `GET /v1/models/fire_detector` | versiones cargadas y su estado (`AVAILABLE`) |
| `GET /v1/models/fire_detector/metadata` | la signature: entradas y salidas |
| `POST /v1/models/fire_detector/versions/3:predict` | `{"instances": [{"b64": "<jpeg>"}]}` → `{"predictions": [[0.012]]}` |

`{"b64": …}` es la forma de mandar datos binarios en el JSON de TF Serving. Otras funciones: *batching* (agrupa
peticiones simultáneas), *warmup* (predicciones de prueba al cargar) y métricas para Prometheus.

### SavedModel

TF Serving no lee `.keras` (necesita Python y Keras para reconstruir el modelo). Lee **SavedModel**: el modelo
traducido a un grafo de TensorFlow puro y autónomo, con los pesos y sin el optimizador, y con una o varias
**signatures** (`serving_default` es la que llama TF Serving).

```
deploy/serving/models/fire_detector/3/      "3" = versión 3 de MLflow
├── saved_model.pb                    grafo
├── variables/
│   ├── variables.data-…     141 MB   pesos
│   └── variables.index
├── fingerprint.pb
├── assets.extra/tf_serving_warmup_requests
└── meta.json                         umbral y preparación (añadido por export.py)
```

Signature (`saved_model_cli show --dir … --tag_set serve --signature_def serving_default`):

```
inputs['image_bytes']  string   (-1)          un JPEG por imagen; -1 = cualquier número de imágenes
outputs['alarm']       float32  (-1, 1)
grafo: decode_image → resize(antialias) → clip(0, 255) → modelo
```

La entrada son los **bytes del JPEG**, no píxeles: el modelo decodifica dentro del grafo igual que
`src/train/data.py` en el entrenamiento (ver [Preparación fiel al entrenamiento](training_serving_skew.md)).
Por dentro sigue calculando en float16 (la *mixed precision* se conserva en el grafo).

Qué hace el modelo servido y qué queda fuera:

| etapa | ¿dentro del SavedModel? | dónde está |
|---|---|---|
| arquitectura + pesos | ✅ | `variables/` |
| decodificar el JPEG | ✅ | grafo, igual que `data.py` |
| normalización del backbone | ✅ (recibe RGB 0–255) | grafo |
| estado del optimizador | ❌ | solo en el `.keras` de MLflow |
| **CLAHE**, **resize** y los JPEG intermedios | ❌ | la API, con los parámetros de `meta.json` |
| **umbral de decisión** | ❌ | `meta.json` (de la `metadata` del `MLmodel`); lo aplica la API |

### Versiones y cambio sin cortes

TF Serving relee su fichero de configuración cada pocos segundos. Si fija otra versión, la carga en segundo plano
mientras la anterior sigue respondiendo, ejecuta el *warmup*, cambia y descarga la anterior: **cero tiempo sin
servicio**.

- **La conversión se hace una vez por versión:** la carpeta se queda, y volver a una versión ya convertida solo
  cambia la config.
- **Versión fijada en la config**, no "la más alta" (el comportamiento por defecto): con "la más alta", un rollback a
  una versión menor no funcionaría.
- **La API pide una versión explícita** (`/versions/<v>:predict`) y usa el `meta.json` de esa misma versión: preparación,
  modelo y umbral son siempre de la misma versión, también durante un cambio.

**Despliegue *push*:** la conversión la lanza `promote.py deploy`, no un servicio que vigile el registro (*pull*).
Consecuencia: mover `@production` desde la UI de MLflow no despliega nada. Se cubre con una regla (el alias solo se
mueve con `promote.py`) y detectándolo: `promote.py status` compara `@production` con la versión que sirve la API
(`DRIFT`). No lo hace la API: el servicio no debe depender de la plataforma de entrenamiento (si MLflow cae, la API
sigue).

El registro sigue mandando: `@production` es la orden y la carpeta es la orden cumplida. Se mantienen el gate, el
rollback, la trazabilidad (`model_version` en la respuesta → run → commit → datos) y el historial.

### Cómo se hace en la industria

| patrón | idea |
|---|---|
| **CI/CD (push)**, el de este proyecto | la promoción dispara un pipeline: descargar, convertir, probar, desplegar |
| **modelo dentro de la imagen** | el pipeline construye `fire-serving:model-v3`; lo desplegado es inmutable; rollback = imagen anterior |
| **GitOps** (Argo CD, Flux) | la versión está en un fichero de un repo; promover = pull request; un agente aplica el cambio |
| **plataformas** (KServe, Seldon, SageMaker, Vertex AI) | se declara la URI del modelo; la plataforma descarga, sirve, hace canary y rollback |

En todos: el registro es la fuente de verdad, a producción se llega por un proceso (no con clics), lo desplegado es
inmutable y versionado, y se verifica antes de dar por bueno el cambio.

## FastAPI y uvicorn

**FastAPI** define qué responder a cada ruta: lee las anotaciones de tipo (`file: UploadFile = File(...)`) para saber
qué trae la petición, valida (falta el fichero → 422), convierte el `dict` devuelto en JSON y genera la documentación
(`/docs`, `/openapi.json`). **uvicorn** es el servidor: abre el puerto, entiende el HTTP y pasa cada petición a
FastAPI mediante el estándar **ASGI**.

`def` frente a `async def`: con funciones `def`, FastAPI ejecuta cada petición en un grupo de hilos, así que una llamada que espera (a TF Serving, al disco) no bloquea las demás. `async def` solo compensa si todo lo que se espera es asíncrono.

## Liveness y readiness

Liveness y readiness van separados: si falla el primero, el proceso está roto y se reinicia; si falla el segundo,
está vivo pero no se le manda tráfico (por ejemplo, mientras TF Serving carga). Mezclarlos provoca reinicios
innecesarios.
