# Fire Detection — clasificador de incendios con MLOps en local

Clasificador binario de **alarma de incendio** (fuego y/o humo frente a fondo) entrenado con transfer learning, y
todo el ciclo de vida del modelo montado en local con herramientas de producción: versionado de datos, tracking de
experimentos, registro de modelos con promoción por reglas, servicio con TF Serving, API y una página web para
probarlo.

Es un proyecto de **aprendizaje**: el objetivo es entender cada herramienta y por qué se usa, más que el modelo en sí.
La documentación del proyecto está en [docs/](docs/README.md), y las explicaciones en detalle de cada herramienta
(cómo funcionan DVC, MLflow y TF Serving por dentro) en [docs/learning/](docs/learning/README.md).

**Versión 1.0.0** — sirviendo `fire-detector` v3 (EfficientNetB4 con CLAHE): recall 0,93 y 0,85 % de falsas alarmas
en test.

## Arquitectura

```
  DATOS                      ENTRENAMIENTO                 REGISTRO (stack fire-mlops)
  data/ ◄── dvc pull ──┐
    │                  │     ┌──────────────┐  run +    ┌──────────────────────────────┐
    └──────────────────┼───► │   train.py   │ ────────► │ MLflow :5000                 │
                       │     └──────────────┘  modelo   │  runs · fire-detector v1..vN │
   RustFS bucket ◄─────┘                                │  alias @production …         │
   "datasets" (DVC)                                     │  Postgres (metadatos)        │
                                                        │  RustFS "mlflow" (ficheros)  │
                                                        └──────────────────────────────┘
                                                              ▲ alias,        │ descarga
                                                              │ métricas      ▼ el modelo
  DESPLIEGUE (host)          promote.py ── gate · alias ──────┘     export.py: SavedModel + warmup
                               │ deploy / rollback lanza ──────────►    │ escribe
                               │ verify_serving.py                      ▼
                               │                           deploy/serving/models/
                               │                           (versiones + models.config)
  SERVICIO (stack fire-serving)│                                        │ volumen
                               ▼                                        ▼
  navegador / curl ──► API FastAPI :8080 ── JPEG ──────────────► TF Serving (red interna)
                       web · CLAHE · resize · umbral                   lee la versión fijada
```

- **Los datos** viven en el bucket `datasets`; git solo guarda el `.dvc` con su hash.
- **`train.py`** registra cada entrenamiento en MLflow como una versión nueva de `fire-detector`.
- **`promote.py`** decide con el gate qué versión es `@production`, lanza `export.py` para dejarla en la carpeta que lee
  TF Serving y, al final, `verify_serving.py` para comprobar el resultado a través de la API.
- **TF Serving** no conoce MLflow: sirve lo que fija `models.config`. **La API** es la única entrada pública.

| stack | servicios | dónde |
|---|---|---|
| `fire-mlops` | MLflow :5000, Postgres, RustFS :9000/:9001, Adminer :8081 (opcional) | [deploy/mlops/](deploy/mlops/) |
| `fire-serving` | API + web :8080, TF Serving (sin puerto publicado) | [deploy/serving/](deploy/serving/) |

Todos los puertos se publican solo en `127.0.0.1`. Detalle: [docs/ports.md](docs/ports.md) ·
diagrama: [docs/mlops_architecture.drawio](docs/mlops_architecture.drawio).

## Requisitos

- Linux o WSL2 con Docker y Docker Compose.
- Python 3.12 para el entorno del host (entrenamiento, DVC, MLflow y scripts de despliegue).
- GPU NVIDIA para entrenar (opcional). El servicio funciona en CPU.
- RAM: el servicio con B4 usa ~1–2 GB y MLflow ~400 MB. El entrenamiento necesita la mayor parte del resto.

## Puesta en marcha desde cero

En un clon nuevo no hay datos, ni runs, ni modelos: todo se genera en local siguiendo este orden.

**1. Entorno de Python del host**

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
```

**2. Plataforma MLOps** ([docs/05](docs/05_platform.md))

```bash
cd deploy/mlops
cp .env.example .env                       # poner contraseñas reales; no se sube a git
docker compose up -d postgres rustfs
./scripts/create_bucket.sh                 # bucket "mlflow" (artefactos de MLflow)
./scripts/create_bucket.sh datasets        # bucket "datasets" (remote de DVC)
docker compose up -d --build
cd ../..
```

**3. Datos** ([docs/06](docs/06_data_versioning.md), [docs/02](docs/02_preprocessing.md))

```bash
# Credenciales del remote de DVC (las de RustFS en .env); van a .dvc/config.local, fuera de git
.venv/bin/dvc remote modify --local storage access_key_id <RUSTFS_ACCESS_KEY>
.venv/bin/dvc remote modify --local storage secret_access_key <RUSTFS_SECRET_KEY>

# a) Con acceso a un bucket que ya tiene los datos
.venv/bin/dvc pull

# b) Bucket vacío (instalación nueva): descargar el dataset de Kaggle y subirlo al bucket
#    https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
#    descomprimir de forma que quede data/original_dataset/{train,val,test}/{images,labels}
.venv/bin/dvc add data/original_dataset    # llena la caché (si el .dvc cambia, la descarga no es idéntica)
.venv/bin/dvc push

# Derivados (no versionados: se regeneran)
.venv/bin/python src/preprocess/prepare_labels.py     # etiquetas de clasificación
.venv/bin/python src/preprocess/clahe_preprocess.py   # copia con CLAHE
.venv/bin/python src/preprocess/resize_dataset.py     # copias a 224 y 380 px
```

**4. Primer modelo** ([docs/03](docs/03_training.md), [docs/08](docs/08_model_registry.md))

```bash
.venv/bin/python src/train/train.py --config src/train/configs/efficientnetb4_clahe.json   # → versión 1
.venv/bin/python src/registry/promote.py stage --version 1
.venv/bin/python src/registry/promote.py deploy --verify 0   # exporta a deploy/serving/models/ y fija la versión
```

`--verify 0` porque el servicio aún no está levantado. `deploy/serving/models/` no está en git: se genera desde
MLflow, que es la fuente de verdad. Sin esa carpeta, TF Serving no arranca y la API tampoco.

**5. Servicio** ([docs/09](docs/09_serving.md))

```bash
docker compose -f deploy/serving/docker-compose.yml up -d --build
.venv/bin/python src/registry/verify_serving.py --limit 200   # comprueba la API frente al entrenamiento
```

Abrir http://localhost:8080.

### Día a día

```bash
docker compose -f deploy/mlops/docker-compose.yml up -d
docker compose -f deploy/serving/docker-compose.yml up -d
```

Para pararlos, cambiar `up -d` por `stop`. **Nunca `down -v`**: borra los volúmenes, y con ellos todos los runs y
modelos.

## Uso

### Página web

http://localhost:8080 — arrastrar una imagen (o hacer clic para elegirla). Muestra:

- la decisión (alarma o no);
- la probabilidad frente al umbral;
- la versión del modelo;
- el tiempo de preparación (API) y de modelo (TF Serving).

La cabecera indica qué modelo se está sirviendo.

### API

| endpoint | qué hace |
|---|---|
| `POST /predict` | imagen (multipart, campo `file`) → `{prob, alarm, threshold, model_version, latency_ms}` |
| `GET /model` | versión servida y su `meta.json` (umbral, tamaño, CLAHE) |
| `GET /health` · `GET /ready` | el proceso está vivo · TF Serving responde con un modelo cargado |
| `GET /docs` | documentación interactiva |

```bash
curl -F file=@foto.jpg http://localhost:8080/predict
```

Errores: 400 si no es una imagen · 503 si TF Serving no responde o no hay modelo · 502 si TF Serving falla.

### Entrenar

```bash
.venv/bin/python src/train/train.py --config src/train/configs/efficientnetb4_clahe.json
```

Cada entrenamiento queda como run en MLflow (http://localhost:5000) con su configuración, curvas, métricas, commit y
hash de los datos, y registra una versión nueva de `fire-detector`.

### Registro y despliegue

```bash
P=".venv/bin/python src/registry/promote.py"
$P status                  # versiones, alias, gate y versión servida (OK / DRIFT)
$P stage --version 4       # gate (src/registry/promotion.json) → @staging
$P deploy                  # @staging → @production, exporta, sirve y verifica
$P rollback                # vuelve a @previous, sirve y verifica
```

- **El gate** compara las falsas alarmas en validación con las de `@production` y exige suelos mínimos en test.
- **Tras `deploy` y `rollback`**, el script espera a que la API sirva la versión nueva y la comprueba con 200
  imágenes del test frente a las predicciones del entrenamiento ([`verify_serving.py`](src/registry/verify_serving.py)).

## Estructura

```
src/
├── eda/            análisis exploratorio
├── preprocess/     etiquetas, CLAHE y resize (clahe.py lo comparte la API)
├── train/          entrenamiento, evaluación, tracking en MLflow e importación de runs
├── registry/       promote.py (alias y gate), export.py (a TF Serving), verify_serving.py
└── api/            API FastAPI + página web (static/index.html)
deploy/
├── mlops/          MLflow + Postgres + RustFS
└── serving/        TF Serving + API (models/ se genera, no está en git)
data/               datasets (versionados con DVC: solo los .dvc van a git)
results/            salidas de cada entrenamiento
docs/               documentación
```

## Documentación

| | |
|---|---|
| [01](docs/01_eda.md)–[04](docs/04_results.md) | datos, preprocesado, entrenamiento y resultados |
| [05](docs/05_platform.md) · [06](docs/06_data_versioning.md) | plataforma MLOps · versionado de datos con DVC |
| [07](docs/07_experiment_tracking.md) · [08](docs/08_model_registry.md) | tracking de experimentos · registro y promoción |
| [09](docs/09_serving.md) | servicio: TF Serving, API, *training-serving skew*, despliegue automático y verificación |
| [ports.md](docs/ports.md) · [tools.md](docs/tools.md) | puertos y conexiones · herramientas |
| [learning/](docs/learning/README.md) | DVC, MLflow y TF Serving en detalle; el caso de *training-serving skew* |

## Hoja de ruta

| fase | contenido | estado |
|---|---|---|
| 1. Datos y modelo | EDA, preprocesado, entrenamiento y comparación ([01](docs/01_eda.md)–[04](docs/04_results.md)) | ✅ |
| 2. Plataforma MLOps | DVC, MLflow tracking y registro con gate ([05](docs/05_platform.md)–[08](docs/08_model_registry.md)) | ✅ |
| 3. Servicio | TF Serving + API + web, despliegue, rollback y verificación ([09](docs/09_serving.md)) | ✅ v1.0.0 |
| 4. Observabilidad | Prometheus + Grafana (latencia, errores, tasa de alarmas por versión), simulador de cámaras, Locust | pendiente |
| 5. Kubernetes | kind: Deployment, Service, HPA y canary entre versiones (por ejemplo, B4 frente a MobileNet) | pendiente |
| 6. CI/CD | GitHub Actions: tests, imágenes, y gate + deploy + verificación en el pipeline | pendiente |
| 7. Monitorización del modelo | Evidently: drift simulado con imágenes sin CLAHE, oscurecidas o con niebla | pendiente |
| 8. Nube (opcional) | Cloud Run o EKS/GKE con capa gratuita | pendiente |

Extensiones: vídeo o RTSP con suavizado "k de N" (dar la alarma solo si k de las últimas N imágenes la dan),
exportación a TFLite/ONNX para CPU, y entrenar `efficientnetb4_original` para saber si CLAHE aporta en B4.

**Recursos:** WSL tiene ~7,7 GB de RAM. Para kind con el stack de monitorización conviene subirla a 12 GB; un modelo
MobileNet (o TFLite/ONNX) reduciría memoria y latencia.

## Dataset y licencia

Datos: [Smoke-Fire-Detection-YOLO](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) (Kaggle,
Sayed Gamal, **CC0: Public Domain**), una reorganización de [D-Fire](https://github.com/gaiasd/DFireDataset) (GAIA)
con split de validación: 21.527 imágenes con cajas de fuego y humo. Las imágenes no se incluyen en este repositorio.
Cita del artículo original de D-Fire: [docs/01](docs/01_eda.md#dataset).

## Seguridad

- `deploy/mlops/.env` y `.dvc/config.local` contienen credenciales y no se suben a git (plantilla: `.env.example`).
- La API no tiene autenticación: solo escucha en `127.0.0.1`. Para exponerla, hace falta al menos una API key y HTTPS.
