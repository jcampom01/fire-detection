# Fire Detection — clasificador de incendios con MLOps en local

Clasificador binario de **alarma de incendio** (fuego y/o humo frente a fondo) entrenado con transfer learning, y
todo el ciclo de vida del modelo montado en local con herramientas de producción: versionado de datos, tracking de
experimentos, registro de modelos con promoción por reglas, servicio con TF Serving, API y una página web para
probarlo.

Es un proyecto de **aprendizaje**: el objetivo es entender cada herramienta y por qué se usa, más que el modelo en sí.
La documentación detallada (conceptos, decisiones, incidencias) está en [docs/](docs/README.md).

**Versión 1.0.0** — sirviendo `fire-detector` v3 (EfficientNetB4 con CLAHE): recall 0,93 y 0,85 % de falsas alarmas
en test.

## Arquitectura

```
 DATOS                 ENTRENAMIENTO             REGISTRO                     SERVICIO
 data/original_dataset  train.py ──► MLflow       promote.py                   export.py ──► TF Serving
   │ dvc push            (params, métricas,       stage   → @staging           (SavedModel,    ▲  red interna
   ▼                      modelo, commit,         deploy  → @production         warmup)        │
 RustFS (S3)              hash de los datos)      rollback→ @previous                     API FastAPI :8080
 bucket "datasets"         │                      gate: promotion.json                     ▲          ▲
                           ▼                      verificación tras desplegar      página web     curl / apps
                       Postgres (metadatos)                                       (arrastrar foto)
                       RustFS bucket "mlflow" (ficheros)
```

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

```bash
# 1. Entorno de Python del host
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt

# 2. Plataforma MLOps (la primera vez: contraseñas y bucket)
cd deploy/mlops
cp .env.example .env                       # poner contraseñas reales; no se sube a git
docker compose up -d postgres rustfs
./scripts/create_bucket.sh
docker compose up -d --build
cd ../..

# 3. Datos (credenciales de RustFS en .dvc/config.local, ver docs/07)
.venv/bin/dvc pull

# 4. Modelo para servir: exporta la versión en @production a deploy/serving/models/
.venv/bin/python src/registry/export.py

# 5. Servicio
docker compose -f deploy/serving/docker-compose.yml up -d --build
```

`deploy/serving/models/` no está en git: se genera desde MLflow, que es la fuente de verdad. Si no existe, TF Serving
arranca sin modelo y la API nunca llega a estar lista. **Sin el paso 4, el servicio no funciona.**

En un clon nuevo, el registro de MLflow está vacío: hay que entrenar ([docs/03](docs/03_training.md)) o importar
resultados previos (`src/train/import_results.py`), y promover una versión (paso de *Registro* más abajo).

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
| [05](docs/05_deployment.md) | plan de despliegue y hoja de ruta |
| [06](docs/06_mlops.md) · [07](docs/07_data_versioning.md) | plataforma MLOps · versionado de datos con DVC |
| [08](docs/08_experiment_tracking.md) · [09](docs/09_model_registry.md) | tracking de experimentos · registro y promoción |
| [10](docs/10_serving.md) | servicio: TF Serving, API, *training-serving skew*, despliegue automático y verificación |
| [ports.md](docs/ports.md) · [tools.md](docs/tools.md) | puertos y conexiones · herramientas |

## Hoja de ruta

Hecho: datos versionados · tracking · registro con gate · servicio con despliegue, rollback y verificación · web.
Siguiente, según [05](docs/05_deployment.md): observabilidad (Prometheus, Grafana, Locust), Kubernetes (kind),
CI/CD (GitHub Actions), monitorización del modelo (Evidently).

## Seguridad

- `deploy/mlops/.env` y `.dvc/config.local` contienen credenciales y no se suben a git (plantilla: `.env.example`).
- La API no tiene autenticación: solo escucha en `127.0.0.1`. Para exponerla, hace falta al menos una API key y HTTPS.
