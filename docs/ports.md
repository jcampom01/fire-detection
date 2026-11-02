# Puertos y conexiones

## Puertos en uso

Stack `deploy/mlops` (proyecto Compose `fire-mlops`, red `fire-mlops_default`).

| servicio | puerto | protocolo | publicado en el host | URL / uso |
|---|---:|---|---|---|
| `mlflow` | 5000 | HTTP | `127.0.0.1:5000` | UI y API REST: http://localhost:5000 · health: `/health` |
| `rustfs` | 9001 | HTTP | `127.0.0.1:9001` | consola web: http://localhost:9001/rustfs/console/ |
| `rustfs` | 9000 | HTTP (API S3) | `127.0.0.1:9000` | contenedores: `http://rustfs:9000` · host (DVC): `http://localhost:9000` · health: `/health` |
| `adminer` (perfil `tools`) | 8081 → 8080 | HTTP | `127.0.0.1:8081` | UI de la base de datos: http://localhost:8081 |
| `postgres` | 5432 | PostgreSQL | no, solo red interna | `postgres:5432`, base de datos `mlflow` |

- **Publicado en `127.0.0.1`:** accesible solo desde esta máquina, no desde la red local.
- **Solo red interna:** solo lo alcanzan otros contenedores de la misma red, por el nombre del servicio.

Stack `deploy/serving` (proyecto Compose `fire-serving`, red `fire-serving_default`).

| servicio | puerto | protocolo | publicado en el host | URL / uso |
|---|---:|---|---|---|
| `api` | 8080 | HTTP | `127.0.0.1:8080` | `POST /predict` · `GET /model` · `/health` · `/ready` · `/docs`: http://localhost:8080/docs |
| `tf-serving` | 8501 | HTTP (REST) | no, solo red interna | `http://tf-serving:8501/v1/models/fire_detector/versions/<v>:predict` |
| `tf-serving` | 8500 | gRPC | no | clientes gRPC dentro de la red |

## Conexiones

| origen | destino | cómo | para qué |
|---|---|---|---|
| navegador (host) | `mlflow` | `http://localhost:5000` | ver runs, comparar, registro de modelos |
| navegador (host) | `rustfs` | `http://localhost:9001/rustfs/console/` | ver buckets y ficheros |
| `train.py` (host, futuro) | `mlflow` | `MLFLOW_TRACKING_URI=http://localhost:5000` | registrar runs y modelos (también los ficheros, vía proxy) |
| `mlflow` | `postgres` | `postgresql://…@postgres:5432/mlflow` | metadatos |
| `mlflow` | `rustfs` | `MLFLOW_S3_ENDPOINT_URL=http://rustfs:9000` | ficheros, bucket `mlflow` |
| `create_bucket.sh` (contenedor `aws-cli` temporal) | `rustfs` | `http://rustfs:9000` dentro de la red | crear buckets |
| `dvc` (host) | `rustfs` | `http://localhost:9000`, bucket `datasets` | subir y bajar versiones del dataset (cliente S3 directo, sin proxy) |
| navegador (host) | `adminer` | `http://localhost:8081` | inspeccionar las tablas de MLflow |
| `adminer` | `postgres` | `postgres:5432` dentro de la red | consultas (Postgres sigue sin puerto publicado) |
| aplicación / `curl` (host) | `api` | `http://127.0.0.1:8080` | predecir (`POST /predict`, multipart) |
| `api` | `tf-serving` | `http://tf-serving:8501` dentro de la red `fire-serving_default` | versión servida y predicción (JPEG en `{"b64": ...}`) |
| `promote.py status` (host) | `api` | `http://127.0.0.1:8080/model` (`SERVING_API_URL`) | versión servida, para detectar drift |
| `verify_serving.py` (host; también desde `promote.py deploy/rollback`) | `api` y `mlflow` | `SERVING_API_URL`, `MLFLOW_TRACKING_URI` | versión servida, predicciones de referencia del run y `POST /predict` con fotos del test |
| `export.py` / `promote.py` (host) | `mlflow` | `http://localhost:5000` | descargar la versión a desplegar (escribe en `deploy/serving/models/`, no habla con TF Serving) |

```
                      HOST (WSL2)
  navegador ─────────────┬──────────────────────────┐
  train.py ──┐           │                          │
             ▼           ▼                          ▼
      127.0.0.1:5000                         127.0.0.1:9001
  ┌──────────┼──────────── red fire-mlops_default ──┼──────────┐
  │          ▼                                      ▼          │
  │      ┌────────┐  postgres:5432   ┌──────────┐              │
  │      │ mlflow │ ───────────────► │ postgres │              │
  │      └────────┘                  └──────────┘              │
  │          │       rustfs:9000     ┌──────────┐              │
  │          └─────────────────────► │  rustfs  │ ◄── aws-cli  │
  │                                  └──────────┘  (create_bucket.sh)
  └────────────────────────────────────────────────────────────┘
```

## Puertos previstos (aún no existen)

Según el plan de [05_deployment.md](05_deployment.md):

| servicio | puerto | publicado | uso |
|---|---:|---|---|
| `prometheus` | 9090 | sí | métricas |
| `grafana` | 3000 | sí | dashboards |
| `locust` | 8089 | sí | pruebas de carga |
