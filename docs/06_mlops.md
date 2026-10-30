# 06 — Plataforma MLOps (tracking y registro de modelos)

> Estado: **infraestructura levantada** (Postgres + RustFS + MLflow) y comprobada de extremo a extremo desde el host.
> Pendiente: integración con `train.py`, importar los modelos ya entrenados y gate de promoción.
> Conceptos de MLflow y prueba de humo: [08_experiment_tracking.md](08_experiment_tracking.md).

## Qué resuelve

Sin la plataforma, cada entrenamiento deja una carpeta `results/<exp>/` y no hay forma sencilla de saber qué modelo
está en producción, con qué config y entorno se entrenó, ni de comparar runs. MLflow cubre dos funciones:

- **Tracking:** cada ejecución de `train.py` es un **run** con params (config), métricas (por epoch y finales),
  artefactos (ficheros) y tags. Los runs se agrupan en experimentos.
- **Model Registry:** el modelo registrado `fire-detector` tiene versiones (v1, v2…) y alias móviles
  (`@staging`, `@production`). Los consumidores piden `models:/fire-detector@production`, nunca un número de versión,
  así que desplegar o hacer rollback es mover un alias.

| ahora en `results/<exp>/` | en MLflow |
|---|---|
| `config.json` | params |
| `history.csv` | métricas por epoch |
| `metrics.json` | métricas finales |
| `model.keras` + umbral + CLAHE | modelo registrado + `meta.json` |
| png, `predictions_*.csv` | artefactos |

## Arquitectura

```
train.py (host) ──► MLflow :5000 ──┬──► Postgres :5432   metadatos: runs, params, métricas, versiones, alias
                    (API + UI)     └──► RustFS   :9000   ficheros: bucket "mlflow"
```

| servicio | imagen | guarda | volumen |
|---|---|---|---|
| `postgres` | `postgres:17-alpine` | metadatos (MLflow crea sus tablas al arrancar, 65 en la 3.17) | `postgres_data` |
| `rustfs` | `rustfs/rustfs:1.0.4` | ficheros, con API S3 | `rustfs_data` |
| `mlflow` | `fire-mlops/mlflow:3.17.2` (propia) | nada: sin estado | — |

- MLflow arranca con `--serve-artifacts`: hace de **intermediario** hacia RustFS. Los clientes solo hablan con MLflow,
  así que los clientes de MLflow no necesitan credenciales de S3. El puerto 9000 sí se publica (en `127.0.0.1`)
  para DVC, que es un cliente S3 directo y guarda los datasets en el bucket `datasets`.
- `depends_on` + healthchecks: MLflow no arranca hasta que Postgres y RustFS están `healthy`.
- Puertos y conexiones: [ports.md](ports.md). Herramientas: [tools.md](tools.md).
  Diagrama conceptual: [mlops_architecture.drawio](mlops_architecture.drawio).

## Ficheros

```
deploy/mlops/
├── docker-compose.yml        postgres + rustfs + mlflow
├── .env.example              plantilla de variables (se versiona)
├── .env                      variables reales con contraseñas (NO se versiona)
├── mlflow/Dockerfile         imagen oficial + psycopg (Postgres) + boto3 (S3)
└── scripts/create_bucket.sh  crea un bucket en RustFS (idempotente)
```

## Puesta en marcha

```bash
cd deploy/mlops
cp .env.example .env                  # solo la 1ª vez: poner contraseñas reales
docker compose up -d postgres rustfs
./scripts/create_bucket.sh            # solo la 1ª vez: crea el bucket "mlflow"
docker compose up -d --build          # construye la imagen de MLflow y la arranca
docker compose ps                     # los 3 servicios en "healthy"
```

Interfaces:

- MLflow: http://localhost:5000
- Consola de RustFS: http://localhost:9001/rustfs/console/ (usuario `RUSTFS_ACCESS_KEY`, contraseña `RUSTFS_SECRET_KEY`)

## Comandos habituales

| comando | efecto |
|---|---|
| `docker compose ps` | estado y healthchecks |
| `docker compose logs -f mlflow` | logs de un servicio |
| `docker compose stop` / `start` | parar / arrancar sin borrar nada |
| `docker compose down` | borra los contenedores; **los datos siguen en los volúmenes** |
| `docker compose up -d --build mlflow` | reconstruir la imagen tras cambiar el Dockerfile (sin `--build` usa la vieja) |
| `docker compose down -v` | ⚠️ borra también los volúmenes: se pierden todos los runs y modelos |
| `docker compose --profile tools up -d adminer` | arranca Adminer (UI de la base de datos) en http://localhost:8081 |
| `docker compose --profile tools stop adminer` | lo para |
| `docker compose exec postgres psql -U mlflow -d mlflow` | consola SQL dentro del contenedor (`\dt`, `\d tabla`, `\q`) |
| `mlflow gc` dentro del contenedor | ⚠️ borrado definitivo de runs/experimentos en `deleted`: ver [08](08_experiment_tracking.md#m3--borrado) |

## Ver la base de datos

Solo para consultar: las tablas son de MLflow y los cambios se hacen siempre a través de MLflow.

- **Adminer** (perfil `tools`): http://localhost:8081 → Sistema **PostgreSQL**, servidor `postgres`, usuario `mlflow`,
  contraseña `POSTGRES_PASSWORD` del `.env`, base de datos `mlflow`.
- **psql:** `docker compose exec postgres psql -U mlflow -d mlflow`. Sin contraseña, porque conecta desde dentro del contenedor.

De las 65 tablas, las relevantes son `experiments`, `runs`, `params`, `metrics`, `latest_metrics`, `tags`,
`registered_models`, `model_versions` y `registered_model_aliases`. El resto son funciones de IA generativa de MLflow 3
(trazas, evaluaciones, endpoints) que no se usan.

## Decisiones

- **Volúmenes con nombre** para los datos (no bind mounts): evitan problemas de permisos y van más rápido en WSL.
  Sin volumen, las imágenes de Postgres y RustFS crean uno anónimo que se pierde con `down` + `up`.
- **Solo se publican a `127.0.0.1`** la interfaz de MLflow y la consola de RustFS. Postgres y la API S3 quedan en la red interna.
- **Versiones fijas** de todas las imágenes y paquetes, igual que en `requirements.txt`.
- **Sin política `restart`** (por defecto `no`): el stack solo funciona cuando se levanta a mano con
  `docker compose up -d`. Con `unless-stopped`, los contenedores volvían a arrancar al encender el PC. En un servidor
  sí se usaría `unless-stopped` para que el servicio sobreviva a reinicios.
- **Herramientas de apoyo en el perfil `tools`** (Adminer): no forman parte de la plataforma y no arrancan con un `up` normal.
  Adminer entra por la red interna, así que Postgres sigue sin puerto publicado.
- **El bucket se crea a mano** con un script, no como servicio del compose: es un paso único de preparación.
  El script usa `aws-cli` en un contenedor dentro de la red de compose, así que no hay que instalar nada en el host.
- **`MLFLOW_SERVER_ALLOWED_HOSTS`** incluye `mlflow:*` para que otros contenedores (el futuro puller) puedan llamar a
  `http://mlflow:5000`. Por defecto MLflow 3 solo acepta `localhost` e IPs privadas (protección frente a DNS rebinding).

## Equivalentes en la nube

| local | AWS | Azure | GCP |
|---|---|---|---|
| MLflow | SageMaker AI con MLflow gestionado | Azure Machine Learning (compatible con MLflow) | Vertex AI Experiments + Model Registry |
| Postgres | RDS / Aurora | Azure Database for PostgreSQL | Cloud SQL |
| RustFS | S3 | Blob Storage | Cloud Storage |
| imagen propia | ECR | ACR | Artifact Registry |

El código cliente (`mlflow.log_*`, `mlflow.register_model`) es el mismo en todos: solo cambia `MLFLOW_TRACKING_URI`.

## Incidencias

### MinIO ya no se puede descargar

- **Síntoma:** `docker pull minio/minio` → `repository does not exist`, y `quay.io/minio/minio` → `unauthorized`.
- **Causa:** en 2025 MinIO dejó de publicar imágenes y binarios de la edición gratuita.
- **Solución:** **RustFS**, que implementa la misma API S3 (Apache 2.0, consola web, un contenedor). MLflow no nota la
  diferencia: solo cambia `MLFLOW_S3_ENDPOINT_URL`. Alternativas evaluadas: Garage (estable, pero sin consola y con
  configuración de clúster) y SeaweedFS (demasiado complejo para esto).

### MLflow en bucle de reinicios: `No module named 'psycopg'`

- **Causa:** la imagen de MLflow 3.17 trae SQLAlchemy 2.1, que para `postgresql://` usa por defecto el driver
  `psycopg` (v3), no `psycopg2`.
- **Solución:** instalar `psycopg[binary]` en el Dockerfile (alternativa: URL `postgresql+psycopg2://`).

### MLflow ocupa ~1,9 GB de RAM

- Con `--workers=2`, frente a ~130 MB de RustFS y ~70 MB de Postgres. Con ~7,7 GiB en WSL, conviene parar el stack
  (`docker compose stop`) mientras se entrena EfficientNetB4, o bajar a `--workers=1`.

## Siguientes pasos

1. ~~Prueba de extremo a extremo desde el host~~ → hecha: [08_experiment_tracking.md](08_experiment_tracking.md) (M.2–M.3).
2. ~~Integrar `train.py` con MLflow~~ → hecho: [08_experiment_tracking.md](08_experiment_tracking.md) (M.4).
   Cada run registra una versión de `fire-detector` sin alias.
3. Importar `mobilenetv3large_clahe` y `efficientnetb4_clahe` sin reentrenar.
4. Alias y `promote.py` (gate de promoción).
