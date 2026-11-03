# 05 — Plataforma MLOps (tracking y registro de modelos)

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
| `mlflow gc` dentro del contenedor | ⚠️ borrado definitivo de runs/experimentos en `deleted`: ver [07](07_experiment_tracking.md#operaciones) |

## Ver la base de datos

Solo para consultar: las tablas son de MLflow y los cambios se hacen siempre a través de MLflow.

- **Adminer** (perfil `tools`): http://localhost:8081 → Sistema **PostgreSQL**, servidor `postgres`, usuario `mlflow`,
  contraseña `POSTGRES_PASSWORD` del `.env`, base de datos `mlflow`.
- **psql:** `docker compose exec postgres psql -U mlflow -d mlflow`. Sin contraseña, porque conecta desde dentro del contenedor.

De las 65 tablas, las relevantes son `experiments`, `runs`, `params`, `metrics`, `latest_metrics`, `tags`,
`registered_models`, `model_versions` y `registered_model_aliases`. El resto son funciones de IA generativa de MLflow 3
(trazas, evaluaciones, endpoints) que no se usan.
