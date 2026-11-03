# 07 — Tracking de experimentos con MLflow

Cada entrenamiento es un **run** de MLflow en el experimento `fire-detection`: guarda la config, las curvas por
epoch, las métricas finales, los artefactos, el commit y el hash de los datos, y registra una versión nueva del modelo
`fire-detector`. Toda la integración está en [`src/train/tracking.py`](../src/train/tracking.py).

Infraestructura: [05_platform.md](05_platform.md). Promoción: [08_model_registry.md](08_model_registry.md). Cómo
funciona MLflow por dentro (cliente/servidor, tablas, artefactos, alias): [learning/mlflow.md](learning/mlflow.md).

## Configuración

| elemento | valor | motivo |
|---|---|---|
| cliente | `mlflow-skinny==3.17.2` en el `.venv`, misma versión que el servidor | solo el cliente (19 paquetes frente a 38); la UI y `mlflow gc` se usan desde el contenedor |
| servidor | `--serve-artifacts`, `--workers=1`, `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` | proxy de ficheros (el cliente no necesita credenciales de S3); un solo usuario; sin los *jobs* de IA generativa, que ocupaban 1,65 GB de RAM (queda en ~390 MB) |
| experimento | `fire-detection` | los 4 experimentos son comparables (misma tarea y test) |
| modelo registrado | `fire-detector` | — |

Variables de entorno del cliente (`tracking.py` las fija si no están definidas):

| variable | valor | por qué |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | sin ella, MLflow guarda en local sin avisar |
| `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD` / `_UPLOAD` | `false` | si no, el cliente descarga directamente de S3 con URLs presigned a `http://rustfs:9000`, que no resuelve desde el host, y las descargas se quedan colgadas |
| `MLFLOW_DISABLE_TELEMETRY` | `true` | MLflow 3 envía datos de uso anónimos (también desactivado en el servidor) |
| `MLFLOW_DISABLE_AGENT_HINT` | `1` | silencia un aviso de IA generativa al importar |

**UI:** http://localhost:5000. Abre en modo *GenAI*; este proyecto usa el modo *Machine learning*
(`/#/experiments/<id>/runs`).

## Qué registra cada run

`train.py` entrena dentro de `with start_run(...)`: el run acaba `FINISHED`, o `FAILED` si algo falla.
`results/<exp>/` se sigue escribiendo igual.

| qué | cómo | por qué |
|---|---|---|
| nombre | el de la config | localizarlo en la UI |
| params | la config JSON aplanada (`phase1.lr`, `data.image_root`…) + `n_train/val/test` | la config define el experimento |
| métricas por epoch | callback `EpochLogger`, step global 1…N | `autolog` no sirve con dos fases: registra `epochs` en cada `fit()` (los params son inmutables) y cada fase empezaría en 0 |
| métricas finales | `metrics.json` aplanado (`test.recall`, `test.confusion_matrix.fp`, `test_subtype_alarm_rate.smoke_only`…) | ordenar y comparar runs por cualquiera |
| tags | `model`, `dataset`, `image_size`, `debug`, `git_dirty`, `data.source`, `data.source_md5` + `mlflow.source.git.commit` (automático) | filtrar en la UI y trazabilidad |
| repo sucio | aviso + `git_dirty=true` + `git_diff.patch` | el commit registrado es el HEAD, no el código ejecutado |
| datos | hash DVC de `original_dataset` (tag) + su `.dvc` como artefacto | ver abajo |
| artefactos | `config.json`, `history.csv`, `metrics.json`, `meta.json`, `plots/`, `predictions/`, `data/` | lo mismo que `results/<exp>/` |
| modelo | `mlflow.keras.log_model` con signature, `meta.json` como metadata y versiones de TF/Keras | cargable con `models:/fire-detector/<v>` |
| registro | cada run → nueva versión de `fire-detector`, **sin alias**; los runs con `debug` no se registran | los alias los mueve `promote.py` tras el gate |

Si el servidor no responde, `train.py` se detiene antes de entrenar, para no perder un entrenamiento sin registrar.

**Datos.** `train.py` lee `data/<dataset>_<size>`, que no está en DVC. Se registra el hash de `original_dataset`
porque el resto se deduce de él: las configs de preproceso están en el commit, y `dataset` e `image_size` son params.
Para reproducir los datos de un run: `git checkout <commit>` → `dvc pull` → scripts de preproceso.

**`meta.json`: lo que la inferencia necesita además de los pesos.** Va en la `metadata` del `MLmodel`, así que viaja
con el modelo:

```json
{
  "threshold": 0.8228, "target_recall": 0.95, "image_size": 380, "dataset": "clahe",
  "preprocessing": {
    "clahe":  {"clip_limit": 2.0, "tile_grid": 8, "channel": "L of CIELAB", "applied": "at full resolution, before resizing"},
    "resize": {"size": 380, "interpolation": "cv2.INTER_AREA", "keep_aspect_ratio": false},
    "input":  "RGB, float32 in 0-255 (the model normalizes internally)"
  },
  "output": "probability of alarm (fire and/or smoke); alarm if prob >= threshold"
}
```

En los datasets `original`, `clahe` es `null`. Al exportar para servir, `export.py` añade `jpeg_quality` y cambia
`input` por la entrada de bytes JPEG ([09](09_serving.md)). Signature del modelo registrado: entrada `image` float32
`[-1, size, size, 3]`, salida `alarm` float32 `[-1, 1]`.

## Importar entrenamientos anteriores

[`src/train/import_results.py`](../src/train/import_results.py) convierte carpetas `results/<exp>/` de entrenamientos
hechos sin MLflow en runs con la misma estructura (reutiliza `tracking.py`):

```bash
.venv/bin/python src/train/import_results.py results/mobilenetv3large_clahe results/efficientnetb4_clahe
```

- Tags `imported=true`, `imported_from=results/<exp>` y `mlflow.source.git.commit=unknown` (el código no se conoce).
- Fechas reales: fin = fecha de `model.keras`, inicio = fin − `train_minutes`.
- Idempotente: si ya existe un run con el mismo `imported_from`, lo salta.

## Runs registrados

| versión | run | origen | commit |
|---|---|---|---|
| v1 | `mobilenetv3large_original` | `train.py` | `3d2770c`, `git_dirty=False` |
| v2 | `mobilenetv3large_clahe` | importado | `unknown` |
| v3 | `efficientnetb4_clahe` | importado | `unknown` |

Métricas y comparación: [04_results.md](04_results.md).

## Operaciones

**Borrar un run.** El borrado desde la UI es lógico (pasa a `deleted`); el definitivo es `mlflow gc`. El run y su
modelo son entidades distintas: hay que borrar los dos.

```bash
.venv/bin/python -c "from mlflow import MlflowClient; MlflowClient('http://localhost:5000').delete_logged_model('m-<id>')"
cd deploy/mlops
docker compose exec -T mlflow sh -c '
  URI=$(tr "\0" "\n" < /proc/1/cmdline | sed -n "s/^--backend-store-uri=//p")
  MLFLOW_TRACKING_URI=http://localhost:5000 mlflow gc --backend-store-uri "$URI" \
    --run-ids <run_id> --logged-model-ids m-<id>
'
```

`gc` necesita la URI de Postgres con la contraseña: se lee de los argumentos del servidor (`/proc/1/cmdline`), así no
sale del contenedor. Para un experimento entero: `--experiment-ids <id>`. Nunca se borran tablas ni objetos del bucket
a mano.

**Cerrar un run interrumpido.** Un proceso matado (falta de RAM, `kill`) deja el run en `RUNNING`:

```bash
.venv/bin/python -c "from mlflow import MlflowClient; MlflowClient('http://localhost:5000').set_terminated('<run_id>', status='KILLED')"
```

## Limitaciones conocidas

- **La fase no aparece en las gráficas:** `phase` y `learning_rate` no se registran en todos los epochs.
- **`model.keras` incluye el estado del optimizador** (~75 % del tamaño, ver [03](03_training.md)); no hace falta
  para inferencia.
