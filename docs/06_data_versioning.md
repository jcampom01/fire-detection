# 06 — Versionado de datos con DVC

El dataset de origen, `data/original_dataset`, se versiona con **DVC**. Los ficheros se guardan en el bucket
`datasets` de RustFS, y git solo guarda `data/original_dataset.dvc`, una referencia de 5 líneas con el hash del
dataset. Así cada commit fija también la versión de los datos, sin meter 3 GB de imágenes en git.

Cómo funciona DVC por dentro (hash, caché, índices, organización del bucket): [learning/dvc.md](learning/dvc.md).

## Qué está versionado

| ruta | versionado | cómo se obtiene |
|---|---|---|
| `data/original_dataset/` | **sí** (DVC) | `dvc pull` |
| `data/classification_labels/`, `data/clahe_dataset/`, `data/*_224`, `data/*_380` | no | se regeneran con los scripts de `src/preprocess/` ([02](02_preprocessing.md)) |

Solo se versiona la fuente de verdad: los derivados se reconstruyen con el código, que está en git.

| `original_dataset` | |
|---|---|
| hash (índice) | `fd2b80293db2475ac9c5e6e6186c395a.dir` |
| ficheros | 43.054 (21.527 imágenes + 21.527 etiquetas YOLO), 3,1 GB |
| objetos en el almacén | 33.208 (los 9.838 `.txt` vacíos de las imágenes de fondo son un único objeto) |

**Trazabilidad:** cada run de MLflow guarda el tag `data.source_md5` y el `.dvc` como artefacto
([07](07_experiment_tracking.md)). Con ese hash y el commit del run se recuperan los datos exactos del entrenamiento.

## Configuración

| elemento | valor | motivo |
|---|---|---|
| versión | `dvc[s3]==3.67.1` en el `.venv` | herramienta local, como git; `[s3]` añade el conector S3 |
| remote | `storage` → `s3://datasets/fire-detection` | nombre genérico: si se cambia RustFS por S3, solo cambia la URL |
| endpoint | `http://localhost:9000`, región `us-east-1` | RustFS en lugar de Amazon; S3 exige región y RustFS la ignora |
| bucket | `datasets`, separado de `mlflow` | permisos y ciclos de vida distintos: los artefactos se limpian, los datasets se conservan |
| puerto 9000 | publicado en `127.0.0.1` | DVC es un cliente S3 directo desde el host (MLflow, en cambio, hace de proxy) |
| caché | `copy` (por defecto) | duplica espacio, pero los ficheros de `data/` se pueden editar directamente |
| analytics | desactivados | — |

| fichero | contenido | ¿git? |
|---|---|---|
| `.dvc/config` | remote y opciones | sí |
| `.dvc/config.local` | credenciales del remote (`chmod 600`) | **no** |
| `.dvc/cache/` | almacén local por hash | no |
| `data/original_dataset.dvc` | hash del dataset | sí |
| `data/.gitignore` | excluye `/original_dataset` de git | sí |

## Puesta en marcha

Con el stack `deploy/mlops` levantado ([05](05_platform.md)):

```bash
deploy/mlops/scripts/create_bucket.sh datasets          # una vez por instalación de RustFS

# Credenciales: las de RustFS en deploy/mlops/.env
dvc remote modify --local storage access_key_id <RUSTFS_ACCESS_KEY>
dvc remote modify --local storage secret_access_key <RUSTFS_SECRET_KEY>
chmod 600 .dvc/config.local

dvc ls-url remote://storage/      # comprueba conexión y credenciales
dvc pull                          # ~1 min 20 s
```

Con el bucket vacío, `dvc ls-url` responde `does not exist`: la conexión funciona, pero no hay datos. En ese caso hay
que descargar el dataset de [Kaggle](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)
([01](01_eda.md#dataset)) en `data/original_dataset/` y subirlo:

```bash
dvc add data/original_dataset     # llena la caché; si el .dvc cambia, la descarga no es idéntica a la versionada
dvc push
```

## Uso

**Cambiar los datos:**

```bash
dvc add data/original_dataset          # nuevo hash → nuevo .dvc (~45 s)
git add data/original_dataset.dvc
git commit -m "..."
dvc push                               # sube los ficheros nuevos al bucket
git push
```

**Volver a una versión anterior:**

```bash
git checkout <commit>     # trae el .dvc de esa versión
dvc checkout              # pone en data/ los ficheros correspondientes (si faltan en la caché: dvc pull)
```

**Comprobar el estado:** `dvc status` (¿coincide `data/` con el `.dvc`?) y `dvc status -c` (¿falta algo por subir?).

git y DVC no están conectados: `git commit` no sube datos y `git checkout` no cambia las imágenes del disco.

| olvido | consecuencia |
|---|---|
| `git push` sin `dvc push` | el `.dvc` está en GitHub pero sus datos no están en el bucket: `dvc pull` falla en otra máquina |
| `git checkout` sin `dvc checkout` | código de una versión con datos de otra |

**Nunca** `dvc gc --cloud` sin `--all-commits`, ni borrar a mano en el bucket `datasets`: son las únicas formas de
perder versiones anteriores.

## Comandos

| comando | qué hace |
|---|---|
| `dvc add <ruta>` | calcula hashes, llena la caché y actualiza el `.dvc` |
| `dvc status` / `dvc status -c` | ¿difiere `data/` del `.dvc`? / ¿difiere la caché del remote? |
| `dvc push` / `dvc pull` | caché → remote / remote → caché → `data/` |
| `dvc checkout` | caché → `data/`, según el `.dvc` actual |
| `dvc ls-url remote://storage/` | lista el remote |
| `dvc gc` | ⚠️ borra de la caché lo no referenciado (con `--cloud`, también del remote) |
