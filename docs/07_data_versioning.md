# 07 — Versionado de datos con DVC

> Estado: DVC inicializado, remote `storage` configurado (RustFS, bucket `datasets`) y `data/original_dataset`
> versionado y recuperación comprobada con `dvc pull`. Pendiente: el pipeline de preprocesado (`dvc.yaml`) y
> registrar en MLflow la versión de datos de cada run.

## Por qué

Un modelo es el resultado de **código + config + entorno + datos**. Código y config están en git y el entorno en
`requirements-lock.txt`, pero los datos solo existían en `data/`:

- no había forma de saber con qué versión de los datos se entrenó cada modelo;
- si se corrigen etiquetas o se borran imágenes, la versión anterior se pierde;
- otra máquina (servidor con GPU, CI) no tiene de dónde sacarlos;
- si se rompe el disco, se pierden.

git no sirve para esto: guarda todas las versiones para siempre y cada clon descarga el historial completo. Con 3 GB
de imágenes el repo crecería sin control. **DVC (Data Version Control)** saca los ficheros pesados de git y deja en su
lugar una referencia pequeña.

## Conceptos

### Hash (huella)

Un hash es un resumen de un fichero calculado a partir de **todos sus bytes**. DVC usa MD5 (32 caracteres).

- Mismo contenido → mismo hash, siempre. El nombre del fichero no influye.
- Cualquier cambio, aunque sea de un byte → hash completamente distinto.

Comprobado con una imagen del dataset:

| fichero | MD5 |
|---|---|
| `test/images/AoF06723.jpg` | `eacf88fc8cfcc62241c5f199d9f3bb3d` |
| la misma imagen con otro nombre | `eacf88fc8cfcc62241c5f199d9f3bb3d` (idéntico) |
| la misma imagen con 1 byte cambiado de 227.556 | `f7ffc9ada558efbedb9dc2b4f55c0c3a` (distinto) |

### Almacenamiento por contenido

DVC guarda cada fichero **con su hash como nombre**, no con su nombre original:

```
data/original_dataset/test/images/AoF06723.jpg  ──►  .dvc/cache/files/md5/ea/cf88fc8cfcc62241c5f199d9f3bb3d
```

- Los dos primeros caracteres del hash (`ea/`) son una subcarpeta, para repartir los ficheros en 256 carpetas.
- **El contenido no se transforma**: ni se comprime, ni se cifra, ni se convierte. Es el mismo JPEG, byte a byte, con
  otro nombre (`file` lo reconoce como `JPEG image data, 1280x720`). Solo cambia el nombre.
- `dvc pull` hace lo contrario: copia el fichero y le devuelve su nombre original.

Consecuencias:

- **Deduplicación:** un contenido que aparece varias veces se guarda una sola vez.
- **Las versiones no duplican espacio:** una imagen que no cambia entre versiones es el mismo objeto.
- **Nunca se sobrescribe nada:** un fichero modificado tiene otro hash y es otro objeto, así que el original sigue intacto.
- **Integridad:** al descargar se recalcula el hash, y si no coincide el fichero está corrupto.

### Índice (`.dir`) y ticket (`.dvc`)

Como en el almacén se pierden los nombres, para una carpeta DVC guarda también un **índice**: un JSON con una entrada
`{md5, relpath}` por fichero. El índice tiene a su vez su propio hash y se guarda como un objeto más, con extensión `.dir`.

El **ticket** `data/original_dataset.dvc` es lo único que va a git, y contiene **solo el hash del índice**:

```yaml
outs:
- md5: fd2b80293db2475ac9c5e6e6186c395a.dir   # hash del índice
  size: 3118334084                            # informativo
  nfiles: 43054                               # informativo
  hash: md5
  path: original_dataset
```

Es una cadena de tres niveles ("huellas de huellas", un árbol de Merkle, igual que git internamente):

```
git                         caché / bucket
───                         ──────────────
original_dataset.dvc        índice fd/2b8029….dir                imágenes y etiquetas
  md5: fd2b8029….dir  ───►    test/images/AoF06723.jpg → eacf…  ───►  ea/cf88fc…
  (5 líneas)                  … 43.054 entradas (3,6 MB)              … 33.208 objetos
```

Si cambia una sola imagen, cambia su hash, cambia una línea del índice, cambia el hash del índice y cambia el `.dvc`.
git ve un cambio de una línea, aunque detrás haya 43.054 ficheros.

### Los cuatro sitios

```
 TU MÁQUINA                                                    FUERA
 ① data/original_dataset/   copia de trabajo (la que lee train.py)
          │ dvc add
          ▼
 ② .dvc/cache/              almacén local, por hash     ── dvc push ──►  ④ bucket "datasets" (RustFS)
                                                        ◄── dvc pull ──     copia remota del almacén
 ③ data/original_dataset.dvc  ticket                    ── git commit/push ──►  GitHub
```

| sitio | qué es | quién lo gestiona |
|---|---|---|
| ① `data/original_dataset/` | copia de trabajo, ficheros normales | tú: se puede modificar o borrar libremente |
| ② `.dvc/cache/` | todas las versiones añadidas, por hash | DVC |
| ③ `.dvc` | qué versión toca | git |
| ④ bucket | copia remota de la caché: segura y compartible | DVC (`push` / `pull`) |

Modificar ① no afecta a nada hasta que se hace `dvc add` (como editar un fichero en git antes de `add` + `commit`).
`dvc status` muestra si ① difiere del ticket y `dvc checkout` la devuelve a lo que dice el ticket.

### git, DVC y MLflow

| pieza | guarda | para qué |
|---|---|---|
| **bucket** | el contenido de todas las versiones, por hash, + los índices | almacenamiento seguro y compartible |
| **git** | el `.dvc` vigente en cada commit, junto al código; el historial conserva todos | saber qué versiones existen y recuperar cualquiera |
| **MLflow** | en cada run: el `.dvc` / hash de los datos + el commit de git | saber con qué datos y código se entrenó cada modelo |

- Sin git, cada `dvc add` sobrescribiría el ticket: los datos antiguos seguirían en el bucket, pero no se sabría qué
  hash corresponde a cada versión. `dvc init --no-scm` existe, pero se pierde el historial.
- Un repo puede tener muchos `.dvc` (uno por carpeta versionada) y un `dvc.lock` para las salidas de pipelines.
  Cada commit es una foto coherente de todos ellos. Para nombrar versiones: `git tag data-v1`.

### git y DVC no están conectados

Son herramientas independientes: `git commit` no sube datos, `dvc push` no hace commit y `git checkout` no cambia las
imágenes del disco. El flujo normal tras cambiar datos:

```bash
dvc add data/original_dataset          # nuevo hash → nuevo ticket
git add data/original_dataset.dvc
git commit -m "..."
dvc push                               # sube los objetos nuevos
git push
```

Para volver a una versión anterior:

```bash
git checkout <commit>     # trae el ticket antiguo
dvc checkout              # pone en data/ los ficheros de ese ticket (desde la caché; si faltan: dvc pull)
```

| olvido | consecuencia |
|---|---|
| `git push` sin `dvc push` | en GitHub hay un ticket cuyos datos no están en el bucket, y `dvc pull` falla en otra máquina |
| `git checkout` sin `dvc checkout` | código antiguo con datos nuevos |

`dvc install` instala git hooks que hacen `dvc checkout` tras `git checkout` y `dvc push` antes de `git push`.
De momento no se usan, para hacer cada paso a mano y entenderlo.

### Recuperar ficheros borrados

`dvc push` **solo añade** al bucket. Si se borran imágenes y se hace `dvc add`, se crea un índice nuevo sin ellas, pero
los objetos y el índice anterior siguen en el bucket. Cualquier versión cuyo ticket esté en un commit se puede recuperar.

Solo se pierden datos si:

- se ejecuta **`dvc gc --cloud`** (recogida de basura en el remote). Si se usa, debe ser con `--all-commits`;
- **se borra a mano** en el bucket (consola de RustFS, `aws s3 rm`). El bucket `datasets` es de DVC;
- **nunca se hizo `dvc push`** y se pierde el disco;
- **se pierde el commit** con el ticket (por ejemplo, sin `git push`).

### Organización del bucket

```
s3://datasets/
└── fire-detection/              prefijo del proyecto (url del remote)
    └── files/md5/
        ├── 00/ … ff/            256 subcarpetas: dos primeros caracteres del hash
        │   └── <resto del hash> cada objeto: imagen o etiqueta, sin extensión
        └── fd/2b8029….dir       índices de carpeta
```

No hay carpetas `v1/`, `v2/`: las versiones existen solo a través de los índices, y los índices a través de los tickets
de git. El bucket no es legible para personas; los datos se ven en `data/` tras un `dvc pull`. En S3 las carpetas no
existen realmente: el nombre de un objeto incluye las barras, y `fire-detection/` aparece con el primer objeto subido.

## Pasos realizados

### D.1 — Preparación

**Bucket propio.** Separado de `mlflow`: cada bucket tiene un único dueño, se pueden dar permisos distintos y tienen
ciclos de vida distintos (los artefactos se pueden limpiar; los datasets se conservan).

```bash
cd deploy/mlops
./scripts/create_bucket.sh datasets
```

**Puerto 9000 publicado en `127.0.0.1`.** MLflow hace de proxy hacia RustFS, pero DVC no: es un cliente S3 directo
que se ejecuta en el host. En `docker-compose.yml`, servicio `rustfs`:

```yaml
ports:
  - "127.0.0.1:9000:9000"  # S3 API, for DVC on the host
```

`docker compose up -d rustfs` recrea el contenedor; los datos siguen en el volumen `rustfs_data`.

**DVC en el `.venv`.** Trabaja sobre `data/` y sobre el repo, así que se usa como herramienta local, igual que git.
El extra `[s3]` instala `dvc-s3`, el conector para buckets S3.

```bash
.venv/bin/pip install "dvc[s3]==3.67.1"
```

Añadido a `requirements.txt` (sección `MLOps`) y lock regenerado. Se comprobó que no cambió la versión de ninguna
librería existente: solo se añadieron 84 paquetes (172 en total).

### D.2 — Inicialización y remote

**`dvc init`** (el equivalente de `git init`; no toca los datos):

| fichero | qué es | ¿git? |
|---|---|---|
| `.dvc/config` | configuración compartida (remote, opciones) | sí |
| `.dvc/config.local` | configuración privada: credenciales | **no** |
| `.dvc/.gitignore` | excluye `config.local`, `cache/` y `tmp/` | sí |
| `.dvc/cache/` | almacén local por hash | no |
| `.dvcignore` | ficheros que DVC debe ignorar (como `.gitignore`) | sí |

Analytics anónimos desactivados: `dvc config core.analytics false`.

**Remote `storage`.** Es solo un nombre, como `origin` en git. Es genérico a propósito: si se cambia RustFS por S3,
solo cambia la URL.

```bash
dvc remote add -d storage s3://datasets/fire-detection      # -d: remote por defecto
dvc remote modify storage endpointurl http://localhost:9000  # RustFS en lugar de Amazon
dvc remote modify storage region us-east-1                   # S3 la exige; RustFS la ignora
dvc remote modify --local storage access_key_id <RUSTFS_ACCESS_KEY>      # → config.local
dvc remote modify --local storage secret_access_key <RUSTFS_SECRET_KEY>  # → config.local
chmod 600 .dvc/config.local
```

Resultado en `.dvc/config`:

```ini
[core]
    analytics = false
    remote = storage
['remote "storage"']
    url = s3://datasets/fire-detection
    endpointurl = http://localhost:9000
    region = us-east-1
```

DVC combina `config` y `config.local` al leerlos. Las credenciales se tomaron de `deploy/mlops/.env` sin mostrarlas.

**Prueba de conexión:** `dvc ls-url remote://storage/` → `does not exist: datasets/fire-detection`. Es el resultado
esperado: conecta y se autentica, pero el prefijo aún está vacío. Un fallo habría dado `Could not connect`,
`Access Denied` o `NoSuchBucket`.

**Tipo de caché: `copy`** (el valor por defecto, no hay que configurarlo):

| `cache.type` | espacio | inconveniente |
|---|---|---|
| `copy` | duplica (caché + `data/`) | ninguno, salvo el espacio |
| `hardlink` / `symlink` | sin duplicar | ficheros de solo lectura: hay que hacer `dvc unprotect` antes de modificarlos |
| `reflink` | sin duplicar | no disponible en ext4 |

Con 835 GB libres, el espacio no es un problema. Se puede cambiar más adelante.

**Commit (manual):**

```bash
git add .dvc/config .dvc/.gitignore .dvcignore
git commit -m "Init DVC with RustFS remote"
```

Antes de hacer commit hay que comprobar que `.dvc/config.local` y `deploy/mlops/.env` **no** aparecen en `git status`.

### D.3 — Versionar el dataset original

Solo `data/original_dataset`, la fuente de verdad. Los datos derivados (`classification_labels`, `clahe_dataset`,
`*_224`, `*_380`) se generan con scripts y se versionarán como salidas de un pipeline (D.5).

```bash
dvc add data/original_dataset     # 43 s
```

Qué hizo:

1. Calculó el MD5 de los 43.054 ficheros.
2. Los copió a `.dvc/cache/files/md5/` (3,0 GB).
3. Construyó el índice (`fd/2b80293db2475ac9c5e6e6186c395a.dir`, 3,6 MB) y lo guardó en la caché.
4. Escribió el ticket `data/original_dataset.dvc`.
5. Añadió `/original_dataset` a `data/.gitignore`.

Los ficheros de `data/original_dataset/` no se movieron ni se modificaron.

**Deduplicación real:** 43.054 ficheros → 33.208 objetos.

| hash | ficheros | motivo |
|---|---:|---|
| `d41d8cd98f00b204e9800998ecf8427e` | 9.838 | `.txt` de etiquetas YOLO **vacíos**, uno por imagen de fondo (la misma cifra que en el EDA). Es el hash de un fichero vacío |
| otros 4 hashes | 8 | etiquetas de imágenes distintas que coinciden por casualidad |

**Subida y commit:**

```bash
dvc push                                         # caché → bucket datasets/fire-detection/files/md5/...
git add data/.gitignore data/original_dataset.dvc
git commit -m "Track original dataset with DVC"
```

Tras el push, el prefijo `fire-detection/files/md5/` aparece en la consola de RustFS (http://localhost:9001/rustfs/console/).

### D.4 — Prueba de recuperación

Antes de borrar se movieron la copia de trabajo y la caché a `~/containers/_dvc_test_backup/` (mismo disco, mover
es instantáneo), para poder compararlas después.

**A. Sin copia de trabajo → recuperar de la caché local (sin red):**

```bash
dvc status       # deleted: data/original_dataset
dvc checkout     # 24 s
```

**B. Sin copia de trabajo ni caché → recuperar del bucket** (simula una máquina nueva que solo tiene el repo):

```bash
dvc status -c    # todo "deleted" en local, existe en el remote
dvc pull         # 1 min 22 s: "33208 files fetched and 43054 files added"
```

`fetched` son los objetos distintos descargados a la caché; `added` son los ficheros colocados en `data/` con su
nombre original (el objeto vacío se copia 9.838 veces).

| comprobación | resultado |
|---|---|
| `dvc status` / `dvc status -c` | `up to date` / `in sync` |
| caché reconstruida | 3,0 GB, 33.209 objetos |
| `diff -rq` con el original | idénticos byte a byte |

En una máquina nueva: `git clone` → `pip install -r requirements-lock.txt` → credenciales en `.dvc/config.local`
→ `dvc pull`.

## Comandos de referencia

| comando | qué hace |
|---|---|
| `dvc add <ruta>` | calcula hashes, llena la caché y crea o actualiza el `.dvc` |
| `dvc status` | ¿difiere la copia de trabajo del ticket? |
| `dvc status -c` | ¿difiere la caché del remote? (qué falta por subir) |
| `dvc push` | caché → remote |
| `dvc pull` | remote → caché → copia de trabajo |
| `dvc checkout` | caché → copia de trabajo, según el ticket actual |
| `dvc remote list` | remotes configurados |
| `dvc gc` | ⚠️ borra de la caché lo no referenciado (con `--cloud`, también del remote) |

## Siguientes pasos

- **D.5** — Pipeline `dvc.yaml` con `prepare_labels.py`, `clahe_preprocess.py` y `resize_dataset.py`, y `dvc repro`.
- **D.6** — Cada run de MLflow registra el commit de git, el hash de los datos y el propio `.dvc` como artefacto.
