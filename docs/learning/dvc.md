# DVC en detalle

Cómo funciona DVC por dentro, con ejemplos de este proyecto. Uso y configuración: [06_data_versioning.md](../06_data_versioning.md).

## Por qué

Un modelo es el resultado de **código + config + entorno + datos**. Código y config están en git y el entorno en
`requirements-lock.txt`. Sin versionar los datos:

- no se sabe con qué versión de los datos se entrenó cada modelo;
- si se corrigen etiquetas o se borran imágenes, la versión anterior se pierde;
- otra máquina (servidor con GPU, CI) no tiene de dónde sacarlos;
- si se rompe el disco, se pierden.

git no sirve para esto: guarda todas las versiones para siempre y cada clon descarga el historial completo. Con 3 GB
de imágenes, el repo crecería sin control.

## Conceptos

### Hash

Un hash es un resumen de un fichero calculado a partir de **todos sus bytes**. DVC usa MD5 (32 caracteres).

- Mismo contenido → mismo hash, siempre. El nombre del fichero no influye.
- Cualquier cambio, aunque sea de un byte → hash completamente distinto.

| fichero | MD5 |
|---|---|
| `test/images/AoF06723.jpg` | `eacf88fc8cfcc62241c5f199d9f3bb3d` |
| la misma imagen con otro nombre | `eacf88fc8cfcc62241c5f199d9f3bb3d` (idéntico) |
| la misma imagen con 1 byte cambiado de 227.556 | `f7ffc9ada558efbedb9dc2b4f55c0c3a` (distinto) |

### Almacenamiento por contenido

DVC guarda cada fichero **con su hash como nombre**:

```
data/original_dataset/test/images/AoF06723.jpg  ──►  .dvc/cache/files/md5/ea/cf88fc8cfcc62241c5f199d9f3bb3d
```

- Los dos primeros caracteres del hash (`ea/`) son una subcarpeta, para repartir los ficheros en 256 carpetas.
- **El contenido no se transforma:** ni se comprime, ni se cifra. Es el mismo JPEG, byte a byte, con otro nombre.
- `dvc pull` hace lo contrario: copia el fichero y le devuelve su nombre original.

Consecuencias:

- **Deduplicación:** un contenido que aparece varias veces se guarda una sola vez.
- **Las versiones no duplican espacio:** una imagen que no cambia entre versiones es el mismo objeto.
- **Nunca se sobrescribe nada:** un fichero modificado tiene otro hash y es otro objeto, así que el original sigue intacto.
- **Integridad:** al descargar se recalcula el hash; si no coincide, el fichero está corrupto.

### Índice (`.dir`) y ticket (`.dvc`)

Como en el almacén se pierden los nombres, para una carpeta DVC guarda también un **índice**: un JSON con una entrada
`{md5, relpath}` por fichero. El índice tiene su propio hash y se guarda como un objeto más, con extensión `.dir`.

El **ticket** `data/original_dataset.dvc` es lo único que va a git, y contiene **solo el hash del índice**:

```yaml
outs:
- md5: fd2b80293db2475ac9c5e6e6186c395a.dir   # hash del índice
  size: 3118334084                            # informativo
  nfiles: 43054                               # informativo
  hash: md5
  path: original_dataset
```

Es una cadena de tres niveles (un árbol de Merkle, como git internamente):

```
git                         caché / bucket
───                         ──────────────
original_dataset.dvc        índice fd/2b8029….dir                imágenes y etiquetas
  md5: fd2b8029….dir  ───►    test/images/AoF06723.jpg → eacf…  ───►  ea/cf88fc…
  (5 líneas)                  … 43.054 entradas (3,6 MB)              … 33.208 objetos
```

Si cambia una sola imagen, cambia su hash, una línea del índice, el hash del índice y el `.dvc`. git ve un cambio de
una línea aunque detrás haya 43.054 ficheros.

### Los cuatro sitios

```
 MÁQUINA LOCAL                                                 FUERA
 ① data/original_dataset/   copia de trabajo (la que leen los scripts)
          │ dvc add
          ▼
 ② .dvc/cache/              almacén local, por hash     ── dvc push ──►  ④ bucket "datasets" (RustFS)
                                                        ◄── dvc pull ──     copia remota del almacén
 ③ data/original_dataset.dvc  ticket                    ── git commit/push ──►  GitHub
```

| sitio | qué es | quién lo gestiona |
|---|---|---|
| ① `data/original_dataset/` | copia de trabajo, ficheros normales | el usuario: se puede modificar o borrar |
| ② `.dvc/cache/` | todas las versiones añadidas, por hash | DVC |
| ③ `.dvc` | qué versión corresponde a cada commit | git |
| ④ bucket | copia remota de la caché: segura y compartible | DVC (`push` / `pull`) |

Modificar ① no afecta a nada hasta que se hace `dvc add`, como editar un fichero en git antes de `add` + `commit`.

### git, DVC y MLflow

| pieza | guarda | para qué |
|---|---|---|
| **bucket** | el contenido de todas las versiones, por hash, y los índices | almacenamiento seguro y compartible |
| **git** | el `.dvc` vigente en cada commit, junto al código | saber qué versiones existen y recuperar cualquiera |
| **MLflow** | en cada run: el hash de los datos y el commit de git | saber con qué datos y código se entrenó cada modelo |

- Sin git, cada `dvc add` sobrescribiría el ticket: los datos antiguos seguirían en el bucket, pero no se sabría qué
  hash corresponde a cada versión.
- Cada commit es una foto coherente del código y de todos los `.dvc`. Para nombrar una versión de datos:
  `git tag data-v1`.

**Son herramientas independientes:** `git commit` no sube datos, `dvc push` no hace commit y `git checkout` no cambia
las imágenes del disco. `dvc install` instalaría git hooks que lo automatizan; en este proyecto no se usan.

## Organización del bucket

```
s3://datasets/
└── fire-detection/              prefijo del proyecto (url del remote)
    └── files/md5/
        ├── 00/ … ff/            256 subcarpetas: dos primeros caracteres del hash
        │   └── <resto del hash> cada objeto: imagen o etiqueta, sin extensión
        └── fd/2b8029….dir       índices de carpeta
```

No hay carpetas `v1/`, `v2/`: las versiones existen solo a través de los índices, y los índices a través de los
tickets de git. El bucket no es legible para personas; los datos se ven en `data/` tras un `dvc pull`. En S3 las
carpetas no existen realmente: el nombre de un objeto incluye las barras.
