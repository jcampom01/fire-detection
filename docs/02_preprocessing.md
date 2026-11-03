# 02 — Preprocesado

Código en [src/preprocess/](../src/preprocess/). Tres scripts generan, a partir de `data/original_dataset`, las
etiquetas de clasificación y los datasets con los que se entrena:

```bash
.venv/bin/python src/preprocess/prepare_labels.py      # etiquetas de clasificación
.venv/bin/python src/preprocess/clahe_preprocess.py    # copia del dataset con CLAHE
.venv/bin/python src/preprocess/resize_dataset.py      # copias redimensionadas a 224 y 380 px
```

Los datos generados no se versionan: se regeneran con estos scripts y sus configs, que están en git
([06](06_data_versioning.md)). Los dos últimos leen un JSON ([clahe_config.json](../src/preprocess/clahe_config.json),
[resize_config.json](../src/preprocess/resize_config.json)), se paralelizan con `ProcessPoolExecutor` y por defecto
no reprocesan las imágenes que ya existen (`overwrite: false`).

## 1. Etiquetas de clasificación — `prepare_labels.py`

Convierte las etiquetas YOLO de detección en un CSV por split con una fila por imagen:
`data/classification_labels/{train,val,test}.csv`.

| columna | descripción |
|---|---|
| `image`, `image_path` | nombre y ruta relativa de la imagen |
| `label` | `0 = no_fire`, `1 = smoke` (solo humo), `2 = fire` (hay fuego, con o sin humo) |
| `label_name` | nombre de la clase |
| `n_smoke`, `n_fire` | número de cajas de cada clase |

- El fuego tiene prioridad sobre el humo: fuego + humo → `fire`.
- El entrenamiento usa la etiqueta binaria `alarm = label > 0`. El subtipo (`background`, `smoke_only`, `fire_only`,
  `fire_and_smoke`) se reconstruye con `n_smoke` y `n_fire` para la evaluación por subtipo.
- Falla de forma explícita si encuentra un id de clase desconocido o una línea mal formada.

## 2. CLAHE — `clahe_preprocess.py`

Genera `data/clahe_dataset/` con la misma estructura que el original. Las etiquetas se copian sin cambios porque la
geometría no varía.

**Cómo se aplica:** BGR → CIELAB, CLAHE **solo sobre el canal L** (luminosidad) y vuelta a BGR. Los canales de color
`a` y `b` no se tocan, así que **se conserva el tono del fuego y del humo**. La función está en
[`clahe.py`](../src/preprocess/clahe.py) y es la misma que usa la API al servir el modelo.

| parámetro | valor |
|---|---|
| `clip_limit` | 2.0 |
| `tile_grid` | 8 × 8 |
| `jpeg_quality` (al guardar) | 95 |

**Motivación:** mejorar el contraste local en imágenes oscuras, nocturnas o con niebla, donde el humo tiene poco
contraste con el fondo.

**Análisis visual:** [clahe_pre_example.ipynb](../src/preprocess/clahe_pre_example.ipynb). Sobre 5 imágenes
aleatorias:

| métrica | cambio medio |
|---|---:|
| brillo medio (L) | +26 % |
| rango dinámico | +10 % |
| entropía | +5 % |
| nitidez (varianza del Laplaciano) | +60 % |
| densidad de bordes (Canny) | +97 % |
| colorfulness | +3 % |
| saturación media | −18 % |

Con `clip_limit = 2` el SSIM respecto a la original es de ~0,83, un cambio moderado. Valores más altos (4, 8) aumentan
el detalle pero también el ruido y los artefactos JPEG en zonas lisas como el cielo o el humo (SSIM de 0,73 y 0,66).

Si CLAHE mejora el modelo se comprueba entrenando cada modelo con `original` y con `clahe`. En MobileNet no aporta
([04](04_results.md#aporta-clahe)).

## 3. Redimensionado — `resize_dataset.py`

Genera `data/{original,clahe}_dataset_{224,380}/`.

**Motivación:** decodificar imágenes de 1280×720 en cada epoch convertía la CPU en el cuello de botella y hacía que
el pipeline `tf.data` usara varios GB de RAM. Redimensionar una sola vez a disco elimina los dos problemas.

- A `size × size` **sin mantener la relación de aspecto**, con interpolación `INTER_AREA` (antialiasing al reducir),
  y guardado como JPEG de calidad 95.
- 224 px para MobileNetV3; 380 px para EfficientNetB4 (su resolución nativa).
- Las etiquetas YOLO se copian tal cual porque usan coordenadas normalizadas.
- El `tf.image.resize` de `src/train/data.py` no cambia nada, porque las imágenes ya tienen el tamaño final. Por tanto,
  **el modelo aprende con estas imágenes concretas** (interpolación de OpenCV y compresión JPEG), no con las que daría
  redimensionar en TensorFlow.

| dataset | tamaño en disco |
|---|---:|
| `original_dataset` / `clahe_dataset` (1280×720) | 3,0 GB / 4,1 GB |
| `original_dataset_224` / `clahe_dataset_224` | 479 MB / 583 MB |
| `original_dataset_380` / `clahe_dataset_380` | 1,1 GB / 1,4 GB |

## Implicación para el servicio

El modelo no incluye ni el CLAHE ni el resize, así que el servicio tiene que reproducir exactamente la misma cadena:

```
imagen → CLAHE (solo modelos _clahe, a resolución completa) → JPEG q95 → resize INTER_AREA → JPEG q95
```

Las compresiones JPEG intermedias también forman parte de lo que el modelo ha aprendido. La API lo hace así
([09](09_serving.md#preparación-fiel-al-entrenamiento)). Por qué importa y cómo se comprobó:
[learning/training_serving_skew.md](learning/training_serving_skew.md).
