# 02 — Preprocesado

Código en [src/preprocess/](../src/preprocess/). Hay tres scripts, que se ejecutan en este orden:

```bash
python src/preprocess/prepare_labels.py      # etiquetas de clasificación
python src/preprocess/clahe_preprocess.py    # copia del dataset con CLAHE
python src/preprocess/resize_dataset.py      # copias redimensionadas a 224 y 380
```

Los dos últimos leen su configuración de un JSON ([clahe_config.json](../src/preprocess/clahe_config.json),
[resize_config.json](../src/preprocess/resize_config.json)), se paralelizan con `ProcessPoolExecutor` y por defecto
no reprocesan las imágenes que ya existen (`overwrite: false`).

---

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
- En el entrenamiento se usa la etiqueta binaria `alarm = label > 0`. El subtipo
  (`background`, `smoke_only`, `fire_only`, `fire_and_smoke`) se reconstruye a partir de `n_smoke` y `n_fire` para la evaluación por subtipo.
- Falla de forma explícita si encuentra un id de clase desconocido o una línea mal formada.

## 2. CLAHE — `clahe_preprocess.py`

Genera `data/clahe_dataset/` con la misma estructura que el original. Las etiquetas se copian sin cambios porque la geometría no varía.

**Cómo se aplica:** BGR → CIELAB, CLAHE **solo sobre el canal L** (luminosidad) y vuelta a BGR. Los canales de color
`a` y `b` no se tocan, así que **se conserva el tono del fuego y del humo**.

| parámetro | valor |
|---|---|
| `clip_limit` | 2.0 |
| `tile_grid` | 8 × 8 |
| `jpeg_quality` | 95 |

**Motivación:** mejorar el contraste local en imágenes oscuras, nocturnas o con niebla, donde el humo tiene poco contraste con el fondo.

**Análisis visual:** [clahe_pre_example.ipynb](../src/preprocess/clahe_pre_example.ipynb). Sobre 5 imágenes aleatorias:

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
el detalle pero también el ruido y los artefactos JPEG en zonas lisas como el cielo o el humo (SSIM de 0,73 y 0,66), por eso se elige 2.

Si CLAHE mejora o no el modelo se decide empíricamente, entrenando cada modelo con `original` y con `clahe`
(ver [04_results.md](04_results.md)).

## 3. Redimensionado — `resize_dataset.py`

Genera `data/{original,clahe}_dataset_{224,380}/`.

**Motivación:** decodificar imágenes de 1280×720 en cada epoch convertía la CPU en el cuello de botella y hacía que
el pipeline `tf.data` usara varios GB de RAM. Redimensionar una sola vez a disco elimina los dos problemas.

- Se redimensiona a `size × size` **sin mantener la relación de aspecto**, igual que `tf.image.resize` en el entrenamiento,
  así que es equivalente a entrenar con las imágenes originales.
- Interpolación `INTER_AREA` (antialiasing al reducir).
- 224 px para MobileNetV3, EfficientNetB0 y ConvNeXt-Tiny. 380 px para EfficientNetB4 (su resolución nativa).
- Las etiquetas YOLO se copian tal cual porque usan coordenadas normalizadas.

| dataset | tamaño en disco |
|---|---:|
| `original_dataset_380` | 1,1 GB |
| `clahe_dataset_380` | 1,4 GB |

## ⚠ Implicación para el despliegue

El modelo **no** incluye ni el CLAHE ni el resize. En inferencia hay que aplicar exactamente la misma secuencia:
imagen → (CLAHE con `clip_limit = 2`, `tile_grid = 8`, solo en los modelos `_clahe`) → resize a 224/380 sin mantener el aspecto.
En el entrenamiento el CLAHE se aplicó **a resolución completa y antes del resize**, y en inferencia debe hacerse en el
mismo orden. Ver [05_deployment.md](05_deployment.md).
