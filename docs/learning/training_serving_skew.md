# Training-serving skew: un caso real

El modelo es sensible a diferencias de píxeles invisibles, así que la API tiene que preparar la imagen **exactamente**
como se preparó el dataset.

**Síntoma.** Con la preparación evidente (CLAHE → resize en memoria → píxeles a TF Serving), las predicciones de fotos
originales diferían del entrenamiento hasta 0,15, y 14 de 228 imágenes cambiaban de decisión. Por ejemplo,
`PublicDataset01134` (humo) pasaba de 0,8235 a 0,7839 y perdía la alarma. Enviando los ficheros del propio dataset, la
diferencia máxima era 0,005: el problema estaba en la preparación, no en el modelo ni en TF Serving.

**Causas** (medidas sobre 200–228 imágenes):

| causa | efecto en los píxeles |
|---|---|
| **JPEG intermedios:** los scripts del dataset guardan un JPEG de calidad 95 tras el CLAHE y otro tras el resize; el modelo se entrenó con esos artefactos de compresión | diferencia media 1,54; al repetir los dos JPEG, 0,00 |
| **Decodificador:** TensorFlow descomprime JPEG por defecto con una DCT rápida y aproximada; OpenCV, con la exacta | diferencia media 1,52; con `dct_method="INTEGER_ACCURATE"`, 0,000 |

Cada una mueve los píxeles ~1,5 niveles de gris (invisible) y la probabilidad hasta 0,15–0,19. No hay errores: el
umbral, calibrado con la preparación del entrenamiento, deja de estar calibrado.

**Solución:**

- **Una sola implementación de CLAHE:** [`src/preprocess/clahe.py`](../../src/preprocess/clahe.py) (solo OpenCV), que
  usan los scripts del dataset y la API. Copiarla habría permitido que las dos versiones divergieran sin que nada
  fallara.
- La API reproduce la cadena del dataset en `prepare()`: CLAHE → **JPEG** → resize → **JPEG**, con la calidad de
  `meta.json` (`jpeg_quality`).
- Envía **los bytes del JPEG**, y el modelo los decodifica **dentro del grafo** con TensorFlow, como `data.py`.

| 228 fotos originales vs entrenamiento | píxeles en memoria | bytes JPEG (actual) |
|---|---:|---:|
| diferencia máxima | 0,1458 | 0,0107 |
| media | 0,0116 | **0,0004** |
| p95 | 0,0630 | **0,0021** |
| decisiones que cambian | 14 | **2** (a menos de 0,001 del umbral) |
| tamaño de la petición | ~1.500 KB | **85 KB** |

Lo que queda es ruido numérico: la CPU calcula float16 de forma distinta a la GPU. Es inevitable con cualquier umbral:
en el test solo 4 de 4.306 imágenes están a menos de 0,004 de él.
