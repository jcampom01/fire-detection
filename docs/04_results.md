# 04 — Resultados

Análisis en [compare_results.ipynb](../src/train/compare_results.ipynb). Datos en `results/<experimento>/metrics.json`
y en MLflow (experimento `fire-detection`, modelo registrado `fire-detector`).

**Estado:** entrenados 3 de 4 experimentos. `efficientnetb4_original` está pospuesto (~76 min de GPU y no hace
falta para responder si CLAHE aporta, ver abajo).

| versión registrada | experimento | alias |
|---|---|---|
| v1 | `mobilenetv3large_original` | — |
| v2 | `mobilenetv3large_clahe` | — |
| v3 | `efficientnetb4_clahe` | **`@production`** |

## Métricas en test (umbral elegido en val para recall ≥ 95 %)

| | MobileNetV3-L original | MobileNetV3-L CLAHE | EfficientNetB4 CLAHE |
|---|---:|---:|---:|
| umbral | 0,879 | 0,728 | 0,823 |
| recall | 92,6 % | **94,9 %** | 93,2 % |
| precisión | 98,5 % | 98,2 % | **99,2 %** |
| FPR (falsas alarmas) | 1,6 % | 2,0 % | **0,85 %** |
| FN / FP | 170 / 32 | 118 / 40 | 157 / 17 |
| F1 | 0,955 | **0,965** | 0,961 |
| ROC-AUC | 0,9929 | 0,9921 | **0,9947** |
| PR-AUC | 0,9943 | 0,9938 | **0,9959** |
| mejor `val_loss` | 0,097 | 0,107 | **0,073** |

**Con umbral 0,5** las tres tienen una accuracy del 96,6–96,7 %:

| modelo | FN | FP |
|---|---:|---:|
| MobileNet original | 72 | 73 |
| MobileNet CLAHE | 75 | 67 |
| B4 | 82 | 58 |

## Tasa de alarma por subtipo (test)

| subtipo (imágenes) | MobileNet original | MobileNet CLAHE | EfficientNetB4 |
|---|---:|---:|---:|
| background (2.005) — FPR | 1,6 % | 2,0 % | **0,8 %** |
| smoke_only (1.186) | 87,2 % | **91,1 %** | 88,3 % |
| fire_only (220) | 94,1 % | 95,0 % | **95,9 %** |
| fire_and_smoke (895) | 99,4 % | **99,8 %** | 99,0 % |

- Como se esperaba en el EDA, **smoke_only es el caso más difícil** en los tres modelos.
- Las diferencias entre columnas mezclan el modelo con su umbral (ver la sección siguiente).
- En `fire_only`, cada imagen mueve la tasa ~0,45 puntos.

## Comparación justa: el mismo punto de operación

Cada modelo tiene su propio umbral, elegido en val, y unos se trasladan a test mejor que otros. Por ejemplo, el de B4
lo deja más conservador en test (menos recall y menos falsas alarmas), y el de MobileNet original todavía más. Para
comparar los modelos y no los umbrales, se fija el mismo recall o el mismo FPR en test:

| | MobileNet original | MobileNet CLAHE | EfficientNetB4 |
|---|---:|---:|---:|
| FP con recall = 95 % | 45 | 40 | **28** |
| FP con recall = 97 % | 76 | 81 | **67** |
| recall con FPR = 1 % | 88,1 % | 88,9 % | **93,6 %** |
| recall con FPR = 2 % | 94,0 % | 95,5 % | **96,1 %** |

**B4 es mejor clasificador** en cualquier punto de la curva. La diferencia es mayor en la zona de pocas falsas
alarmas, que es la que importa en un detector de incendios.

## ¿Aporta CLAHE?

Con MobileNetV3-L, el mismo modelo entrenado con y sin CLAHE:

| prueba | resultado | lectura |
|---|---|---|
| ROC-AUC, CLAHE − original (bootstrap, 2.000 remuestreos) | −0,0008, IC 95 % [−0,0019, +0,0003] | sin diferencia (el intervalo incluye 0) |
| PR-AUC, CLAHE − original | −0,0005, IC 95 % [−0,0016, +0,0005] | sin diferencia |
| McNemar con umbral 0,5 | 43 imágenes solo acierta original, 46 solo CLAHE, p = 0,83 | sin diferencia |
| McNemar con el umbral de cada uno | 32 frente a 76, p < 0,0001 | "gana" CLAHE, pero es el **umbral**: el de original (0,879) se traslada peor a test |
| mismo punto de operación | CLAHE mejor en 3 de 4 y original en 1, siempre por poco (≤ 5 FP o 1,5 puntos de recall) | dentro del ruido que muestra el bootstrap |

**Conclusión: CLAHE no mejora la capacidad de clasificación** de MobileNet. La aparente ventaja con los umbrales de
val es un efecto del umbral, la misma lección que en la comparación entre arquitecturas.

En cambio, CLAHE sí **tiene coste en el servicio**:

- ~30 ms más por imagen en la API;
- una etapa más que reproducir exactamente; fue parte del *training-serving skew*
  ([09](09_serving.md#preparación-fiel-al-entrenamiento)).

Queda abierto si en B4 ocurre lo mismo: falta `efficientnetb4_original`. Si tampoco aporta, un B4 sin CLAHE
simplificaría el servicio sin perder calidad.

## Coste

| | MobileNetV3-L | EfficientNetB4 |
|---|---:|---:|
| tiempo de entrenamiento | **4,8 min** CLAHE (29 epochs) · 12,7 min original (32 epochs)* | 76 min (19 epochs) |
| rendimiento en GPU (solo el modelo) | **374–406 img/s** | 25 img/s |
| parámetros | **3,0 M** | 17,7 M |
| latencia servida en CPU (TF Serving, 1 imagen) | **~72 ms** | ~290–320 ms |

\* El original se entrenó con MLflow activo y con la memoria de WSL justa ([07](07_experiment_tracking.md#configuración)): sus tiempos no son
comparables con los del CLAHE.

La latencia en CPU es la medida en el servicio ([09](09_serving.md)). B4 da ~3 img/s de una en una y 5,3 img/s con 4
peticiones a la vez: suficiente para unas pocas cámaras a una imagen por segundo.

## Elección del modelo de producción

El gate de [`promote.py`](../src/registry/promote.py) ([08](08_model_registry.md)) compara las **falsas alarmas en
val**, porque el umbral fija el recall de val en ~95 % para todos:

| v | val.FPR | ¿sustituye a la v3? |
|---|---:|---|
| v1 MobileNet original | 1,67 % | no (+0,44 puntos) |
| v2 MobileNet CLAHE | 2,84 % | no (+1,60 puntos) |
| v3 B4 CLAHE | **1,24 %** | en `@production` |

La v3 está en producción: es la mejor en el punto de operación que importa (pocas falsas alarmas) y su latencia en
CPU es aceptable para el uso previsto. Elegir con test habría dado otro resultado: por recall de test ganaría la v2,
que es la peor en val.

## Conclusiones

- **B4** es mejor clasificador, sobre todo con pocas falsas alarmas, a cambio de unas 4 veces más latencia en CPU y
  15 veces menos rendimiento en GPU. Es el modelo en producción.
- **MobileNetV3** da alrededor del 95 % de la calidad con una fracción del coste. Es la alternativa si hace falta
  servir muchas cámaras o vídeo. Una regla de latencia en el gate lo decidiría de forma explícita (pendiente).
- **CLAHE no aporta** en MobileNet y añade complejidad al servicio.
- **Lección:** comparar modelos con su umbral de val aplicado a test mezcla la calidad del modelo con lo bien que se
  traslada el umbral. Hay que comparar con AUC o en el mismo punto de operación, y elegir con val.
