# 01 — Análisis exploratorio (EDA)

Notebook: [src/eda/classes_analysis.ipynb](../src/eda/classes_analysis.ipynb)

## Dataset

[**Smoke-Fire-Detection-YOLO**](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) (Kaggle, Sayed
Gamal, licencia **CC0: Public Domain**). Es una reorganización del dataset
[D-Fire](https://github.com/gaiasd/DFireDataset) de GAIA, a la que añade un split de validación.

Si se usa, el autor pide citar el artículo original de D-Fire:

> P. V. A. B. de Venâncio, A. C. Lisboa, A. V. Barbosa. *An automatic fire detection system based on deep
> convolutional neural networks for low-power, resource-constrained devices.* Neural Computing and Applications,
> 34(18), 15349–15368, 2022. [doi:10.1007/s00521-022-07467-z](https://doi.org/10.1007/s00521-022-07467-z)

- Ubicación: `data/original_dataset/{train,val,test}/{images,labels}`, versionado con DVC ([06](06_data_versioning.md)).
- Formato de las etiquetas: **YOLO (detección)**. Cada imagen `X.jpg` tiene un `X.txt` con una línea por caja:
  `class x_center y_center width height` (coordenadas normalizadas).
- Clases YOLO: `0 = smoke`, `1 = fire`.
- Una imagen **sin fichero de etiquetas o con el fichero vacío** es *background*: no tiene fuego ni humo.
- Resolución de origen: mayoritariamente 1280×720, algunas 1920×1080.

## Integridad

| split | imágenes | ficheros de etiquetas | sin etiqueta | etiquetas vacías | etiquetas huérfanas | líneas mal formadas |
|---|---:|---:|---:|---:|---:|---:|
| train | 14.122 | 14.122 | 0 | 6.458 | 0 | 0 |
| val | 3.099 | 3.099 | 0 | 1.375 | 0 | 0 |
| test | 4.306 | 4.306 | 0 | 2.005 | 0 | 0 |
| **total** | **21.527** | 21.527 | 0 | 9.838 | 0 | 0 |

El dataset está limpio: no hay etiquetas huérfanas ni líneas mal formadas, y los fondos se representan con ficheros vacíos.

## Tamaño de los splits

| split | imágenes | % |
|---|---:|---:|
| train | 14.122 | 65,6 |
| val | 3.099 | 14,4 |
| test | 4.306 | 20,0 |

## Balance de clases (dataset completo)

**Alarma frente a fondo (nivel imagen)**

| | imágenes | % |
|---|---:|---:|
| fuego/humo | 11.689 | 54,3 |
| background | 9.838 | 45,7 |

Ratio de desbalance de 1,19 : 1. Es prácticamente balanceado para la tarea binaria, así que **no hace falta
ponderar clases ni hacer re-muestreo**.

**Categorías de imagen**

| categoría | imágenes | % |
|---|---:|---:|
| background | 9.838 | 45,7 |
| smoke_only | 5.867 | 27,3 |
| fire_only | 1.164 | 5,4 |
| fire_and_smoke | 4.658 | 21,6 |

**Cajas (nivel instancia):** 26.557 en total, 11.865 de humo (44,7 %) y 14.692 de fuego (55,3 %).
Las imágenes positivas tienen de media 2,27 cajas (mediana 1, máximo 23).

## Consistencia entre splits

Las proporciones son casi idénticas en train, val y test (diferencias de 1-2 puntos):

| split | % fuego/humo | % smoke_only | % fire_only | % fire_and_smoke |
|---|---:|---:|---:|---:|
| train | 54,3 | 27,2 | 5,5 | 21,7 |
| val | 55,6 | 27,3 | 5,6 | 22,8 |
| test | 53,4 | 27,5 | 5,1 | 20,8 |

Por eso las métricas de val y test son comparables, y elegir el umbral en val es razonable.

## Conclusiones que guían el resto del proyecto

1. **La tarea se plantea como clasificación binaria** (alarma sí o no) a nivel de imagen, a partir de las etiquetas de detección.
2. **No hay que corregir el desbalance** en la tarea binaria.
3. **Se mide la tasa de alarma por subtipo** además de la global, porque la métrica global puede esconder un subtipo
   que falla:
   - **`smoke_only` (27 %), hipótesis: es el caso difícil.** El humo sin llama se confunde con niebla, nubes o polvo.
     Los resultados la confirman ([04](04_results.md)): en test se detecta un 87–91 % frente al 94–96 % de
     `fire_only` y el 99 % de `fire_and_smoke`.
   - **`fire_only` es la minoritaria (5 %):** en test son ~220 imágenes, así que cada una mueve su tasa ~0,45 puntos.
     Diferencias de uno o dos puntos entre modelos en este subtipo no son significativas.
   - En `background`, la tasa de alarma son las **falsas alarmas**.
4. Los splits ya vienen dados y son consistentes, así que se usan tal cual.
