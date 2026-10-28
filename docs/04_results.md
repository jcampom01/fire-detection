# 04 — Resultados

Análisis completo en [compare_results.ipynb](../src/train/compare_results.ipynb). Datos en `results/<experimento>/metrics.json`.

**Estado:** entrenados `mobilenetv3large_clahe` y `efficientnetb4_clahe`. Pendientes: `mobilenetv3large_original` y
`efficientnetb4_original`.

## Métricas en test (umbral elegido en val para recall ≥ 95 %)

| | MobileNetV3-L CLAHE | EfficientNetB4 CLAHE |
|---|---:|---:|
| umbral | 0,728 | 0,823 |
| recall | **94,9 %** | 93,2 % |
| precisión | 98,2 % | **99,2 %** |
| FPR (falsas alarmas) | 2,0 % | **0,85 %** |
| FN / FP | 118 / 40 | 157 / 17 |
| F1 | **0,965** | 0,961 |
| ROC-AUC | 0,9921 | **0,9947** |
| PR-AUC | 0,9938 | **0,9959** |
| mejor `val_loss` | 0,107 | **0,073** |

**Con umbral 0,5:** las dos tienen accuracy del 96,7 %. MobileNet tiene 75 FN y 67 FP, B4 tiene 82 FN y 58 FP.

## Tasa de alarma por subtipo (test)

| subtipo | MobileNetV3-L | EfficientNetB4 |
|---|---:|---:|
| background (FPR) | 2,0 % | **0,8 %** |
| smoke_only | **91,1 %** | 88,3 % |
| fire_only | 95,0 % | **95,9 %** |
| fire_and_smoke | **99,8 %** | 99,0 % |

Como se esperaba en el EDA, **smoke_only es el caso más difícil** en los dos modelos.

## Comparación justa: el mismo punto de operación

En la tabla principal parece que MobileNet gana en recall, pero es un efecto del umbral: el de B4 (0,82) se trasladó
peor de val a test y lo deja en un punto más conservador. Si se fija el mismo recall o el mismo FPR en test:

| | MobileNetV3-L | EfficientNetB4 |
|---|---:|---:|
| FP con recall = 95 % | 40 | **28** |
| FP con recall = 97 % | 81 | **67** |
| recall con FPR = 1 % | 88,9 % | **93,6 %** |
| recall con FPR = 2 % | 95,5 % | **96,1 %** |

**B4 es mejor clasificador** en cualquier punto de la curva. La diferencia es mayor en la zona de pocas falsas alarmas.

## Coste

| | MobileNetV3-L | EfficientNetB4 |
|---|---:|---:|
| tiempo de entrenamiento | **4,8 min** (29 epochs) | 76 min (19 epochs) |
| throughput (GPU, solo el modelo) | **374 img/s** | 25 img/s |
| parámetros | **3,0 M** | 17,7 M |

## Conclusión provisional

- **B4**: mejor calidad, sobre todo si se necesitan muy pocas falsas alarmas, pero es unas 15 veces más lento.
- **MobileNetV3**: alrededor del 95 % del rendimiento con el 6 % del coste. Es el **candidato para el primer despliegue**
  (CPU, tiempo real, varias réplicas).
- Lección: comparar modelos con su umbral de val aplicado a test mezcla la calidad del modelo con lo bien que se
  traslada el umbral. Hay que comparar con AUC o en el mismo punto de operación.
- Falta confirmar si **CLAHE aporta** frente a `original` (McNemar y bootstrap en el notebook) cuando estén las 4 ejecuciones.
