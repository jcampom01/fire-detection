# Documentación — Fire Detection

Clasificador binario de alarma de incendio (fuego y/o humo frente a fondo) entrenado con transfer learning,
como base para practicar un despliegue MLOps completo.

| Documento | Contenido |
|---|---|
| [01_eda.md](01_eda.md) | Dataset, integridad de las etiquetas y balance de clases |
| [02_preprocessing.md](02_preprocessing.md) | Etiquetas de clasificación, CLAHE y redimensionado |
| [03_training.md](03_training.md) | Pipeline de entrenamiento, modelos, configs y artefactos generados |
| [04_results.md](04_results.md) | Resultados obtenidos y comparación entre modelos |
| [05_deployment.md](05_deployment.md) | Arquitectura de inferencia y plan de despliegue MLOps |

## Flujo general

```
data/original_dataset (YOLO)
   │
   ├─► prepare_labels.py ──► data/classification_labels/{train,val,test}.csv
   ├─► clahe_preprocess.py ──► data/clahe_dataset
   └─► resize_dataset.py ──► data/{original,clahe}_dataset_{224,380}
                                   │
                                   ▼
                     src/train/train.py --config configs/<exp>.json
                                   │
                                   ▼
                     results/<exp>/ (modelo, métricas, predicciones, gráficas)
                                   │
                                   ▼
                     compare_results.ipynb ──► elección del modelo ──► despliegue
```

## Estructura del repositorio

```
src/
├── eda/classes_analysis.ipynb        análisis exploratorio
├── preprocess/                       etiquetas, CLAHE, resize (+ configs JSON)
└── train/                            entrenamiento, evaluación y comparación
    └── configs/                      un JSON por experimento (modelo × dataset)
results/<experimento>/                salidas de cada entrenamiento
docs/                                 esta documentación
```
