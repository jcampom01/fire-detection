# Documentación — Fire Detection

Clasificador binario de alarma de incendio (fuego y/o humo frente a fondo) entrenado con transfer learning,
como base para practicar un despliegue MLOps completo.

| Documento | Contenido |
|---|---|
| [01_eda.md](01_eda.md) | Dataset, integridad de las etiquetas y balance de clases |
| [02_preprocessing.md](02_preprocessing.md) | Etiquetas de clasificación, CLAHE y redimensionado |
| [03_training.md](03_training.md) | Pipeline de entrenamiento, modelos, configs y artefactos generados |
| [04_results.md](04_results.md) | Resultados obtenidos y comparación entre modelos |
| [05_platform.md](05_platform.md) | Plataforma MLOps: MLflow + Postgres + RustFS en Docker |
| [06_data_versioning.md](06_data_versioning.md) | Versionado de datos con DVC: qué se versiona, configuración y uso |
| [07_experiment_tracking.md](07_experiment_tracking.md) | Tracking de experimentos con MLflow: qué registra cada run y operaciones |
| [08_model_registry.md](08_model_registry.md) | Registro de modelos: alias, gate y promoción con `promote.py` |
| [09_serving.md](09_serving.md) | Servir el modelo: TF Serving, API, preparación fiel al entrenamiento, despliegue automático y verificación |
| [ports.md](ports.md) | Puertos en uso y conexiones entre servicios |
| [tools.md](tools.md) | Herramientas del proyecto y para qué sirve cada una |
| [learning/](learning/README.md) | Las herramientas en detalle: cómo funcionan por dentro, con ejemplos de este proyecto |

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
├── train/                            entrenamiento, evaluación, tracking en MLflow e importación de runs
│   └── configs/                      un JSON por experimento (modelo × dataset)
├── registry/                         promoción de modelos (alias), sus reglas, exportación a TF Serving y verificación
└── api/                              API de inferencia (FastAPI) y página web (static/)
results/<experimento>/                salidas de cada entrenamiento
deploy/mlops/                         plataforma MLOps (docker-compose: MLflow, Postgres, RustFS)
deploy/serving/                       servicio de inferencia (docker-compose: TF Serving + API)
docs/                                 esta documentación
```
