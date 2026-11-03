# Learning — las herramientas en detalle

Explicaciones de cómo funcionan por dentro las herramientas del proyecto, con ejemplos reales de este repositorio.
No hacen falta para usar el proyecto: para eso está [docs/](../README.md).

| Documento | Contenido |
|---|---|
| [dvc.md](dvc.md) | Hash, almacenamiento por contenido, índice y ticket `.dvc`, los cuatro sitios (datos, caché, ticket, bucket), relación con git y MLflow, organización del bucket |
| [mlflow.md](mlflow.md) | Cliente y servidor, jerarquía (experimento, run, params, métricas, artefactos, modelo), dónde guarda cada cosa (Postgres y S3), cómo viaja cada llamada, MLflow frente a DVC, Model Registry (alias, tags, entornos), borrado en dos fases |
| [serving.md](serving.md) | Por qué separar la API del modelo y alternativas, TF Serving, SavedModel, cambio de versión sin cortes, despliegue push frente a pull, patrones de la industria, FastAPI y uvicorn, liveness y readiness |
| [training_serving_skew.md](training_serving_skew.md) | Un caso real: cómo dos compresiones JPEG y un decodificador distinto cambiaban decisiones del modelo sin dar ningún error, y cómo se resolvió |
