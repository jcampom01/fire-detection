# Herramientas

## En uso

| herramienta | para qué sirve |
|---|---|
| **Docker** | Ejecuta cada servicio en un contenedor aislado, a partir de una imagen reproducible. |
| **Docker Compose** | Define y levanta varios contenedores a la vez (red, volúmenes, variables) desde un único `docker-compose.yml`. |
| **Dockerfile** | Receta para construir una imagen propia; aquí, la de MLflow con drivers extra. |
| **PostgreSQL** | Base de datos relacional donde MLflow guarda los metadatos: runs, params, métricas, versiones y alias. |
| **RustFS** | Almacenamiento de objetos compatible con S3 donde MLflow guarda los ficheros (modelos, png, csv). Sustituye a MinIO. |
| **MLflow** | Tracking de experimentos y registro de modelos, con API e interfaz web. |
| **Adminer** | Interfaz web para la base de datos; permite ver las tablas de MLflow sin publicar el puerto de Postgres. Perfil `tools`. |
| **aws-cli** | Cliente oficial de AWS para línea de comandos; aquí, para crear buckets en RustFS a través de la API S3. |
| **psycopg** | Driver de Python para PostgreSQL; lo usa MLflow para conectarse a la base de datos. |
| **boto3** | Cliente S3 de Python; lo usa MLflow para subir y bajar ficheros de RustFS. |
| **TensorFlow / Keras** | Entrenamiento de los modelos (transfer learning con backbones de `keras.applications`). |
| **OpenCV** | Lectura de imágenes, CLAHE y redimensionado. |
| **pandas / NumPy / scikit-learn / SciPy** | Datos tabulares, cálculo numérico, métricas y tests estadísticos. |
| **Matplotlib** | Gráficas de entrenamiento y evaluación. |
| **Jupyter (ipykernel)** | Notebooks de EDA y comparación de resultados. |
| **draw.io** | Diagramas de arquitectura ([deployment_architecture.drawio](deployment_architecture.drawio)). |

## Previstas

| herramienta | para qué servirá |
|---|---|
| **TensorFlow Serving** | Servidor de modelos: carga el SavedModel y responde peticiones de inferencia por gRPC/REST. |
| **FastAPI** | API de inferencia: preproceso, llamada al modelo, umbral y endpoints `/predict`, `/health`, `/metrics`. |
| **Prometheus** | Recoge métricas de los servicios (latencia, peticiones, alarmas). |
| **Grafana** | Dashboards y alertas sobre las métricas de Prometheus. |
| **Evidently** | Detección de drift comparando las predicciones de producción con las de referencia. |
| **Locust** | Pruebas de carga contra la API. |
| **GitHub Actions** | CI/CD: tests, construcción y publicación de imágenes. |
