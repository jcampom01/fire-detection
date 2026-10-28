# MLflow tracking server + drivers for Postgres (metadata) and S3 (artifacts)
FROM ghcr.io/mlflow/mlflow:v3.17.2

RUN pip install --no-cache-dir \
    "psycopg[binary]==3.3.6" \
    boto3==1.43.121
