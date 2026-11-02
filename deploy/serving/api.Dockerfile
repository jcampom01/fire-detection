# Inference API: FastAPI + OpenCV, no TensorFlow (the model runs in TF Serving)
# Build context: the repository root (see docker-compose.yml)
FROM python:3.12.16-slim

WORKDIR /app

# Dependencies first: this layer is reused while only the code changes
COPY src/api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Same layout as src/, so `api.app` and `preprocess.clahe` import like on the host
COPY src/api/ api/
COPY src/preprocess/clahe.py preprocess/clahe.py

# Do not run as root
RUN useradd --create-home --uid 1000 api
USER api

ENV MODELS_DIR=/models \
    TF_SERVING_URL=http://tf-serving:8501 \
    MODEL_NAME=fire_detector \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
