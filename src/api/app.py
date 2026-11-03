"""Inference API: receives an image, prepares it, asks TF Serving and applies the threshold.

The image is prepared exactly like the training dataset: CLAHE (if the model uses it) and resize, each followed by
a JPEG write at the dataset quality. The final JPEG bytes are sent to TF Serving, and the model decodes them with
TensorFlow, as in training (see src/registry/export.py for why).

Endpoints:
    GET  /          web page: drop an image and see the prediction (static/index.html, calls /model and /predict)
    POST /predict   image file -> {prob, alarm, threshold, model_version, latency_ms}
    GET  /model     version being served and its metadata
    GET  /health    liveness: the process is up (never calls TF Serving)
    GET  /ready     readiness: TF Serving answers and a model version with its metadata is loaded
    GET  /docs      interactive documentation (FastAPI)

Errors: 400 the file is not an image · 503 TF Serving unreachable or no model loaded · 502 TF Serving failed.

Run from the project root (src/ on the path, so preprocess.clahe is importable):
    uvicorn api.app:app --app-dir src --port 8080
"""

import base64
import json
import logging
import os
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from preprocess.clahe import apply_clahe  # same function used to build the training dataset

TF_SERVING_URL = os.environ.get("TF_SERVING_URL", "http://127.0.0.1:8501")
MODEL_NAME = os.environ.get("MODEL_NAME", "fire_detector")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "deploy/serving/models"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Fire detection API")


# --- Preparation ---

def to_jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "could not encode the image")
    return buf


def prepare(data, meta):
    """Image file bytes -> JPEG bytes for the model, built like the training dataset:
    CLAHE at full resolution -> JPEG -> resize -> JPEG. The dataset scripts wrote a JPEG after each step and
    read it back, so the model saw those compression artifacts. None if `data` is not an image."""
    prep = meta["preprocessing"]
    quality = prep["jpeg_quality"]
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return None
    if prep["clahe"]:  # null for models trained on the original dataset
        img = apply_clahe(img, prep["clahe"]["clip_limit"], prep["clahe"]["tile_grid"])
        img = cv2.imdecode(to_jpeg(img, quality), cv2.IMREAD_COLOR)
    size = meta["image_size"]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return to_jpeg(img, quality).tobytes()  # BGR in, standard JPEG out: the model decodes it to RGB


# --- Model server ---

def served_model():
    """(version, meta) of the model TF Serving has loaded. 503 if it cannot be determined."""
    try:
        status = httpx.get(f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}", timeout=5)
        status.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(503, f"model server unavailable: {e}") from None
    versions = [int(v["version"]) for v in status.json()["model_version_status"] if v["state"] == "AVAILABLE"]
    if not versions:
        raise HTTPException(503, "no model version loaded yet")
    # During a switch both versions can be AVAILABLE for a moment; if the chosen one is unloaded before the
    # prediction, ask_model returns None and /predict retries with the remaining one
    version = max(versions)
    meta_path = MODELS_DIR / MODEL_NAME / str(version) / "meta.json"
    if not meta_path.is_file():
        raise HTTPException(503, f"metadata missing for version {version}")
    return version, json.loads(meta_path.read_text())


def ask_model(version, jpeg):
    """Probability for one prepared image, or None if TF Serving no longer has this version (it just switched)."""
    url = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}/versions/{version}:predict"
    payload = {"instances": [{"b64": base64.b64encode(jpeg).decode()}]}  # binary data goes in JSON as {"b64": ...}
    try:
        response = httpx.post(url, json=payload, timeout=30)
    except httpx.HTTPError as e:
        raise HTTPException(503, f"model server unavailable: {e}") from None
    if response.status_code == 404:
        return None
    if response.is_error:
        raise HTTPException(502, f"model server error {response.status_code}: {response.text[:200]}")
    return response.json()["predictions"][0][0]


# --- Endpoints ---

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = file.file.read()

    # Two attempts: if the served version changes between asking and predicting, redo it with the new one
    for _ in range(2):
        version, meta = served_model()
        jpeg = prepare(data, meta)
        if jpeg is None:
            raise HTTPException(400, "not a valid image")
        t1 = time.perf_counter()
        prob = ask_model(version, jpeg)
        if prob is not None:
            break
    else:
        raise HTTPException(503, "model version changed during the request, try again")
    t2 = time.perf_counter()

    result = {
        "prob": prob,
        "alarm": prob >= meta["threshold"],
        "threshold": meta["threshold"],
        "model_version": version,
        "latency_ms": {"preprocess": round((t1 - t0) * 1000), "model": round((t2 - t1) * 1000)},
    }
    log.info("predict file=%s version=%d prob=%.4f alarm=%s latency_ms=%d",
             file.filename, version, prob, result["alarm"], round((t2 - t0) * 1000))
    return result


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/model")
def model():
    version, meta = served_model()
    return {"name": MODEL_NAME, "version": version, "meta": meta}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    version, _ = served_model()  # raises 503 if not ready
    return {"status": "ready", "model_version": version}
