# 05 — Despliegue (plan)

> Estado: **planificado, sin implementar**. Primer modelo candidato: `mobilenetv3large_clahe`.

Diagramas de arquitectura (API y servidor de modelos en contenedores separados): [deployment_architecture.drawio](deployment_architecture.drawio).

## Qué incluye el modelo y qué no

| | ¿dentro de `model.keras`? |
|---|---|
| arquitectura + pesos | ✅ |
| normalización de entrada del backbone | ✅ (recibe RGB 0–255) |
| estado del optimizador | ✅ (sobra para inferencia, se elimina al exportar) |
| **CLAHE** | ❌ hay que aplicarlo en el servicio |
| **resize a 224/380** | ❌ hay que aplicarlo en el servicio |
| **umbral de decisión** | ❌ está en `metrics.json` |

Cargar el modelo: `keras.models.load_model("results/<exp>/model.keras")`.

## Arquitectura del programa de inferencia

```
                         ┌──────────────────────────────┐
                         │  deploy_config.json           │
                         │  · modelo (ruta / registry)   │
                         │  · image_size (224 / 380)     │
                         │  · CLAHE on/off + parámetros  │
                         │  · umbral  ← metrics.json     │
                         │  · suavizado (k de N frames)  │
                         └──────────────┬───────────────┘
                                        ▼
┌──────────────┐   ┌─────────────────────────────────────────────────────────────┐
│   ENTRADA    │   │                    PROGRAMA DE INFERENCIA                    │
│ · imagen     │   │  ┌───────────────┐   ┌───────────────┐   ┌────────────────┐  │
│ · carpeta    ├──►│  │ 1. PREPROCESO │──►│ 2. MODELO     │──►│ 3. POSTPROCESO │  │
│ · vídeo      │   │  │ decodificar   │   │ (.keras /     │   │ prob ≥ umbral  │  │
│ · webcam     │   │  │ → RGB         │   │  TFLite /     │   │  → ¿alarma?    │  │
│ · RTSP       │   │  │ CLAHE*        │   │  ONNX)        │   │ (vídeo) k de N │  │
│ · POST /api  │   │  │ resize        │   │ → prob [0,1]  │   │ frames → evento│  │
└──────────────┘   │  └───────────────┘   └───────────────┘   └───────┬────────┘  │
                   └──────────────────────────────────────────────────┼───────────┘
                                                                       ▼
                                   ┌──────────────────────────────────────────────┐
                                   │ SALIDA: JSON {prob, alarm, latency_ms},       │
                                   │ frame anotado, log, aviso (webhook/MQTT)      │
                                   └──────────────────────────────────────────────┘
   * CLAHE solo en los modelos *_clahe, mismo orden que en el entrenamiento (CLAHE → resize)
```

Estructura prevista:

```
src/deploy/
├── export.py        model.keras → SavedModel / TFLite / ONNX + comprobación de que coinciden las predicciones
├── preprocess.py    decodificar + CLAHE + resize (reutiliza src/preprocess)
├── predictor.py     clase Predictor: modelo + preprocesado + umbral → predict(img) → (prob, alarm)
├── smoothing.py     ventana temporal k de N para vídeo
├── cli.py           imagen | carpeta | vídeo | webcam
├── api.py           FastAPI: POST /predict, GET /health, GET /metrics
├── e2e_test.py      manda el test a la API y comprueba que salen las métricas de metrics.json
└── Dockerfile
```

Principio clave: `Predictor` es el **único** sitio donde viven el preprocesado, el modelo y el umbral. CLI, API y vídeo
son capas finas encima, así se evita que el entrenamiento y el servicio procesen las imágenes de forma distinta.

## Simulación MLOps en local

```
 ENTRENAMIENTO ──► MLflow Tracking + Model Registry (alias @staging / @production)
                   artefactos en RustFS (= S3)
                          │  CI/CD (GitHub Actions): tests → build imagen → push → deploy
                          ▼
 Cámaras simuladas ──► Cluster K8s (kind): Ingress → Service → Pods FastAPI (HPA, canary v2)
 + Locust (carga)                                    │ /metrics         │ log de predicciones
                                                     ▼                  ▼
                                              Prometheus ──► Grafana ◄── Evidently (drift)
```

| local | AWS | GCP | Azure |
|---|---|---|---|
| RustFS | S3 | Cloud Storage | Blob Storage |
| MLflow | SageMaker Model Registry | Vertex AI Model Registry | Azure ML |
| registry local / GHCR | ECR | Artifact Registry | ACR |
| kind / minikube | EKS | GKE | AKS |
| Prometheus + Grafana | CloudWatch | Cloud Monitoring | Azure Monitor |
| GitHub Actions | CodePipeline | Cloud Build | Azure DevOps |

## Hoja de ruta

1. **Servicio en contenedor:** `Predictor` + FastAPI + Dockerfile (TFLite/ONNX en CPU). Prueba de extremo a extremo con el test.
2. **Registro de modelos:** MLflow + Postgres + RustFS (docker-compose, ver [06_mlops.md](06_mlops.md)). `train.py` registra cada modelo y la API carga `models:/fire-detector@production`.
3. **Observabilidad:** Prometheus + Grafana, simulador de cámaras y Locust.
4. **Kubernetes:** kind, Deployment, Service, HPA, canary MobileNet v1 frente a B4 v2.
5. **CI/CD:** GitHub Actions con un umbral mínimo de PR-AUC para dejar pasar el modelo.
6. **Monitorización del modelo:** Evidently. Se simula drift con imágenes `original` sobre un modelo `clahe`, o con imágenes oscurecidas o con niebla.
7. **Nube real (opcional):** Cloud Run o EKS/GKE con capa gratuita.

**Restricción de recursos:** WSL tiene ~7,7 GiB de RAM. Para kind con el stack de monitorización conviene subirlo a 12 GB y
servir MobileNet en TFLite/ONNX (~200 MB por pod).
