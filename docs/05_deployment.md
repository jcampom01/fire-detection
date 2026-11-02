# 05 — Despliegue (plan)

> Estado: **en curso**. Registro de modelos hecho ([09](09_model_registry.md)); v3 en `@production`, servida con
> TF Serving detrás de una API FastAPI y desplegada automáticamente con `promote.py` ([10_serving.md](10_serving.md)).
> Pendiente: la prueba de extremo a extremo con todo el test.
>
> **Cambio de plan:** este documento proponía una API que cargaba el modelo ella misma. Se ha cambiado a **API y
> servidor de modelos (TF Serving) en contenedores separados**, con la conversión a SavedModel al hacer `deploy`. Lo
> que sigue vigente de aquí son las etapas de la inferencia (preproceso → modelo → postproceso) y el principio de un
> único sitio para el preproceso; la arquitectura actual está en [10_serving.md](10_serving.md).

Arquitectura MLOps actual: [mlops_architecture.drawio](mlops_architecture.drawio).

## Qué incluye el modelo y qué no

| | ¿dentro del modelo? | dónde está |
|---|---|---|
| arquitectura + pesos | ✅ | SavedModel |
| normalización de entrada del backbone | ✅ (recibe RGB 0–255) | SavedModel |
| estado del optimizador | solo en `model.keras` | se elimina al exportar a SavedModel |
| **CLAHE** | ❌ | lo aplica la API, con los parámetros de `meta.json` |
| **resize a 224/380** | ❌ | lo aplica la API (`meta.json`) |
| **umbral de decisión** | ❌ | `meta.json` (campo `metadata` del `MLmodel` en MLflow) |

## Etapas de la inferencia (plan original)

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

Con la arquitectura separada, el preproceso y el postproceso viven en la API y el modelo en TF Serving. Sigue vigente el
principio clave: el preprocesado está en **un único sitio** (la API, guiada por `meta.json` y reutilizando la misma
función CLAHE del preprocesado), para que entrenamiento y servicio no procesen las imágenes de forma distinta
(*training-serving skew*). El vídeo (suavizado k de N), la CLI y la exportación a TFLite/ONNX quedan como extensiones.

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

1. ~~**Registro de modelos**~~ → hecho: MLflow + Postgres + RustFS ([06](06_mlops.md), [08](08_experiment_tracking.md), [09](09_model_registry.md)).
2. **Servicio en contenedores** (en curso, [10_serving.md](10_serving.md)): TF Serving ✅ + API FastAPI ✅ + despliegue
   automático desde `@production` ✅ + prueba de extremo a extremo con el test (pendiente).
3. **Observabilidad:** Prometheus + Grafana, simulador de cámaras y Locust.
4. **Kubernetes:** kind, Deployment, Service, HPA, canary entre versiones (por ejemplo, B4 v3 frente a MobileNet).
5. **CI/CD:** GitHub Actions con un umbral mínimo de PR-AUC para dejar pasar el modelo.
6. **Monitorización del modelo:** Evidently. Se simula drift con imágenes `original` sobre un modelo `clahe`, o con imágenes oscurecidas o con niebla.
7. **Nube real (opcional):** Cloud Run o EKS/GKE con capa gratuita.

**Restricción de recursos:** WSL tiene ~7,7 GiB de RAM. TF Serving con B4 ocupa 0,6–1,7 GB y responde en ~320 ms por
imagen en CPU ([10](10_serving.md#s1e--predicciones-por-rest-y-rendimiento-en-cpu)). Para kind con el stack de
monitorización conviene subir la RAM de WSL a 12 GB; un modelo MobileNet (o TFLite/ONNX) reduciría memoria y latencia.
