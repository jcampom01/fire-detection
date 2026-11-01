# 03 — Entrenamiento

Código en [src/train/](../src/train/). Punto de entrada:

```bash
python src/train/train.py --config src/train/configs/mobilenetv3large_clahe.json
```

## Tarea

Clasificación binaria a nivel de imagen:

- `alarm = 1`: la imagen contiene fuego y/o humo.
- `alarm = 0`: background.

## Organización del código

| fichero | responsabilidad |
|---|---|
| `train.py` | orquesta los pasos: config → datos → modelo → entrenamiento → evaluación → guardado |
| `tf_setup.py` | precarga las librerías CUDA (en WSL, TensorFlow no siempre las encuentra y cae a CPU), semillas, mixed precision, `memory_growth` en GPU |
| `config.py` | lee el JSON y resuelve las rutas respecto a la raíz del proyecto |
| `data.py` | lee los CSV, construye `alarm` y `subtype`, pipeline `tf.data` que lee las imágenes desde disco |
| `model.py` | backbones, cabeza binaria, descongelado parcial, compilación |
| `trainer.py` | entrenamiento en dos fases con checkpoint de los mejores pesos |
| `evaluation.py` | predicciones, elección del umbral, métricas, tasa por subtipo, velocidad de inferencia |
| `plots.py` | curvas de entrenamiento, matriz de confusión, ROC/PR |
| `tracking.py` | run de MLflow: params, métricas, artefactos y modelo registrado ([08](08_experiment_tracking.md#m4--integración-en-trainpy)) |
| `import_results.py` | importa a MLflow entrenamientos anteriores a la integración ([08](08_experiment_tracking.md#m5--primer-entrenamiento-real-e-importación-de-los-anteriores)) |
| `compare_results.ipynb` | compara todos los experimentos de `results/` |

## Modelos

Todos son backbones de `keras.applications` preentrenados en ImageNet. Incluyen su propia normalización de entrada,
así que reciben **RGB en el rango 0–255**.

| modelo | resolución | batch | parámetros |
|---|---:|---:|---:|
| MobileNetV3-Large | 224 | 64 | 3,0 M |
| EfficientNetB4 | 380 | 12 | 17,7 M |

`model.py` también admite `efficientnetb0` y `convnexttiny`, pero se descartaron y no tienen config.

**Cabeza:** `backbone(training=False)` → `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(1, sigmoid, float32)`.
`training=False` mantiene las BatchNorm en modo inferencia también cuando se descongela el backbone.

## Experimentos

Uno por cada combinación modelo × dataset, con un JSON en [configs/](../src/train/configs/):
`{mobilenetv3large, efficientnetb4} × {original, clahe}`, 4 en total.

## Entrenamiento en dos fases

| | fase 1 | fase 2 (fine-tuning) |
|---|---|---|
| backbone | congelado | último 30 % de capas entrenable (BatchNorm sigue congelada) |
| learning rate | 1e-3 | 1e-5 |
| epochs máx. | 10 | 30 |
| early stopping (patience) | 3 | 5 |
| ReduceLROnPlateau | — | factor 0,2, patience 2, min 1e-7 |

- Optimizador Adam, loss `binary_crossentropy`, mixed precision `float16`.
- Métrica monitorizada: **`val_pr_auc`**. Se guardan los mejores pesos de las dos fases en conjunto y se cargan al final.
- La fase 2 parte de los mejores pesos de la fase 1.

## Umbral de decisión

No se usa 0,5. El umbral se elige **en val** como el de mayor precisión entre los que alcanzan
**recall ≥ 95 %** (`threshold.target_recall`) y se aplica sin cambios a test. La idea es priorizar no perder
incendios. `metrics.json` también incluye `test_at_0.5` como referencia.

## Salidas — `results/<experimento>/`

Cada entrenamiento es además un run de MLflow (experimento `fire-detection`) con todo lo de esta tabla salvo
`best.weights.h5`, y el modelo se registra como nueva versión de `fire-detector`. Requiere el stack levantado
(`cd deploy/mlops && docker compose up -d`): ver [08_experiment_tracking.md](08_experiment_tracking.md).

| fichero | contenido |
|---|---|
| `config.json` | copia de la config usada |
| `model.keras` | **modelo completo**: arquitectura, pesos, estado del optimizador y compile |
| `best.weights.h5` | solo los pesos (y el optimizador). Necesita `build_model(cfg)` para usarse |
| `history.csv` | métricas por epoch de las dos fases |
| `metrics.json` | umbral, métricas de val/test, tasa de alarma por subtipo, velocidad |
| `meta.json` | lo que necesita la inferencia además del modelo: umbral y preproceso (CLAHE, tamaño, interpolación) |
| `predictions_{val,test}.csv` | probabilidad por imagen |
| `curves.png`, `confusion_matrix.png`, `metrics.png` | gráficas |

En los dos ficheros de modelo, **~75 % del tamaño es el estado de Adam** (por ejemplo, en MobileNet hay 12 MB de pesos y 39,5 MB de optimizador).
Para inferencia no hace falta, y desaparece al exportar.

## Incidencias

### EfficientNetB4: `Killed` al empezar la fase 2

- **Síntoma:** el proceso muere en `Epoch 1/30` de la fase 2 con el mensaje `Killed`.
- **Causa** (en `dmesg`: `ptxas invoked oom-killer`): Keras 3 activa por defecto XLA (`jit_compile="auto"`) en GPU.
  Al descongelar el backbone, el grafo de entrenamiento de B4 a 380 px es muy grande, y compilarlo con `ptxas`
  agotó la RAM de WSL (~7,7 GiB). Es un OOM de **RAM del sistema**, no de memoria de la GPU.
- **Solución:** `jit_compile=False` en `compile_model` ([model.py](../src/train/model.py)). El entrenamiento **sigue en la GPU**
  usando los kernels de cuDNN y cuBLAS. Solo se desactiva la fusión de operaciones de XLA, y puede ser algo más lento.
- **Alternativas:** subir la memoria de WSL en `.wslconfig` (`memory=12GB`, `swap=8GB`) o bajar el `batch_size`.

## Entorno

TensorFlow 2.21, Keras 3.15, GPU RTX 3070 (8 GB), WSL2 con ~7,7 GiB de RAM.

Dependencias:

- `requirements.txt`: dependencias directas, con las versiones fijadas a las usadas para entrenar.
- `requirements-lock.txt`: el entorno completo (`pip freeze`). Sirve para reproducirlo exactamente:
  `pip install -r requirements-lock.txt`.
