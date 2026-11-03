# 03 — Entrenamiento

Código en [src/train/](../src/train/). Cada experimento es un JSON en [configs/](../src/train/configs/):

```bash
.venv/bin/python src/train/train.py --config src/train/configs/efficientnetb4_clahe.json
```

Cada entrenamiento se registra en MLflow, así que el stack `deploy/mlops` tiene que estar levantado
([05](05_platform.md)); si no responde, `train.py` se detiene antes de empezar.

## Tarea

Clasificación binaria a nivel de imagen:

- `alarm = 1`: la imagen contiene fuego y/o humo.
- `alarm = 0`: fondo (*background*).

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
| `tracking.py` | run de MLflow: params, métricas, artefactos y modelo registrado ([07](07_experiment_tracking.md#qué-registra-cada-run)) |
| `import_results.py` | importa a MLflow entrenamientos hechos sin MLflow ([07](07_experiment_tracking.md#importar-entrenamientos-anteriores)) |
| `compare_results.ipynb` | compara los experimentos de `results/` |

## Config

```json
{
  "experiment": "efficientnetb4_clahe", "model": "efficientnetb4", "dataset": "clahe",
  "data": {"labels_dir": "data/classification_labels", "image_root": "data/clahe_dataset_380"},
  "output_dir": "results", "image_size": 380, "batch_size": 12, "seed": 42, "mixed_precision": true,
  "head": {"dropout": 0.3},
  "phase1": {"epochs": 10, "lr": 0.001, "patience": 3},
  "phase2": {"epochs": 30, "lr": 1e-05, "patience": 5, "unfreeze_fraction": 0.3,
             "reduce_lr_factor": 0.2, "reduce_lr_patience": 2},
  "monitor": "val_pr_auc",
  "threshold": {"target_recall": 0.95},
  "debug_max_images": null
}
```

| clave | qué controla |
|---|---|
| `model`, `dataset`, `image_size`, `data.image_root` | backbone y qué copia del dataset se lee ([02](02_preprocessing.md)) |
| `phase1`, `phase2` | las dos fases (ver abajo) |
| `monitor` | métrica que deciden el checkpoint, el early stopping y `ReduceLROnPlateau` |
| `threshold.target_recall` | recall mínimo en val al elegir el umbral |
| `debug_max_images` | si tiene un número, usa solo esa muestra de cada split para probar el pipeline en minutos. El run se marca `debug` y el modelo no se registra |

## Modelos

Backbones de `keras.applications` preentrenados en ImageNet. Incluyen su propia normalización de entrada, así que
reciben **RGB en el rango 0–255**.

| modelo | resolución | batch | parámetros |
|---|---:|---:|---:|
| MobileNetV3-Large | 224 | 64 | 3,0 M |
| EfficientNetB4 | 380 | 12 | 17,7 M |

`model.py` también admite `efficientnetb0` y `convnexttiny`, que se descartaron y no tienen config.

**Cabeza:** `backbone(training=False)` → `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(1, sigmoid, float32)`.
`training=False` mantiene las BatchNorm en modo inferencia también cuando se descongela el backbone.

## Experimentos

Uno por cada combinación modelo × dataset: `{mobilenetv3large, efficientnetb4} × {original, clahe}`.

| experimento | estado |
|---|---|
| `mobilenetv3large_original` | entrenado (v1 en el registro) |
| `mobilenetv3large_clahe` | entrenado (v2) |
| `efficientnetb4_clahe` | entrenado (v3, en producción) |
| `efficientnetb4_original` | sin entrenar: diría si CLAHE aporta en B4 |

Resultados: [04_results.md](04_results.md).

## Entrenamiento en dos fases

| | fase 1 | fase 2 (fine-tuning) |
|---|---|---|
| backbone | congelado | último 30 % de capas entrenable (BatchNorm sigue congelada) |
| learning rate | 1e-3 | 1e-5 |
| epochs máx. | 10 | 30 |
| early stopping (patience) | 3 | 5 |
| ReduceLROnPlateau | — | factor 0,2, patience 2, mínimo 1e-7 |

- Optimizador Adam, loss `binary_crossentropy`, mixed precision `float16`.
- Métrica monitorizada: **`val_pr_auc`**. Se guardan los mejores pesos de las dos fases en conjunto y se cargan al final.
- La fase 2 parte de los mejores pesos de la fase 1.
- **Sin data augmentation:** las imágenes se usan tal cual. Es una de las razones de la sensibilidad del modelo a la
  compresión JPEG ([09](09_serving.md#limitaciones-conocidas)).

## Umbral de decisión

No se usa 0,5. El umbral se elige **en val** como el de mayor precisión entre los que alcanzan **recall ≥ 95 %**
(`threshold.target_recall`) y se aplica sin cambios a test. La idea es priorizar no perder incendios. `metrics.json`
también incluye `test_at_0.5` como referencia.

## Salidas — `results/<experimento>/`

Todo lo de esta tabla, salvo `best.weights.h5`, se guarda también en el run de MLflow, y el modelo se registra como
versión nueva de `fire-detector` ([07](07_experiment_tracking.md)).

| fichero | contenido |
|---|---|
| `config.json` | copia de la config usada |
| `model.keras` | **modelo completo**: arquitectura, pesos, estado del optimizador y compile |
| `best.weights.h5` | solo los pesos (y el optimizador); necesita `build_model(cfg)` para usarse |
| `history.csv` | métricas por epoch de las dos fases |
| `metrics.json` | umbral, métricas de val/test, tasa de alarma por subtipo, velocidad en GPU |
| `meta.json` | lo que necesita la inferencia además del modelo: umbral y preparación (CLAHE, tamaño, interpolación) |
| `predictions_{val,test}.csv` | probabilidad por imagen (la columna `alarm` es la etiqueta real) |
| `curves.png`, `confusion_matrix.png`, `metrics.png` | gráficas |

En los dos ficheros de modelo, **~75 % del tamaño es el estado de Adam** (por ejemplo, en MobileNet hay 12 MB de pesos
y 39,5 MB de optimizador). Para inferencia no hace falta, y desaparece al exportar.

## Problemas conocidos

**EfficientNetB4: `Killed` al empezar la fase 2.**

- **Causa** (en `dmesg`: `ptxas invoked oom-killer`): Keras 3 activa por defecto XLA (`jit_compile="auto"`) en GPU. Al
  descongelar el backbone, el grafo de entrenamiento de B4 a 380 px es muy grande, y compilarlo con `ptxas` agota la
  RAM de WSL (~7,7 GiB). Es un OOM de **RAM del sistema**, no de memoria de la GPU.
- **Solución aplicada:** `jit_compile=False` en `compile_model` ([model.py](../src/train/model.py)). El entrenamiento
  **sigue en la GPU** con los kernels de cuDNN y cuBLAS; solo se desactiva la fusión de operaciones de XLA.
- **Alternativas:** subir la memoria de WSL en `.wslconfig` (`memory=12GB`, `swap=8GB`) o bajar el `batch_size`.

**Poca RAM durante la fase 2.** Con ~7,7 GiB en WSL, el entrenamiento compite con MLflow y con el editor (Pylance
llega a 1,8 GB). Si el proceso se queda parado al empezar la fase 2, cerrar o reiniciar el editor, o subir `memory=`
en `.wslconfig`. Un run interrumpido queda en `RUNNING` en MLflow: ver
[07](07_experiment_tracking.md#operaciones).

## Entorno

TensorFlow 2.21, Keras 3.15, GPU RTX 3070 (8 GB), WSL2 con ~7,7 GiB de RAM.

- `requirements.txt`: dependencias directas, con las versiones fijadas.
- `requirements-lock.txt`: el entorno completo (`pip freeze`), para reproducirlo exactamente con
  `pip install -r requirements-lock.txt`.
