"""MLflow tracking: one run per training with params, per-epoch and final metrics, artifacts and the model.

The server address comes from MLFLOW_TRACKING_URI (default http://localhost:5000). The server is checked
before training starts, so a long training is never lost because tracking was down.

Run contents:
    params      the JSON config, flattened (phase1.lr, data.image_root...) + split sizes
    tags        model, dataset, image_size, git_dirty, debug, data.source_md5 (DVC hash of the source
                dataset) (+ mlflow.source.git.commit, set by MLflow)
    metrics     every Keras metric per epoch (step = global epoch across both phases)
                + metrics.json flattened (test.recall, val.threshold, speed.images_per_second...)
    artifacts   config, history, metrics, meta.json, plots/, predictions/, data/original_dataset.dvc
                (+ git_diff.patch if dirty)
    model       Keras model with the inference metadata, registered as a new version of fire-detector
                (not registered for debug runs; aliases are set by promote.py, never here)
"""

import json
import os
import urllib.request
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # anonymous usage data sent to MLflow

import git  # noqa: E402
import mlflow  # noqa: E402
import mlflow.keras  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from mlflow.models import ModelSignature  # noqa: E402
from mlflow.types import Schema, TensorSpec  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402
from tf_setup import keras, tf  # noqa: E402

EXPERIMENT = "fire-detection"
REGISTERED_MODEL = "fire-detector"
DEFAULT_TRACKING_URI = "http://localhost:5000"
CLAHE_CONFIG = PROJECT_ROOT / "src/preprocess/clahe_config.json"
# Source dataset versioned with DVC. The derived datasets (labels, CLAHE, resized) are not versioned:
# they follow from this one + the preprocessing configs at the logged git commit.
DVC_SOURCE = PROJECT_ROOT / "data/original_dataset.dvc"

# results/<exp>/ file -> artifact folder in the run
ARTIFACTS = {
    "config.json": "",
    "history.csv": "",
    "metrics.json": "",
    "meta.json": "",
    "curves.png": "plots",
    "metrics.png": "plots",
    "confusion_matrix.png": "plots",
    "predictions_val.csv": "predictions",
    "predictions_test.csv": "predictions",
}


def flatten(d, prefix=""):
    """{"a": {"b": 1}} -> {"a.b": 1}"""
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


# --- Run ---

def check_server(uri):
    if not uri.startswith("http"):
        return
    try:
        urllib.request.urlopen(f"{uri}/health", timeout=5)
    except OSError as e:
        raise SystemExit(f"MLflow not reachable at {uri} ({e}).\n"
                         f"Start it with: cd deploy/mlops && docker compose up -d") from None


def git_state():
    """(dirty, diff against HEAD). Only tracked files count: untracked ones are ignored."""
    repo = git.Repo(PROJECT_ROOT)
    dirty = repo.is_dirty()
    return dirty, repo.git.diff("HEAD") if dirty else ""


def log_data_source():
    """Tag the DVC hash of the source dataset and keep its .dvc file in the run."""
    if not DVC_SOURCE.is_file():
        print(f"WARNING: {DVC_SOURCE.name} not found, data version not logged")
        return
    out = yaml.safe_load(DVC_SOURCE.read_text())["outs"][0]
    mlflow.set_tags({"data.source": f"data/{out['path']}", "data.source_md5": out["md5"]})
    mlflow.log_artifact(str(DVC_SOURCE), artifact_path="data")


def connect():
    """Point the client at the server (checked first) and activate the experiment. Returns (uri, experiment)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    check_server(uri)
    mlflow.set_tracking_uri(uri)
    return uri, mlflow.set_experiment(EXPERIMENT)


def run_tags(cfg):
    return {"model": cfg["model"], "dataset": cfg["dataset"], "image_size": cfg["image_size"],
            "debug": bool(cfg.get("debug_max_images"))}


@contextmanager
def start_run(cfg, config_path):
    """Open the run and log what is known before training: tags, params and the git state."""
    uri, _ = connect()

    with mlflow.start_run(run_name=cfg["experiment"]) as run:
        dirty, diff = git_state()
        if dirty:
            print("WARNING: uncommitted changes, the git commit does not match the code that runs "
                  "(diff saved as git_diff.patch)")
            mlflow.log_text(diff, "git_diff.patch")
        mlflow.set_tags({**run_tags(cfg), "git_dirty": dirty})
        log_data_source()
        mlflow.log_params(flatten(json.loads(Path(config_path).read_text())))
        print(f"MLflow run: {uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        yield run


def log_split_sizes(splits):
    mlflow.log_params({f"n_{split}": len(df) for split, df in splits.items()})


class EpochLogger(keras.callbacks.Callback):
    """Log the Keras metrics of every epoch. The step keeps counting across both phases (1..N),
    like the epoch column of history.csv."""

    def __init__(self):
        super().__init__()
        self.step = 0

    def on_epoch_end(self, epoch, logs=None):
        self.step += 1
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=self.step)


# --- Results ---

def model_metadata(cfg, results):
    """What inference needs besides the weights: threshold and the exact preprocessing."""
    clahe = None
    if cfg["dataset"] == "clahe":
        clahe = {**json.loads(CLAHE_CONFIG.read_text())["clahe"],
                 "channel": "L of CIELAB", "applied": "at full resolution, before resizing"}
    return {
        "threshold": results["test"]["threshold"],
        "target_recall": cfg["threshold"]["target_recall"],
        "image_size": cfg["image_size"],
        "dataset": cfg["dataset"],
        "preprocessing": {
            "clahe": clahe,
            "resize": {"size": cfg["image_size"], "interpolation": "cv2.INTER_AREA", "keep_aspect_ratio": False},
            "input": "RGB, float32 in 0-255 (the model normalizes internally)",
        },
        "output": "probability of alarm (fire and/or smoke); alarm if prob >= threshold",
    }


def log_results(results, out):
    """Final metrics (numbers of metrics.json, flattened) and the files of results/<exp>/."""
    metrics = flatten(results)
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for name, folder in ARTIFACTS.items():
        mlflow.log_artifact(str(out / name), artifact_path=folder or None)


def log_model(model, cfg, meta):
    """Log the Keras model with its metadata and register it as a new version (except debug runs)."""
    size = cfg["image_size"]
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, size, size, 3), "image")]),
        outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 1), "alarm")]),
    )
    return mlflow.keras.log_model(
        model,
        name="model",
        signature=signature,
        metadata=meta,
        pip_requirements=[f"tensorflow=={tf.__version__}", f"keras=={keras.__version__}"],
        registered_model_name=None if cfg.get("debug_max_images") else REGISTERED_MODEL,
    )
