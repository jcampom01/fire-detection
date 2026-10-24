"""Experiment config: read the JSON file and resolve its paths against the project root."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    for key in ("labels_dir", "image_root"):
        cfg["data"][key] = PROJECT_ROOT / cfg["data"][key]
    cfg["output_dir"] = PROJECT_ROOT / cfg["output_dir"] / cfg["experiment"]
    return cfg
