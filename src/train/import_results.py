"""Import trainings made before the MLflow integration (results/<experiment>/) as MLflow runs.

Each folder becomes a run with the same structure as one logged by train.py (params, per-epoch and final
metrics, artifacts, meta.json, registered model), so all runs are comparable in the UI. Differences:
    - the code version is unknown: tag mlflow.source.git.commit = unknown, no git_dirty
    - tags imported=true and imported_from=results/<experiment>
    - start/end time = when the training really happened (model.keras mtime minus train_minutes)

Idempotent: a folder already imported (same imported_from tag) is skipped.

Usage:
    python src/train/import_results.py results/mobilenetv3large_clahe results/efficientnetb4_clahe
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from tf_setup import keras  # first: preloads the CUDA libraries before TensorFlow is imported

from tracking import connect, flatten, log_data_source, log_model, log_results, model_metadata, run_tags  # noqa: E402

import mlflow  # noqa: E402  (after tracking, which silences MLflow's startup hint)
from config import PROJECT_ROOT, load_config  # noqa: E402
from mlflow import MlflowClient  # noqa: E402


def already_imported(client, experiment_id, source):
    runs = client.search_runs([experiment_id], f"tags.imported_from = '{source}'")
    return runs[0].info.run_id if runs else None


def import_folder(client, experiment_id, folder):
    folder = folder.resolve()
    source = str(folder.relative_to(PROJECT_ROOT))
    if run_id := already_imported(client, experiment_id, source):
        print(f"{source}: already imported (run {run_id}), skipped")
        return

    cfg = load_config(folder / "config.json")
    results = json.loads((folder / "metrics.json").read_text())
    history = pd.read_csv(folder / "history.csv")

    # Real training time: the model is saved right after training
    end_ms = int((folder / "model.keras").stat().st_mtime * 1000)
    start_ms = end_ms - int(results["train_minutes"] * 60_000)

    tags = {**run_tags(cfg), "imported": "true", "imported_from": source, "mlflow.source.git.commit": "unknown"}
    # The low-level client needs string values (mlflow.set_tags converts them, create_run does not)
    run = client.create_run(experiment_id, start_time=start_ms, run_name=cfg["experiment"],
                            tags={k: str(v) for k, v in tags.items()})
    run_id = run.info.run_id
    print(f"{source}: run {run_id}")

    with mlflow.start_run(run_id=run_id):
        # Params: config + split sizes (from the label CSVs; these trainings did not use debug_max_images)
        mlflow.log_params(flatten(json.loads((folder / "config.json").read_text())))
        mlflow.log_params({f"n_{s}": len(pd.read_csv(cfg["data"]["labels_dir"] / f"{s}.csv"))
                           for s in ("train", "val", "test")})
        log_data_source()

        # Per-epoch metrics, same step as train.py (global epoch 1..N); empty cells (e.g. lr in phase 1) skipped
        for _, row in history.iterrows():
            values = {k: float(v) for k, v in row.drop(["phase", "epoch"]).items() if pd.notna(v)}
            mlflow.log_metrics(values, step=int(row["epoch"]))

        meta = model_metadata(cfg, results)
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        log_results(results, folder)

        print("   uploading the model...")
        model = keras.models.load_model(folder / "model.keras")
        log_model(model, cfg, meta)

    client.set_terminated(run_id, "FINISHED", end_time=end_ms)
    print(f"   done: {len(history)} epochs, test.recall={results['test']['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="+", help="results/<experiment> folders to import")
    args = parser.parse_args()

    _, experiment = connect()
    client = MlflowClient()
    for folder in args.folders:
        import_folder(client, experiment.experiment_id, folder)


if __name__ == "__main__":
    main()
