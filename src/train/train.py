"""Train a binary fire/smoke alarm classifier with transfer learning (TensorFlow / Keras).

Task:
    positive (alarm=1) -> the image contains fire and/or smoke  (label smoke or fire in the CSVs)
    negative (alarm=0) -> no_fire

Training runs in two phases:
    phase 1  backbone frozen, only the new head is trained
    phase 2  the last part of the backbone is unfrozen and fine-tuned with a low learning rate
             (BatchNorm layers stay frozen)

The best weights over both phases (by the monitored validation metric) are kept. The decision
threshold is then chosen on the validation set as the one with the best precision among those
that reach the target recall, and applied unchanged to the test set.

Code layout (src/train/):
    train.py        this entry point: runs the steps below in order
    tf_setup.py     TensorFlow import (preloads the CUDA libraries), seeds, mixed precision, GPU
    config.py       reads the JSON config and resolves its paths
    data.py         split CSVs -> binary labels and subtypes; tf.data pipeline
    model.py        backbones, binary head, unfreezing and compilation
    trainer.py      two-phase training with checkpointing of the best weights
    evaluation.py   predictions, threshold, metrics, per-subtype rates, inference speed
    plots.py        training curves, confusion matrix and metrics overview
    tracking.py     MLflow run: params, metrics, artifacts and the registered model

Every parameter comes from a JSON config (see src/train/configs/). Each training is an MLflow run
(experiment fire-detection; requires the server: cd deploy/mlops && docker compose up -d), and the
results are also written to results/<experiment>/:
    config.json             copy of the config used
    best.weights.h5         best weights
    model.keras             full model with the best weights
    history.csv             per-epoch metrics of both phases
    curves.png              training curves
    confusion_matrix.png    test confusion matrix at the chosen threshold
    metrics.png             ROC and PR curves, probability histogram, metrics, per-subtype rates
    metrics.json            threshold, val/test metrics, recall per subtype, inference speed
    meta.json               what inference needs: threshold and preprocessing (CLAHE, size)
    predictions_val.csv     per-image probabilities (val)
    predictions_test.csv    per-image probabilities (test)

Usage:
    python src/train/train.py --config src/train/configs/mobilenetv3large_original.json
    MLFLOW_TRACKING_URI=http://host:5000 python src/train/train.py --config ...   # other server
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from tf_setup import setup_runtime  # first: preloads the CUDA libraries before TensorFlow is imported

from config import load_config  # noqa: E402
from data import load_split, make_dataset  # noqa: E402
from evaluation import evaluate  # noqa: E402
from model import build_model  # noqa: E402
from plots import plot_confusion, plot_curves, plot_metrics  # noqa: E402
from tracking import EpochLogger, log_model, log_results, log_split_sizes, model_metadata, start_run  # noqa: E402
from trainer import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="JSON config of the experiment")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.json")

    # Run ends as FINISHED, or FAILED if anything below raises
    with start_run(cfg, args.config):
        run(cfg, out)


def run(cfg, out):
    gpus = setup_runtime(cfg["seed"], cfg["mixed_precision"])
    print(f"experiment {cfg['experiment']}  |  model {cfg['model']}  |  images {cfg['data']['image_root']}")
    print(f"GPU: {[g.name for g in gpus] or 'none, running on CPU'}")

    # Data
    splits = {s: load_split(s, cfg) for s in ("train", "val", "test")}
    for s, df in splits.items():
        print(f"{s:<5} {len(df):>6,} images  |  alarm={df['alarm'].mean():.1%}")
    log_split_sizes(splits)
    train_ds = make_dataset(splits["train"], cfg, training=True)
    val_ds = make_dataset(splits["val"], cfg, training=False)

    # Training
    model, backbone = build_model(cfg)
    t0 = time.perf_counter()
    history = train(cfg, model, backbone, train_ds, val_ds, callbacks=[EpochLogger()])
    train_minutes = (time.perf_counter() - t0) / 60
    history.to_csv(out / "history.csv", index=False)
    plot_curves(history, out / "curves.png", cfg["monitor"])
    model.save(out / "model.keras")

    # Evaluation
    results, pred_val, pred_test = evaluate(model, splits, history, cfg, train_minutes)
    pred_val.to_csv(out / "predictions_val.csv", index=False)
    pred_test.to_csv(out / "predictions_test.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    threshold = results["test"]["threshold"]
    plot_confusion(results["test"]["confusion_matrix"], out / "confusion_matrix.png",
                   f"{cfg['experiment']} — test (threshold {threshold:.3f})")
    plot_metrics(results, pred_val, pred_test, out / "metrics.png")
    meta = model_metadata(cfg, results)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Tracking
    log_results(results, out)
    print("\nuploading the model to MLflow...")
    log_model(model, cfg, meta)

    t = results["test"]
    print(f"\nthreshold {threshold:.3f} (val recall >= {cfg['threshold']['target_recall']:.0%})")
    print(f"test  recall={t['recall']:.3f}  precision={t['precision']:.3f}  specificity={t['specificity']:.3f}  "
          f"f1={t['f1']:.3f}  roc_auc={t['roc_auc']:.3f}  pr_auc={t['pr_auc']:.3f}")
    print("alarm rate per subtype: " + "  ".join(f"{k}={v:.3f}" for k, v in results["test_subtype_alarm_rate"].items()))
    print(f"speed {results['speed']['images_per_second']:.0f} img/s  |  trained in {train_minutes:.1f} min  ->  {out}")


if __name__ == "__main__":
    main()
