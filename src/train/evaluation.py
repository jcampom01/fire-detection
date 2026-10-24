"""Evaluation: predictions, decision threshold, binary metrics, per-subtype rates and inference speed."""

import time

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from data import SUBTYPES, make_dataset
from tf_setup import tf


def predict(model, df, cfg):
    prob = model.predict(make_dataset(df, cfg, training=False), verbose=0).ravel()
    return df[["image", "subtype", "alarm"]].assign(prob=prob)


def choose_threshold(y_true, prob, target_recall):
    """Highest-precision threshold among those with recall >= target_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    precision, recall = precision[:-1], recall[:-1]  # the last point has no threshold
    ok = recall >= target_recall
    if not ok.any():
        return float(thresholds[0])
    idx = np.flatnonzero(ok)
    idx = idx[precision[idx] == precision[idx].max()]
    return float(thresholds[idx].max())


def binary_metrics(y_true, prob, threshold):
    y_pred = (prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    div = lambda a, b: a / b if b else 0.0
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "threshold": threshold,
        "accuracy": div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "specificity": div(tn, tn + fp),
        "false_positive_rate": div(fp, tn + fp),
        "f1": div(2 * precision * recall, precision + recall),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def subtype_rates(df, threshold):
    """Detection rate per subtype: recall for fire/smoke subtypes, false alarm rate for background."""
    pred = df["prob"] >= threshold
    return {s: float(pred[df["subtype"] == s].mean()) for s in SUBTYPES if (df["subtype"] == s).any()}


def inference_speed(model, cfg, n_batches=20):
    """Model-only throughput (no disk I/O) on random inputs, after a warm-up."""
    size, bs = cfg["image_size"], cfg["batch_size"]
    batch = tf.random.uniform((bs, size, size, 3), 0, 255)
    single = batch[:1]
    for _ in range(3):
        model(batch, training=False)
        model(single, training=False)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        model(batch, training=False).numpy()
    throughput = n_batches * bs / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for _ in range(50):
        model(single, training=False).numpy()
    latency_ms = (time.perf_counter() - t0) / 50 * 1000
    return {"images_per_second": throughput, "latency_ms_batch1": latency_ms,
            "params_millions": model.count_params() / 1e6}


def evaluate(model, splits, history, cfg, train_minutes):
    """Predict val and test, choose the threshold on val and compute every metric.

    Returns (results dict for metrics.json, val predictions, test predictions).
    """
    pred_val = predict(model, splits["val"], cfg)
    pred_test = predict(model, splits["test"], cfg)

    # Threshold chosen on val, applied unchanged to test
    threshold = choose_threshold(pred_val["alarm"].values, pred_val["prob"].values,
                                 cfg["threshold"]["target_recall"])
    results = {
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "best_epoch": int(history.loc[history[cfg["monitor"]].idxmax(), "epoch"]),
        "epochs_run": len(history),
        "train_minutes": train_minutes,
        "target_recall": cfg["threshold"]["target_recall"],
        "val": binary_metrics(pred_val["alarm"].values, pred_val["prob"].values, threshold),
        "test": binary_metrics(pred_test["alarm"].values, pred_test["prob"].values, threshold),
        "test_at_0.5": binary_metrics(pred_test["alarm"].values, pred_test["prob"].values, 0.5),
        "test_subtype_alarm_rate": subtype_rates(pred_test, threshold),
        "speed": inference_speed(model, cfg),
    }
    return results, pred_val, pred_test
