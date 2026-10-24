"""Figures saved for each experiment: training curves, test confusion matrix and metrics overview.

Run as a script to (re)build the figures of an existing experiment from its saved files:

    python src/train/plots.py results/mobilenetv3large_clahe
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#1f1f1c", "#6b6b66", "#e4e4df"
SPLIT_COLORS = {"val": BLUE, "test": ORANGE}


def style(ax, title):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_curves(history, path, monitor):
    metrics = [("loss", "Loss"), ("pr_auc", "PR-AUC"), ("recall", "Recall"), ("precision", "Precision")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 3.8))
    phase2_start = history.loc[history["phase"] == "phase2", "epoch"].min()
    for ax, (key, title) in zip(axes, metrics):
        ax.plot(history["epoch"], history[key], color=BLUE, linewidth=2, label="train")
        ax.plot(history["epoch"], history[f"val_{key}"], color=ORANGE, linewidth=2, label="val")
        if not np.isnan(phase2_start):
            ax.axvline(phase2_start - 0.5, color=MUTED, linestyle=":", linewidth=1)
        style(ax, title)
        ax.set_xlabel("epoch")
    best_epoch = history.loc[history[monitor].idxmax(), "epoch"]
    for ax in axes:
        ax.axvline(best_epoch, color=AQUA, linewidth=1)
    axes[0].legend(frameon=False)
    fig.suptitle(f"dotted: start of phase 2  ·  green: best epoch by {monitor}", x=0.01, ha="left",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_confusion(ax, cm, title):
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}\n{v / mat[i].sum():.1%}", ha="center", va="center",
                color="white" if v > mat.max() / 2 else INK)
    ax.set_xticks([0, 1], ["no alarm", "alarm"])
    ax.set_yticks([0, 1], ["no_fire", "fire/smoke"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    draw_confusion(ax, cm, title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_roc(ax, preds, results):
    for split, df in preds.items():
        fpr, tpr, _ = roc_curve(df["alarm"], df["prob"])
        ax.plot(fpr, tpr, color=SPLIT_COLORS[split], linewidth=2,
                label=f"{split}  AUC {results[split]['roc_auc']:.3f}")
    for key, marker in (("test", "o"), ("test_at_0.5", "s")):
        m = results[key]
        ax.plot(m["false_positive_rate"], m["recall"], marker, color=INK, markersize=8,
                markerfacecolor="white", markeredgewidth=2, label=f"test @ {m['threshold']:.3f}")
    # A good model hugs the top-left corner: zoom into it
    ax.set_xlim(-0.005, 0.25)
    ax.set_ylim(0.6, 1.01)
    style(ax, "ROC curve (zoomed)")
    ax.set_xlabel("false positive rate (false alarms)")
    ax.set_ylabel("recall (fires detected)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)


def draw_pr(ax, preds, results):
    for split, df in preds.items():
        precision, recall, _ = precision_recall_curve(df["alarm"], df["prob"])
        ax.plot(recall, precision, color=SPLIT_COLORS[split], linewidth=2,
                label=f"{split}  AP {results[split]['pr_auc']:.3f}")
    ax.axvline(results["target_recall"], color=MUTED, linestyle=":", linewidth=1,
               label=f"target recall {results['target_recall']:.0%}")
    for key, marker in (("test", "o"), ("test_at_0.5", "s")):
        m = results[key]
        ax.plot(m["recall"], m["precision"], marker, color=INK, markersize=8,
                markerfacecolor="white", markeredgewidth=2, label=f"test @ {m['threshold']:.3f}")
    # A good model hugs the top-right corner: zoom into it
    ax.set_xlim(0.6, 1.01)
    ax.set_ylim(0.6, 1.01)
    style(ax, "Precision-recall curve (zoomed)")
    ax.set_xlabel("recall (fires detected)")
    ax.set_ylabel("precision (alarms that are real)")
    ax.legend(frameon=False, loc="lower left", fontsize=8)


def draw_prob_hist(ax, pred_test, threshold):
    bins = np.linspace(0, 1, 41)
    for alarm, name, color in ((0, "no_fire", AQUA), (1, "fire/smoke", VIOLET)):
        ax.hist(pred_test.loc[pred_test["alarm"] == alarm, "prob"], bins=bins, histtype="step", color=color,
                linewidth=2, label=f"true {name}")
    ax.axvline(threshold, color=INK, linewidth=1.5, label=f"threshold {threshold:.3f}")
    ax.axvline(0.5, color=MUTED, linestyle=":", linewidth=1.5, label="0.5")
    ax.set_yscale("log")
    ax.set_ylim(0.7, ax.get_ylim()[1] * 8)  # headroom for the legend
    style(ax, "Predicted probability (test)")
    ax.set_xlabel("predicted probability of alarm")
    ax.set_ylabel("images (log scale)")
    ax.legend(frameon=False, loc="upper center", fontsize=8)


def draw_metric_dots(ax, results):
    names = ["accuracy", "precision", "recall", "specificity", "f1"]
    y = np.arange(len(names))[::-1]
    values = []
    for split, offset in (("val", 0.12), ("test", -0.12)):
        v = [results[split][n] for n in names]
        values += v
        ax.plot(v, y + offset, "o", color=SPLIT_COLORS[split], markersize=9, label=split)
        for xi, yi in zip(v, y + offset):
            ax.text(xi, yi, f"  {xi:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names)
    low = np.floor(min(values) * 50) / 50 - 0.02  # dot plot: the axis need not start at 0
    ax.set_xlim(max(low, 0), 1.0 + (1.0 - low) * 0.15)
    style(ax, f"Metrics at threshold {results['test']['threshold']:.3f}")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left", fontsize=8)


def draw_subtypes(ax, pred_test, results):
    rates = results["test_subtype_alarm_rate"]
    counts = pred_test["subtype"].value_counts()
    names = list(rates)[::-1]
    ax.barh(names, [rates[n] for n in names], color=ORANGE, height=0.6)
    for i, n in enumerate(names):
        ax.text(rates[n], i, f" {rates[n]:.1%}  (n={counts.get(n, 0):,})", va="center", fontsize=8, color=INK)
    ax.axvline(results["target_recall"], color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    style(ax, "Alarm rate per subtype (test)")
    ax.set_xlabel("share of images that raise the alarm\n(background: false alarms, lower is better)")
    ax.grid(axis="y", visible=False)


def plot_metrics(results, pred_val, pred_test, path):
    """Overview of the evaluation: ROC, PR, probability histogram, metrics, subtypes, confusion matrix."""
    preds = {"val": pred_val, "test": pred_test}
    threshold = results["test"]["threshold"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    draw_roc(axes[0, 0], preds, results)
    draw_pr(axes[0, 1], preds, results)
    draw_prob_hist(axes[0, 2], pred_test, threshold)
    draw_metric_dots(axes[1, 0], results)
    draw_subtypes(axes[1, 1], pred_test, results)
    draw_confusion(axes[1, 2], results["test"]["confusion_matrix"], f"Confusion matrix (test @ {threshold:.3f})")
    s = results["speed"]
    fig.suptitle(f"{results['experiment']}  ·  best epoch {results['best_epoch']}/{results['epochs_run']}  ·  "
                 f"{results['train_minutes']:.1f} min  ·  {s['params_millions']:.1f} M params  ·  "
                 f"{s['images_per_second']:.0f} img/s", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="experiment folder, e.g. results/mobilenetv3large_clahe")
    args = parser.parse_args()

    out = args.results_dir
    results = json.loads((out / "metrics.json").read_text())
    history = pd.read_csv(out / "history.csv")
    monitor = json.loads((out / "config.json").read_text())["monitor"]
    pred_val, pred_test = pd.read_csv(out / "predictions_val.csv"), pd.read_csv(out / "predictions_test.csv")

    threshold = results["test"]["threshold"]
    plot_curves(history, out / "curves.png", monitor)
    plot_confusion(results["test"]["confusion_matrix"], out / "confusion_matrix.png",
                   f"{results['experiment']} — test (threshold {threshold:.3f})")
    plot_metrics(results, pred_val, pred_test, out / "metrics.png")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
