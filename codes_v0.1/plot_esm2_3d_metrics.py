#!/usr/bin/env python3
"""Generate 3D per-class F1, precision, and recall plots for ESM-2 models."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/esm2_confusion_figures_1_4"
LABELS = ["assembly", "replication", "infection", "packaging", "integration",
          "regulation", "lysis", "immune", "tRNA_related"]
MODELS = [
    ("ESM2-8M", "05b_esm2_8m_100k_head_rescue", "#eeffe6", "#79b9a7", "o"),
    ("ESM2-650M", "05c_esm2_650m_100k_frozen_head", "#0bdacd", "#079f99", "s"),
    ("ESM2-3B", "05d_esm2_3b_100k_frozen_head", "#163179", "#163179", "^"),
]
PLOTS = [
    ("accuracy", "Accuracy", "Per-class accuracy across ESM-2 model sizes", "figure_8_3d_accuracy"),
    ("f1", "F1 score", "Per-class F1 scores across ESM-2 model sizes", "figure_5_3d_f1"),
    ("precision", "Precision", "Per-class precision across ESM-2 model sizes", "figure_6_3d_precision"),
    ("recall", "Recall", "Per-class recall across ESM-2 model sizes", "figure_7_3d_recall"),
]

def load_metrics():
    data = {}
    for model, dirname, *_ in MODELS:
        result_dir = ROOT / "results" / dirname
        path = result_dir / "test_metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        per_class = metrics["per_class"]

        # Accuracy is a global metric in test_metrics.json.  For the 3D
        # category plot, use the standard one-vs-rest accuracy for each class:
        # (TP + TN) / total, computed from the saved confusion matrix.
        matrix_path = result_dir / "confusion_matrix.tsv"
        with matrix_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        matrix = np.asarray([[int(value) for value in row[1:]] for row in rows[1:]], dtype=float)
        total = matrix.sum()
        binary_accuracy = {}
        for i, label in enumerate(LABELS):
            tp = matrix[i, i]
            fn = matrix[i, :].sum() - tp
            fp = matrix[:, i].sum() - tp
            tn = total - tp - fn - fp
            binary_accuracy[label] = (tp + tn) / total if total else 0.0

        data[model] = {
            label: {**values, "accuracy": binary_accuracy[label]}
            for label, values in per_class.items()
        }
    return data

def make_plot(data, metric, zlabel, title, filename):
    x = np.arange(len(LABELS), dtype=float)
    fig = plt.figure(figsize=(14.2, 8.4))
    ax = fig.add_subplot(111, projection="3d")

    for y, (model, _, face, edge, marker) in enumerate(MODELS):
        values = np.array([data[model][label][metric] for label in LABELS], dtype=float)
        ax.plot(x, np.full_like(x, y), values, color=edge, linewidth=2.7,
                marker=marker, markersize=6.5, markerfacecolor=face,
                markeredgecolor=edge, markeredgewidth=1.1, zorder=5)
        verts = [list(zip(np.r_[x, x[::-1]], np.r_[values, np.zeros_like(values)[::-1]]))]
        poly = PolyCollection(verts, facecolors=face, edgecolors=edge,
                              linewidths=1.4, alpha=0.42)
        ax.add_collection3d(poly, zs=y, zdir="y")
        for xi, zi in zip(x, values):
            ax.text(xi, y, zi + 0.025, f"{zi:.2f}", fontsize=7.2,
                    ha="center", va="bottom", color="#1d2733")

    ax.set_xlim(-0.35, len(LABELS) - 0.65)
    ax.set_ylim(-0.45, len(MODELS) - 0.55)
    ax.set_zlim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=25, ha="right", fontsize=8.2)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[0] for m in MODELS], fontsize=10)
    ax.set_zticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Functional category", fontsize=12, labelpad=34)
    ax.set_ylabel("Model", fontsize=12, labelpad=12)
    ax.set_zlabel(zlabel, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=7)
    ax.view_init(elev=24, azim=-60)
    ax.set_box_aspect((1.8, 0.72, 0.82))
    ax.xaxis.pane.set_facecolor((0.96, 0.98, 0.98, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.98, 0.98, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.grid(True)
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.10, top=0.91)
    fig.savefig(OUT / f"{filename}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    for args in PLOTS:
        make_plot(data, *args)
    print(f"Wrote Figures 5-8 to {OUT}")

if __name__ == "__main__":
    main()
