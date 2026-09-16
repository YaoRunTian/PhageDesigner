#!/usr/bin/env python3
"""Generate Figures 1-4: three standard confusion matrices per figure."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/esm2_confusion_figures_1_4"
MODELS = [
    ("ESM2-8M", "05b_esm2_8m_100k_head_rescue"),
    ("ESM2-650M", "05c_esm2_650m_100k_frozen_head"),
    ("ESM2-3B", "05d_esm2_3b_100k_frozen_head"),
]
LABELS = ["assembly", "replication", "infection", "packaging", "integration",
          "regulation", "lysis", "immune", "tRNA_related"]
FIG_METRICS = [
    ("accuracy", "Accuracy", "accuracy"),
    ("macro_f1", "Macro-F1", "macro_f1"),
    ("macro_precision", "Macro-Precision", "macro_precision"),
    ("macro_recall", "Macro-Recall", "macro_recall"),
]
ACCURACY_CMAP = LinearSegmentedColormap.from_list(
    "accuracy_green_cyan_blue", ["#eeffe6", "#0bdacd", "#163179"]
)

def read_matrix(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[0].split("\t")[1:] != LABELS:
        raise ValueError(f"Unexpected header in {path}")
    matrix = np.asarray([[int(v) for v in line.split("\t")[1:]] for line in lines[1:]], dtype=int)
    if matrix.shape != (9, 9):
        raise ValueError(f"Unexpected matrix shape: {matrix.shape}")
    return matrix

def load_data():
    matrices, metrics = {}, {}
    for model, dirname in MODELS:
        base = ROOT / "results" / dirname
        matrices[model] = read_matrix(base / "confusion_matrix.tsv")
        raw = json.loads((base / "test_metrics.json").read_text())
        per_class = raw["per_class"]
        metrics[model] = {
            "accuracy": raw["accuracy"],
            "macro_f1": raw["macro_f1"],
            "macro_precision": float(np.mean([x["precision"] for x in per_class.values()])),
            "macro_recall": float(np.mean([x["recall"] for x in per_class.values()])),
        }
    return matrices, metrics

def make_figure(metric_key, metric_title, suffix, figure_number, matrices, metrics):
    vmax = max(int(m.max()) for m in matrices.values())
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.0), constrained_layout=True)
    im = None
    for ax, (model, _) in zip(axes, MODELS):
        matrix = matrices[model]
        row_sum = matrix.sum(axis=1, keepdims=True)
        frac = matrix / row_sum
        normalized_accuracy = metric_key == "accuracy"
        display_matrix = frac if normalized_accuracy else matrix
        im = ax.imshow(
            display_matrix,
            cmap=ACCURACY_CMAP if normalized_accuracy else "Blues",
            vmin=0,
            vmax=1 if normalized_accuracy else vmax,
            interpolation="nearest",
        )
        for i in range(9):
            for j in range(9):
                val = display_matrix[i, j]
                threshold = 0.48 if normalized_accuracy else vmax * 0.42
                color = "white" if val > threshold else "#132238"
                text = f"{val:.3f}" if normalized_accuracy else f"{int(val):,}\n{frac[i,j]*100:.1f}%"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=7.2, color=color)
        ax.set_title(f"{model}\n{metric_title} = {metrics[model][metric_key]:.4f}",
                     fontsize=14, fontweight="bold", pad=11)
        ax.set_xticks(range(9), LABELS, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(9), LABELS, fontsize=8.5)
        ax.set_xlabel("Predicted function", fontsize=11, labelpad=8)
        ax.set_ylabel("True function", fontsize=11, labelpad=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    heading = (f"Standard confusion matrices ({metric_title})" if figure_number == 1
               else f"Figure {figure_number} — Standard confusion matrices ({metric_title})")
    fig.suptitle(heading,
                 fontsize=17, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02)
    cbar.set_label("Row-normalized value (0–1)" if metric_key == "accuracy" else "Number of proteins",
                   fontsize=11)
    fig.savefig(OUT / f"figure_{suffix}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"figure_{suffix}.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def make_metric_line_figure(raw_metrics, metric_key, metric_label, title, filename):
    fig, ax = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
    x = np.arange(len(LABELS))
    styles = [
        ("ESM2-8M", "#79b9a7", "o"),
        ("ESM2-650M", "#0bdacd", "s"),
        ("ESM2-3B", "#163179", "^"),
    ]
    for model, color, marker in styles:
        values = [raw_metrics[model]["per_class"][label][metric_key] for label in LABELS]
        ax.plot(x, values, color=color, marker=marker, linewidth=2.6, markersize=7.5,
                markeredgecolor="white", markeredgewidth=0.8, label=model)
    ax.set_xticks(x, LABELS, rotation=32, ha="right", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_xlabel("Functional category", fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=17,
                 fontweight="bold", pad=14)
    ax.grid(axis="y", color="#d9e4e1", linewidth=0.8, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, ncol=3, loc="upper left")
    fig.savefig(OUT / f"{filename}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrices, metrics = load_data()
    raw_metrics = {}
    for model, dirname in MODELS:
        raw_metrics[model] = json.loads(
            (ROOT / "results" / dirname / "test_metrics.json").read_text()
        )
    for idx, (key, title, suffix) in enumerate(FIG_METRICS, start=1):
        if idx == 2:
            make_metric_line_figure(raw_metrics, "f1", "F1 score",
                                    "Per-class F1 scores across ESM-2 model sizes",
                                    "figure_2_macro_f1")
        elif idx == 3:
            make_metric_line_figure(raw_metrics, "precision", "Precision",
                                    "Per-class precision across ESM-2 model sizes",
                                    "figure_3_macro_precision")
        elif idx == 4:
            make_metric_line_figure(raw_metrics, "recall", "Recall",
                                    "Per-class recall across ESM-2 model sizes",
                                    "figure_4_macro_recall")
        else:
            make_figure(key, title, f"{idx}_{suffix}", idx, matrices, metrics)
    print(f"Wrote Figures 1-4 to {OUT}")

if __name__ == "__main__":
    main()
