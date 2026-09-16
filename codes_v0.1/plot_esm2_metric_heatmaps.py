#!/usr/bin/env python3
"""Create six separate 600-dpi heatmaps for ESM-2 model comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/esm2_metric_heatmaps"
MODELS = [
    ("ESM2-8M", "05b_esm2_8m_100k_head_rescue"),
    ("ESM2-650M", "05c_esm2_650m_100k_frozen_head"),
    ("ESM2-3B", "05d_esm2_3b_100k_frozen_head"),
]
CMAP = "YlGnBu"  # low values yellow, high values dark blue, as requested

def load_json_metrics():
    out = {}
    for name, dirname in MODELS:
        with (ROOT / "results" / dirname / "test_metrics.json").open() as f:
            out[name] = json.load(f)
    return out

def load_predictions():
    out = {}
    for name, dirname in MODELS:
        rows = (ROOT / "results" / dirname / "test_predictions.tsv").read_text().splitlines()[1:]
        parsed = [line.split("\t") for line in rows]
        out[name] = (np.array([x[2] for x in parsed]), np.array([x[3] for x in parsed]))
    return out

def save_single(name, value, ylabel, filename, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    arr = np.array([[value[n]] for n, _ in MODELS], dtype=float)
    im = ax.imshow(arr, cmap=CMAP, aspect="auto", vmin=vmin, vmax=vmax)
    for i, val in enumerate(arr[:, 0]):
        ax.text(0, i, f"{val:.4f}", ha="center", va="center", fontsize=13,
                color="white" if val > (im.norm.vmin + im.norm.vmax) / 2 else "#17202a")
    ax.set_yticks(range(len(MODELS)), [n for n, _ in MODELS], fontsize=11)
    ax.set_xticks([0], [ylabel], fontsize=11)
    ax.set_title(ylabel, fontsize=15, fontweight="bold", pad=10)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Value", fontsize=10)
    fig.savefig(OUT / f"{filename}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def save_pairwise(matrix, title, filename, vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap=CMAP, vmin=vmin, vmax=vmax)
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=11,
                    color="white" if val > (vmin + vmax) / 2 else "#17202a")
    labels = [n for n, _ in MODELS]
    ax.set_xticks(range(3), labels, fontsize=11)
    ax.set_yticks(range(3), labels, fontsize=11)
    ax.set_xlabel("Model 2", fontsize=11)
    ax.set_ylabel("Model 1", fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label("Value", fontsize=10)
    fig.savefig(OUT / f"{filename}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_json_metrics()
    preds = load_predictions()
    # Four per-model metrics: each is exported as an individual heatmap.
    for key, title, fn in [
        ("accuracy", "Accuracy", "accuracy"),
        ("macro_f1", "Macro-F1", "macro_f1"),
        ("trainer_test_loss", "Cross-entropy loss", "loss"),
        ("balanced_accuracy", "Balanced accuracy", "balanced_accuracy"),
    ]:
        values = {name: metrics[name][key] for name, _ in MODELS}
        save_single(name=title, value=values, ylabel=title, filename=fn,
                    vmin=min(values.values()), vmax=max(values.values()))
    # Two genuinely pairwise metrics from the aligned test predictions.
    n = len(MODELS)
    agreement = np.eye(n)
    kappa = np.eye(n)
    for i, (name_i, _) in enumerate(MODELS):
        for j, (name_j, _) in enumerate(MODELS):
            yi, pi = preds[name_i]
            yj, pj = preds[name_j]
            if not np.array_equal(yi, yj):
                raise ValueError("Test rows are not aligned across models")
            agreement[i, j] = float(np.mean(pi == pj))
            kappa[i, j] = cohen_kappa_score(pi, pj)
    save_pairwise(agreement, "Prediction agreement", "prediction_agreement")
    save_pairwise(kappa, "Cohen's kappa", "cohen_kappa", vmin=-1.0, vmax=1.0)
    print(f"Wrote six metric heatmaps to {OUT}")

if __name__ == "__main__":
    main()
