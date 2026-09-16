#!/usr/bin/env python3
"""Plot count and row-normalized annotations for the three ESM-2 models."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "esm2_confusion_matrices"
MODELS = [
    ("ESM2-8M", ROOT / "results/05b_esm2_8m_100k_head_rescue/confusion_matrix.tsv"),
    ("ESM2-650M", ROOT / "results/05c_esm2_650m_100k_frozen_head/confusion_matrix.tsv"),
    ("ESM2-3B", ROOT / "results/05d_esm2_3b_100k_frozen_head/confusion_matrix.tsv"),
]
LABELS = ["assembly", "replication", "infection", "packaging", "integration",
          "regulation", "lysis", "immune", "tRNA_related"]

def read_matrix(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")[1:]
    if header != LABELS:
        raise ValueError(f"Unexpected labels in {path}: {header}")
    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        if fields[0] not in LABELS:
            raise ValueError(f"Unexpected row label: {fields[0]}")
        rows.append([int(x) for x in fields[1:]])
    matrix = np.asarray(rows, dtype=int)
    if matrix.shape != (len(LABELS), len(LABELS)):
        raise ValueError(f"Unexpected matrix shape: {matrix.shape}")
    return matrix

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrices = [read_matrix(path) for _, path in MODELS]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), constrained_layout=True)
    cmap = plt.get_cmap("Blues")
    for ax, (name, _), matrix in zip(axes, MODELS, matrices):
        row_sum = matrix.sum(axis=1, keepdims=True)
        frac = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum != 0)
        im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=0, vmax=max(m.max() for m in matrices))
        threshold = im.norm.vmax * 0.45
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if matrix[i, j] > threshold else "#16213e"
                ax.text(j, i, f"{matrix[i,j]:,}\n{frac[i,j]*100:.1f}%", ha="center", va="center",
                        fontsize=7.2, color=color)
        ax.set_title(name, fontsize=15, fontweight="bold", pad=12)
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(LABELS)), LABELS, fontsize=9)
        ax.set_xlabel("Predicted function", fontsize=12, labelpad=9)
        ax.set_ylabel("True function", fontsize=12, labelpad=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("ESM-2 single-function classification confusion matrices", fontsize=17, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02)
    cbar.set_label("Number of proteins", fontsize=11)
    fig.savefig(OUT / "esm2_confusion_matrices.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "esm2_confusion_matrices.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / 'esm2_confusion_matrices.png'}")
    print(f"Wrote {OUT / 'esm2_confusion_matrices.pdf'}")

if __name__ == "__main__":
    main()
