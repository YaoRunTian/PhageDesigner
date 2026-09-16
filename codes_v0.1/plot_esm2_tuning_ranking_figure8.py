#!/usr/bin/env python3
"""Create Figure 8: ranking and factorial heatmaps for 18 ESM2-650M grids."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

# Server source:
# results/07_esm2_650m_fixed_split_tuning/grid_summary.json
# All values are from the fixed validation_30k split; final_test was not read.
RESULTS = [
    (1, "cls", 1e-4, "none", 0.5867906446),
    (2, "cls", 1e-4, "inverse_sqrt", 0.5930887330),
    (3, "cls", 1e-4, "balanced", 0.5771320253),
    (4, "cls", 3e-4, "none", 0.6203092396),
    (5, "cls", 3e-4, "inverse_sqrt", 0.6316688144),
    (6, "cls", 3e-4, "balanced", 0.5982984866),
    (7, "cls", 1e-3, "none", 0.6475259200),
    (8, "cls", 1e-3, "inverse_sqrt", 0.6411785866),
    (9, "cls", 1e-3, "balanced", 0.6251941384),
    (10, "masked_mean", 1e-4, "none", 0.6271859566),
    (11, "masked_mean", 1e-4, "inverse_sqrt", 0.6288144918),
    (12, "masked_mean", 1e-4, "balanced", 0.6085958858),
    (13, "masked_mean", 3e-4, "none", 0.6589527945),
    (14, "masked_mean", 3e-4, "inverse_sqrt", 0.6625668600),
    (15, "masked_mean", 3e-4, "balanced", 0.6440501547),
    (16, "masked_mean", 1e-3, "none", 0.6871655982),
    (17, "masked_mean", 1e-3, "inverse_sqrt", 0.6772288963),
    (18, "masked_mean", 1e-3, "balanced", 0.6563163662),
]

POOL_COLORS = {"cls": "#0bdacd", "masked_mean": "#163179"}
WEIGHT_MARKERS = {"none": "o", "inverse_sqrt": "s", "balanced": "^"}
LR_LABELS = {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}
POOL_LABELS = {"cls": "CLS", "masked_mean": "mean"}
WEIGHTS = ["none", "inverse_sqrt", "balanced"]
LRS = [1e-4, 3e-4, 1e-3]


def add_heatmap(ax, pooling, title, cmap, vmin, vmax):
    matrix = np.full((len(LRS), len(WEIGHTS)), np.nan)
    for grid_id, pool, lr, weighting, macro_f1 in RESULTS:
        if pool != pooling:
            continue
        row = LRS.index(lr)
        col = WEIGHTS.index(weighting)
        matrix[row, col] = macro_f1

    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(WEIGHTS)), labels=["none", "inverse_sqrt", "balanced"])
    ax.set_yticks(range(len(LRS)), labels=[LR_LABELS[x] for x in LRS])
    ax.tick_params(axis="both", labelsize=9.5, length=0)
    ax.set_xlabel("Class weighting", fontsize=10.5, labelpad=7)
    ax.set_ylabel("Head learning rate", fontsize=10.5, labelpad=7)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#0B1F33", pad=10)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if value >= 0.645 else "#0B1F33"
            ax.text(
                col,
                row - 0.05,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )

    if pooling == "masked_mean":
        # Grid 16: lr=1e-3, weight=none.
        ax.add_patch(Rectangle((-0.48, 1.52), 0.96, 0.96, fill=False,
                               edgecolor="#C62828", linewidth=2.8))

    for spine in ax.spines.values():
        spine.set_color("#B8C5D1")
        spine.set_linewidth(0.9)
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list(
        "phage_designer", ["#eeffe6", "#0bdacd", "#163179"], N=256
    )

    ranked = sorted(RESULTS, key=lambda item: item[4], reverse=True)
    scores = np.array([item[4] for item in ranked])
    labels = [
        f"{POOL_LABELS[pool]:<4}  {LR_LABELS[lr]:<4}  {weight}"
        for grid_id, pool, lr, weight, _ in ranked
    ]
    y = np.arange(len(ranked))

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.78, 1.0), height_ratios=(1, 1),
        left=0.075, right=0.965, bottom=0.175, top=0.945,
        wspace=0.32, hspace=0.46,
    )
    rank_ax = fig.add_subplot(grid[:, 0])
    cls_ax = fig.add_subplot(grid[0, 1])
    mean_ax = fig.add_subplot(grid[1, 1])

    x_min = 0.565
    for row, item in enumerate(ranked):
        grid_id, pooling, _, weighting, score = item
        color = POOL_COLORS[pooling]
        marker = WEIGHT_MARKERS[weighting]
        rank_ax.hlines(row, x_min, score, color=color, alpha=0.28, linewidth=2.2)
        rank_ax.scatter(
            score,
            row,
            s=92 if grid_id != 16 else 155,
            marker=marker,
            facecolor=color,
            edgecolor="#C62828" if grid_id == 16 else "white",
            linewidth=2.2 if grid_id == 16 else 1.2,
            zorder=3,
        )
        rank_ax.text(
            score + 0.0022,
            row,
            f"{score:.4f}",
            va="center",
            ha="left",
            fontsize=9.2,
            fontweight="bold" if grid_id == 16 else "normal",
            color="#0B1F33",
        )

    rank_ax.set_yticks(y, labels=labels)
    rank_ax.invert_yaxis()
    rank_ax.set_xlim(x_min, 0.701)
    rank_ax.set_xlabel("Validation Macro-F1", fontsize=12.5, labelpad=10)
    rank_ax.set_title(
        "Ranking of all 18 configurations",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color="#0B1F33",
        pad=13,
    )
    rank_ax.tick_params(axis="y", labelsize=8.9, length=0, pad=8)
    rank_ax.tick_params(axis="x", labelsize=10)
    rank_ax.grid(axis="x", color="#D7E0E8", linewidth=0.8)
    rank_ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        rank_ax.spines[side].set_visible(False)
    rank_ax.spines["bottom"].set_color("#9BAABA")

    pooling_handles = [
        Line2D(
            [0], [0], linestyle="-", linewidth=4.5,
            color=POOL_COLORS[pool], solid_capstyle="round", label=label,
        )
        for pool, label in (("cls", "CLS"), ("masked_mean", "masked_mean"))
    ]
    weighting_handles = [
        Line2D(
            [0], [0], marker=WEIGHT_MARKERS[weight], linestyle="none",
            markersize=8, markerfacecolor="#667788", markeredgecolor="white",
            label=weight,
        )
        for weight in ("none", "inverse_sqrt", "balanced")
    ]
    pooling_legend = rank_ax.legend(
        handles=pooling_handles,
        title="Pooling color",
        loc="upper left",
        bbox_to_anchor=(0.00, -0.105),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.4,
        fontsize=9.5,
        title_fontsize=9.5,
        borderaxespad=0,
    )
    rank_ax.add_artist(pooling_legend)
    rank_ax.legend(
        handles=weighting_handles,
        title="Class-weight marker",
        loc="upper left",
        bbox_to_anchor=(0.37, -0.105),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
        fontsize=9.2,
        title_fontsize=9.5,
        borderaxespad=0,
    )

    vmin = min(score for *_, score in RESULTS)
    vmax = max(score for *_, score in RESULTS)
    image = add_heatmap(cls_ax, "cls", "CLS pooling", cmap, vmin, vmax)
    add_heatmap(mean_ax, "masked_mean", "Masked-mean pooling", cmap, vmin, vmax)

    colorbar = fig.colorbar(image, ax=[cls_ax, mean_ax], fraction=0.032, pad=0.04)
    colorbar.set_label("Validation Macro-F1", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    fig.savefig(OUT / "figure8.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "figure8.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / 'figure8.png'}")
    print(f"Wrote {OUT / 'figure8.pdf'}")


if __name__ == "__main__":
    main()
