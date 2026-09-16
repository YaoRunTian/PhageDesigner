#!/usr/bin/env python3
"""Generate ranking + interaction figures for Accuracy, Macro-Recall and Macro-Precision."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

LABELS = ["accuracy", "macro_recall", "macro_precision"]
METRIC_INFO = {
    "accuracy": ("Validation Accuracy", "figure9_accuracy"),
    "macro_recall": ("Validation Macro-Recall", "figure10_recall"),
    "macro_precision": ("Validation Macro-Precision", "figure11_precision"),
}

# grid_id, pooling, learning_rate, class_weighting, accuracy, macro_recall,
# macro_precision. Values are from the server-side grid_summary.json.
RESULTS = [
    (1, "cls", 1e-4, "none", 0.6420666667, 0.5493263943, 0.6761455692),
    (2, "cls", 1e-4, "inverse_sqrt", 0.6339000000, 0.5866123925, 0.6064663867),
    (3, "cls", 1e-4, "balanced", 0.6069333333, 0.5995807991, 0.5778273242),
    (4, "cls", 3e-4, "none", 0.6630666667, 0.5824894250, 0.7005366998),
    (5, "cls", 3e-4, "inverse_sqrt", 0.6578666667, 0.6068052336, 0.6732628933),
    (6, "cls", 3e-4, "balanced", 0.6337000000, 0.5930131828, 0.6168508929),
    (7, "cls", 1e-3, "none", 0.6733333333, 0.6078856759, 0.7170643827),
    (8, "cls", 1e-3, "inverse_sqrt", 0.6654666667, 0.6389361811, 0.6461159037),
    (9, "cls", 1e-3, "balanced", 0.6478666667, 0.6336308994, 0.6237245159),
    (10, "masked_mean", 1e-4, "none", 0.6742666667, 0.5890651728, 0.7063172156),
    (11, "masked_mean", 1e-4, "inverse_sqrt", 0.6650333333, 0.6227047603, 0.6397777366),
    (12, "masked_mean", 1e-4, "balanced", 0.6380666667, 0.6406844061, 0.5937346448),
    (13, "masked_mean", 3e-4, "none", 0.6945333333, 0.6230753479, 0.7264441702),
    (14, "masked_mean", 3e-4, "inverse_sqrt", 0.6891666667, 0.6543888685, 0.6744546142),
    (15, "masked_mean", 3e-4, "balanced", 0.6691000000, 0.6728444753, 0.6245635270),
    (16, "masked_mean", 1e-3, "none", 0.7154666667, 0.6529341657, 0.7418505728),
    (17, "masked_mean", 1e-3, "inverse_sqrt", 0.6998000000, 0.6758138147, 0.6816165663),
    (18, "masked_mean", 1e-3, "balanced", 0.6794333333, 0.6812895559, 0.6396874148),
]

POOL_COLORS = {"cls": "#0bdacd", "masked_mean": "#163179"}
WEIGHT_MARKERS = {"none": "o", "inverse_sqrt": "s", "balanced": "^"}
LR_LABELS = {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}
POOL_LABELS = {"cls": "CLS", "masked_mean": "mean"}
WEIGHTS = ["none", "inverse_sqrt", "balanced"]
LRS = [1e-4, 3e-4, 1e-3]


def add_heatmap(ax, pooling, metric_index, title, cmap, vmin, vmax, best_grid):
    matrix = np.full((len(LRS), len(WEIGHTS)), np.nan)
    grid_ids = np.zeros_like(matrix, dtype=int)
    for grid_id, pool, lr, weighting, *metrics in RESULTS:
        if pool != pooling:
            continue
        row = LRS.index(lr)
        col = WEIGHTS.index(weighting)
        matrix[row, col] = metrics[metric_index]
        grid_ids[row, col] = grid_id

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
            text_color = "white" if value >= (vmin + vmax) / 2 else "#0B1F33"
            ax.text(
                col, row, f"{value:.4f}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_color,
            )

    best_row = best_col = None
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if grid_ids[row, col] == best_grid:
                best_row, best_col = row, col
    if best_row is not None:
        ax.add_patch(Rectangle((best_col - 0.48, best_row - 0.48), 0.96, 0.96,
                               fill=False, edgecolor="#C62828", linewidth=2.8))

    for spine in ax.spines.values():
        spine.set_color("#B8C5D1")
        spine.set_linewidth(0.9)
    return image


def make_figure(metric_name):
    metric_title, output_stem = METRIC_INFO[metric_name]
    metric_index = {"accuracy": 0, "macro_recall": 1, "macro_precision": 2}[metric_name]
    cmap = LinearSegmentedColormap.from_list(
        "phage_designer", ["#eeffe6", "#0bdacd", "#163179"], N=256
    )

    ranked = sorted(RESULTS, key=lambda item: item[4 + metric_index], reverse=True)
    scores = np.array([item[4 + metric_index] for item in ranked])
    labels = [
        f"{POOL_LABELS[pool]:<4}  {LR_LABELS[lr]:<4}  {weight}"
        for _, pool, lr, weight, *_ in ranked
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

    x_min = max(0.0, np.floor((scores.min() - 0.015) * 100) / 100)
    x_max = min(1.0, np.ceil((scores.max() + 0.012) * 100) / 100)
    for row, item in enumerate(ranked):
        grid_id, pooling, _, weighting, *_ = item
        score = item[4 + metric_index]
        color = POOL_COLORS[pooling]
        marker = WEIGHT_MARKERS[weighting]
        rank_ax.hlines(row, x_min, score, color=color, alpha=0.28, linewidth=2.2)
        rank_ax.scatter(
            score, row, s=92 if grid_id != 16 else 155, marker=marker,
            facecolor=color, edgecolor="#C62828" if grid_id == 16 else "white",
            linewidth=2.2 if grid_id == 16 else 1.2, zorder=3,
        )
        rank_ax.text(score + (x_max - x_min) * 0.015, row, f"{score:.4f}",
                     va="center", ha="left", fontsize=9.2,
                     fontweight="bold" if grid_id == 16 else "normal",
                     color="#0B1F33")

    rank_ax.set_yticks(y, labels=labels)
    rank_ax.invert_yaxis()
    rank_ax.set_xlim(x_min, x_max)
    rank_ax.set_xlabel(metric_title, fontsize=12.5, labelpad=10)
    rank_ax.set_title("Ranking of all 18 configurations", loc="left",
                      fontsize=14, fontweight="bold", color="#0B1F33", pad=13)
    rank_ax.tick_params(axis="y", labelsize=8.9, length=0, pad=8)
    rank_ax.tick_params(axis="x", labelsize=10)
    rank_ax.grid(axis="x", color="#D7E0E8", linewidth=0.8)
    rank_ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        rank_ax.spines[side].set_visible(False)
    rank_ax.spines["bottom"].set_color("#9BAABA")

    pooling_handles = [Line2D([0], [0], linestyle="-", linewidth=4.5,
                              color=POOL_COLORS[pool], solid_capstyle="round",
                              label=label)
                       for pool, label in (("cls", "CLS"), ("masked_mean", "masked_mean"))]
    weighting_handles = [Line2D([0], [0], marker=WEIGHT_MARKERS[weight],
                                linestyle="none", markersize=8,
                                markerfacecolor="#667788", markeredgecolor="white",
                                label=weight)
                         for weight in ("none", "inverse_sqrt", "balanced")]
    pooling_legend = rank_ax.legend(handles=pooling_handles, title="Pooling color",
                                    loc="upper left", bbox_to_anchor=(0.00, -0.105),
                                    ncol=2, frameon=False, handletextpad=0.5,
                                    columnspacing=1.4, fontsize=9.5, title_fontsize=9.5,
                                    borderaxespad=0)
    rank_ax.add_artist(pooling_legend)
    rank_ax.legend(handles=weighting_handles, title="Class-weight marker",
                   loc="upper left", bbox_to_anchor=(0.37, -0.105), ncol=3,
                   frameon=False, handletextpad=0.45, columnspacing=1.2,
                   fontsize=9.2, title_fontsize=9.5, borderaxespad=0)

    metric_scores = [item[4 + metric_index] for item in RESULTS]
    vmin, vmax = min(metric_scores), max(metric_scores)
    best_grid = ranked[0][0]
    image = add_heatmap(cls_ax, "cls", metric_index, "CLS pooling",
                        cmap, vmin, vmax, best_grid)
    add_heatmap(mean_ax, "masked_mean", metric_index, "Masked-mean pooling",
                cmap, vmin, vmax, best_grid)
    colorbar = fig.colorbar(image, ax=[cls_ax, mean_ax], fraction=0.032, pad=0.04)
    colorbar.set_label(metric_title, fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    fig.savefig(OUT / f"{output_stem}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{output_stem}.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / (output_stem + '.png')}")
    print(f"Wrote {OUT / (output_stem + '.pdf')}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for metric_name in LABELS:
        make_figure(metric_name)


if __name__ == "__main__":
    main()
