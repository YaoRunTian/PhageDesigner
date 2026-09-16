#!/usr/bin/env python3
"""Plot per-class F1 for the three best completed ESM2-650M tuning grids."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

LABELS = [
    "assembly",
    "replication",
    "infection",
    "packaging",
    "integration",
    "regulation",
    "lysis",
    "immune",
    "tRNA_related",
]

# Source: server-side validation_metrics.json files under
# results/07_esm2_650m_fixed_split_tuning/.  The configurations are ordered by
# increasing validation Macro-F1 so the strongest result is the dark rear line,
# matching the earlier ESM-2 model-size visualization.
CONFIGS = [
    {
        "label": "Grid 14\n3e-4 · inverse_sqrt",
        "macro_f1": 0.6625668600224415,
        "face": "#eeffe6",
        "edge": "#79b9a7",
        "marker": "o",
        "f1": [
            0.69392575928009,
            0.6677438901232076,
            0.7782340862422997,
            0.7387183477153771,
            0.6861939632077156,
            0.5630721143112302,
            0.6951167728237793,
            0.5479915433403806,
            0.5921052631578947,
        ],
    },
    {
        "label": "Grid 17\n1e-3 · inverse_sqrt",
        "macro_f1": 0.6772288962608655,
        "face": "#0bdacd",
        "edge": "#079f99",
        "marker": "s",
        "f1": [
            0.7065396377322983,
            0.6838398068021734,
            0.7838147346990711,
            0.7584988962472405,
            0.7001751313485113,
            0.567588624853915,
            0.7036878216123499,
            0.5718677940046117,
            0.6190476190476191,
        ],
    },
    {
        "label": "Grid 16\n1e-3 · none",
        "macro_f1": 0.6871655981525348,
        "face": "#163179",
        "edge": "#163179",
        "marker": "^",
        "f1": [
            0.7210764195542626,
            0.6941752422819462,
            0.7979957805907173,
            0.7787338660110633,
            0.7134914751667902,
            0.5897321428571428,
            0.7133670188332567,
            0.566543438077634,
            0.609375,
        ],
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(LABELS), dtype=float)

    fig = plt.figure(figsize=(14.2, 8.4))
    ax = fig.add_subplot(111, projection="3d")

    for y, config in enumerate(CONFIGS):
        values = np.asarray(config["f1"], dtype=float)
        y_values = np.full_like(x, y)

        ax.plot(
            x,
            y_values,
            values,
            color=config["edge"],
            linewidth=2.7,
            marker=config["marker"],
            markersize=6.5,
            markerfacecolor=config["face"],
            markeredgecolor=config["edge"],
            markeredgewidth=1.1,
            zorder=5,
        )

        vertices = [
            list(
                zip(
                    np.r_[x, x[::-1]],
                    np.r_[values, np.zeros_like(values)[::-1]],
                )
            )
        ]
        polygon = PolyCollection(
            vertices,
            facecolors=config["face"],
            edgecolors=config["edge"],
            linewidths=1.4,
            alpha=0.42,
        )
        ax.add_collection3d(polygon, zs=y, zdir="y")

        for x_value, f1_value in zip(x, values):
            ax.text(
                x_value,
                y,
                f1_value + 0.022,
                f"{f1_value:.2f}",
                fontsize=7.2,
                ha="center",
                va="bottom",
                color="#1d2733",
            )

    ax.set_xlim(-0.35, len(LABELS) - 0.65)
    ax.set_ylim(-0.45, len(CONFIGS) - 0.55)
    ax.set_zlim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=25, ha="right", fontsize=8.2)
    ax.set_yticks(range(len(CONFIGS)))
    ax.set_yticklabels(
        [f"G{grid}\nMacro-F1 {item['macro_f1']:.4f}" for grid, item in zip((14, 17, 16), CONFIGS)],
        fontsize=9.2,
    )
    ax.set_zticks(np.arange(0.0, 1.01, 0.1))

    ax.set_xlabel("Functional category", fontsize=12, labelpad=34)
    ax.set_ylabel("Parameter set", fontsize=12, labelpad=18)
    ax.set_zlabel("F1 score", fontsize=12, labelpad=10)
    ax.set_title(
        "Per-class F1 across the top three ESM2-650M tuning configurations",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    ax.view_init(elev=24, azim=-60)
    ax.set_box_aspect((1.8, 0.78, 0.82))
    ax.xaxis.pane.set_facecolor((0.96, 0.98, 0.98, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.98, 0.98, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.grid(True)

    fig.text(
        0.5,
        0.050,
        "G14: 3e-4 + inverse_sqrt   |   G17: 1e-3 + inverse_sqrt   |   G16: 1e-3 + none   (all masked_mean)",
        ha="center",
        fontsize=10.5,
        color="#1D2733",
    )
    fig.text(
        0.5,
        0.022,
        "Fixed train_100k and validation_30k; frozen ESM2-650M backbone; final_test not accessed.",
        ha="center",
        fontsize=10.5,
        color="#51606E",
    )
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.13, top=0.91)
    fig.savefig(OUT / "figure5.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "figure5.pdf", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT / 'figure5.png'}")
    print(f"Wrote {OUT / 'figure5.pdf'}")


if __name__ == "__main__":
    main()
