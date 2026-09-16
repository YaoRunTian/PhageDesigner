#!/usr/bin/env python3
"""Draw a compact transparent ranking bar-chart icon for the Evo2 workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(
    "/Users/yaoruntian/LiLab/phage/phage_designer_v0.1/figures/"
    "evo2_likelihood_ranking_bar_icon.png"
)


def main():
    values = np.array([1.00, 0.82, 0.65, 0.48, 0.31])
    colors = ["#163179", "#126f91", "#0bdacd", "#91eee0", "#eeffe6"]
    edge = "#163179"

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=400)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    x = np.arange(len(values))
    ax.bar(
        x,
        values,
        width=0.66,
        color=colors,
        edgecolor=edge,
        linewidth=2.2,
        zorder=3,
    )

    # Minimal axes with arrowheads make the ordering direction visually explicit.
    ax.annotate(
        "",
        xy=(4.72, 0),
        xytext=(-0.55, 0),
        arrowprops=dict(arrowstyle="-|>", color="#000000", lw=2.0, mutation_scale=14),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(-0.48, 1.18),
        xytext=(-0.48, 0),
        arrowprops=dict(arrowstyle="-|>", color="#000000", lw=2.0, mutation_scale=14),
        annotation_clip=False,
    )

    ax.set_xlim(-0.65, 4.85)
    ax.set_ylim(0, 1.24)
    ax.axis("off")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
