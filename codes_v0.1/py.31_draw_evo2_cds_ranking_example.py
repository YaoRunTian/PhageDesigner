#!/usr/bin/env python3
"""Render a transparent CDS likelihood-ranking example."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


OUT = Path(
    "/Users/yaoruntian/LiLab/phage/phage_designer_v0.1/figures/"
    "evo2_cds_likelihood_ranking_example.png"
)


def main():
    mono = FontProperties(family="DejaVu Sans Mono", size=34)

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=400)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x = 0.055
    ax.text(x, 0.84, "G-like protein 1", ha="left", va="center",
            fontproperties=mono, color="#000000")
    ax.text(x, 0.62, "├── CDS 1: −0.68", ha="left", va="center",
            fontproperties=mono, color="#000000")
    ax.text(x, 0.40, "├── CDS 2: −0.51", ha="left", va="center",
            fontproperties=mono, color="#000000")
    ax.text(0.735, 0.40, "← retained", ha="left", va="center",
            fontproperties=mono, color="#000000")
    ax.text(x, 0.18, "└── CDS 3: −0.76", ha="left", va="center",
            fontproperties=mono, color="#000000")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
