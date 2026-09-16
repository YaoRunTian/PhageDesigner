#!/usr/bin/env python3
"""Render only the Evo2 mean autoregressive log-likelihood formula."""

from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(
    "/Users/yaoruntian/LiLab/phage/phage_designer_v0.1/figures/"
    "evo2_mean_autoregressive_log_likelihood_formula.png"
)


def main():
    fig, ax = plt.subplots(figsize=(12.0, 2.5), dpi=400)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.axis("off")

    equation = (
        r"$\overline{\ell}_{\mathrm{Evo2}}"
        r"=\frac{1}{L}\sum_{i=s}^{e}"
        r"\log p_\theta\,\left(x_i\mid x_{<i}\right)$"
    )
    ax.text(
        0.5,
        0.52,
        equation,
        ha="center",
        va="center",
        fontsize=48,
        color="#000000",
        transform=ax.transAxes,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
