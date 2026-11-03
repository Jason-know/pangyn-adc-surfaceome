from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from fig_style_v6 import PALETTE, save, setup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def node(ax, x, y, title, body, color, dashed=False):
    ax.add_patch(
        Circle(
            (x, y),
            radius=0.63,
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )
    ax.text(x, y + 0.11, title, ha="center", va="center", fontsize=6.4, fontweight="bold")
    ax.text(
        x,
        y - 0.22,
        body,
        ha="center",
        va="top",
        fontsize=5.5,
        color=PALETTE["slate"],
        linespacing=1.16,
    )


def main():
    setup()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(
        0.2,
        5.0,
        "Surfaceome discovery with orthogonal ADC evidence",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="center",
    )

    inputs = [
        ("Bulk", "904 tumors\n54 GTEx tissues"),
        ("Protein", "HPA + CPTAC"),
        ("Literature", "43 IHC cohorts"),
        ("Surface", "CSPA + HPA"),
        ("Models", "DepMap + PRISM"),
    ]
    xs = [0.9, 3.35, 5.8, 8.25, 10.7]
    for (title, body), x in zip(inputs, xs):
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.62, 3.78),
                1.45,
                0.62,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=PALETTE["light"],
                edgecolor=PALETTE["light_gray"],
                linewidth=0.6,
            )
        )
        ax.text(x + 0.1, 4.15, title, ha="center", va="center", fontsize=6.3, fontweight="bold")
        ax.text(x + 0.1, 3.92, body, ha="center", va="center", fontsize=5.5, color=PALETTE["slate"])
        ax.plot([x + 0.1, x + 0.1], [3.75, 2.7], color=PALETTE["gray"], linewidth=0.6)

    pipeline = [
        (1.15, "Discover", "3,640 surface\ngenes", PALETTE["blue"]),
        (3.55, "Benchmark", "known targets\nas positive controls", PALETTE["sky"]),
        (5.95, "Synthesize", "IHC meta + public\nsurface annotations", PALETTE["green"]),
        (8.35, "Perturb", "DepMap + PRISM\nfunctional proxies", PALETTE["orange"]),
        (11.25, "Validate", "membrane-specific\nassay required", PALETTE["vermillion"]),
    ]
    for x, title, body, color in pipeline:
        node(ax, x, 1.65, title, body, color, dashed=title == "Validate")
    for (x1, _, _, _), (x2, _, _, _) in zip(pipeline[:-1], pipeline[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.66, 1.65),
                (x2 - 0.66, 1.65),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color=PALETTE["gray"],
            )
        )

    ax.text(
        5.7,
        0.32,
        "Evidence is kept as separate axes: RNA nomination, IHC prevalence, surface annotation, and functional proxy",
        ha="center",
        va="center",
        fontsize=6.7,
        color=PALETTE["ink"],
    )
    ax.text(
        11.25,
        0.63,
        "unresolved",
        ha="center",
        va="center",
        fontsize=5.7,
        color=PALETTE["vermillion"],
        fontweight="bold",
    )
    save(fig, str(OUT), "Graphical_Abstract")


if __name__ == "__main__":
    main()
