from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from fig_style_v6 import PALETTE, SEQ, clean, panel, save, setup


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    setup()
    recovery = pd.read_csv(RES / "surfaceome_nested_recovery_v6.csv")
    metrics = pd.read_csv(RES / "surfaceome_nested_metrics_v6.csv")
    overlap = pd.read_csv(RES / "surfaceome_cancer_specific_top50_v6.csv")
    stages = pd.read_csv(RES / "surfaceome_nested_stage_counts_v6.csv")

    fig = plt.figure(figsize=(7.4, 5.8))
    grid = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    selected = recovery[recovery["fraction"].isin([0.01, 0.05, 0.10, 0.20])]
    for benchmark, frame in selected.groupby("benchmark"):
        ax.plot(
            100 * frame["fraction"],
            frame["observed_known"],
            marker="o",
            markersize=3.3,
            linewidth=0.8,
            label=benchmark.replace("_", " "),
        )
    ax.set_xlabel("Top surfaceome percentile")
    ax.set_ylabel("Established targets recovered")
    ax.set_title("Known-target recovery", loc="left")
    ax.legend(frameon=False, fontsize=5.2, loc="upper left")
    clean(ax, grid=True)
    panel(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    x = np.arange(len(metrics))
    width = 0.34
    ax.bar(x - width / 2, metrics["auroc"], width, color=PALETTE["blue"], label="AUROC")
    ax.bar(x + width / 2, metrics["average_precision"], width, color=PALETTE["orange"], label="AP")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Full", "RNA", "OV", "UCEC", "CESC"],
        fontsize=6.2,
    )
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Score")
    ax.set_title("Cancer-specific benchmarks", loc="left")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    clean(ax, grid=True)
    panel(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    matrix = np.array(
        [
            [np.nan, overlap.loc[0, "jaccard"], overlap.loc[1, "jaccard"]],
            [overlap.loc[0, "jaccard"], np.nan, overlap.loc[2, "jaccard"]],
            [overlap.loc[1, "jaccard"], overlap.loc[2, "jaccard"], np.nan],
        ]
    )
    image = ax.imshow(matrix, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["OV", "UCEC", "CESC"], fontsize=6.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["OV", "UCEC", "CESC"], fontsize=6.5)
    for row in range(3):
        for col in range(3):
            if np.isfinite(matrix[row, col]):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=6.2)
    ax.set_title("Top-50 rank overlap", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02, label="Jaccard")
    panel(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    y = np.arange(len(stages))
    ax.barh(y, stages["genes"], color=PALETTE["slate"], height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"Stage {index}: {row.stage.split(' ', 1)[1]}" for index, row in stages.iterrows()],
        fontsize=5.9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Genes")
    ax.set_title("Nested discovery pipeline", loc="left")
    for index, row in enumerate(stages.itertuples()):
        ax.text(row.genes + 80, index, f"n={row.genes}; known={row.known_targets}", va="center", fontsize=5.8)
    clean(ax, grid=True)
    panel(ax, "d")

    fig.text(
        0.5,
        0.008,
        "Established targets were never used to fit any score. Cancer-specific RNA-only scores are presented as external context benchmarks.",
        ha="center",
        fontsize=6.0,
        color=PALETTE["slate"],
    )
    save(fig, str(OUT), "FigS8_nested_benchmark")


if __name__ == "__main__":
    main()
