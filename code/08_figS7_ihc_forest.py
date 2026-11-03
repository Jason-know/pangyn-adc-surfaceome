from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_style_v6 import CANCER, PALETTE, clean, panel, save, setup


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def wilson(y, n, z=1.959963984540054):
    p = y / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    setup()
    data = pd.read_csv(RES / "ihc_meta_primary_inputs_v6.csv")
    primary = pd.read_csv(RES / "ihc_meta_primary_v6.csv").set_index("target")
    targets = primary.sort_values("pooled_prevalence", ascending=False).index.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 6.7), sharex=True)
    axes = axes.ravel()
    for ax, target in zip(axes, targets):
        frame = data[data["target"] == target].sort_values(["cancer", "year", "study_id"]).copy()
        frame["wilson_low"] = [wilson(y, n)[0] for y, n in zip(frame["n_positive"], frame["n_total"])]
        frame["wilson_high"] = [wilson(y, n)[1] for y, n in zip(frame["n_positive"], frame["n_total"])]
        y = np.arange(len(frame))
        for index, row in enumerate(frame.itertuples()):
            ax.errorbar(
                row.prevalence,
                index,
                xerr=[
                    [row.prevalence - row.wilson_low],
                    [row.wilson_high - row.prevalence],
                ],
                fmt="o",
                markersize=3,
                color=CANCER[row.cancer],
                ecolor=CANCER[row.cancer],
                elinewidth=0.6,
                capsize=1.3,
            )
        pooled = primary.loc[target]
        ax.axvline(pooled["pooled_prevalence"], color=PALETTE["ink"], linestyle="--", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{row.study_id} ({row.cancer}, n={row.n_total})" for row in frame.itertuples()],
            fontsize=5.6,
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(
            f"{target}\npooled {pooled['pooled_prevalence']:.2f}; I2={pooled['i_squared']:.0f}%",
            loc="left",
            fontsize=7.0,
        )
        clean(ax, grid=True)
        panel(ax, "abcdefg"[targets.index(target)], x=-0.38, size=8)
    axes[-1].axis("off")
    fig.text(
        0.5,
        0.008,
        "Points show study prevalence with Wilson 95% intervals; dashed lines and panel titles show random-effects estimates.",
        ha="center",
        fontsize=6.0,
        color=PALETTE["slate"],
    )
    save(fig, str(OUT), "FigS7_IHC_forest")


if __name__ == "__main__":
    main()
