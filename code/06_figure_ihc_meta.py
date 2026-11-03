from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fig_style_v6 import CANCER, PALETTE, clean, panel, save, setup


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LIT = ROOT / "literature"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_ORDER = ["FOLR1", "TACSTD2", "CLDN6", "ERBB2", "VTCN1", "F3", "MSLN", "NECTIN4", "EPCAM", "SLC34A2"]


def forest(ax, frame):
    y = np.arange(len(frame))
    for index, row in enumerate(frame.itertuples()):
        ax.errorbar(
            row.pooled_prevalence,
            index,
            xerr=[
                [row.pooled_prevalence - row.ci_low],
                [row.ci_high - row.pooled_prevalence],
            ],
            fmt="D",
            markersize=4.0,
            color=PALETTE["ink"],
            ecolor=PALETTE["slate"],
            elinewidth=0.75,
            capsize=1.7,
            zorder=3,
        )
        ax.text(
            0.985,
            index,
            f"k={row.k}; I2={row.i_squared:.0f}%",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.2,
            color=PALETTE["slate"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row.target}\n{row.analysis_threshold}" for row in frame.itertuples()],
        fontsize=6.4,
        color=PALETTE["ink"],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pooled prevalence, high-expression or stated threshold")
    ax.set_title("Random-effects prevalence", loc="left")
    clean(ax, grid=True)
    panel(ax, "a")


def study_distribution(ax, data):
    y_lookup = {target: index for index, target in enumerate(TARGET_ORDER)}
    rng = np.random.default_rng(20261101)
    for target, group in data.groupby("target"):
        if target not in y_lookup:
            continue
        y = y_lookup[target]
        ax.hlines(y, group["prevalence"].min(), group["prevalence"].max(), color=PALETTE["grid"], linewidth=1.0)
        jitter = rng.uniform(-0.17, 0.17, len(group))
        ax.scatter(
            group["prevalence"],
            y + jitter,
            s=10 + 5 * np.log10(group["n_total"] + 1),
            c=[CANCER[cancer] for cancer in group["cancer"]],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
            zorder=3,
        )
    ax.set_yticks(range(len(TARGET_ORDER)))
    ax.set_yticklabels(TARGET_ORDER, fontsize=6.5, color=PALETTE["ink"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Study-level IHC prevalence")
    ax.set_title("Included cohorts", loc="left")
    for cancer, color in CANCER.items():
        ax.scatter([], [], color=color, s=18, label=cancer)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right", ncol=3)
    clean(ax, grid=True)
    panel(ax, "b")


def subgroup_forest(ax, subgroup):
    selected = subgroup[
        (subgroup["subgroup"] != "All thresholds")
        & (subgroup["k"] >= 2)
    ].copy()
    selected["label"] = selected["target"] + " | " + selected["subgroup"]
    selected = selected.sort_values(["target", "subgroup"])
    y = np.arange(len(selected))
    ax.errorbar(
        selected["pooled_prevalence"],
        y,
        xerr=[
            selected["pooled_prevalence"] - selected["ci_low"],
            selected["ci_high"] - selected["pooled_prevalence"],
        ],
        fmt="o",
        markersize=3.2,
        color=PALETTE["blue"],
        ecolor=PALETTE["slate"],
        elinewidth=0.65,
        capsize=1.4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(selected["label"], fontsize=5.9, color=PALETTE["ink"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prevalence")
    ax.set_title("Threshold-defined subgroups", loc="left")
    clean(ax, grid=True)
    panel(ax, "c")


def flow(ax, summary):
    values = summary.set_index("stage")["count"]
    identified = int(values.get("Records identified", 0))
    eligible = int(values.get("Records with target, cancer, IHC, and quantitative signal", 0))
    reports = int(values.get("Reports included after structured-data extraction", 0))
    cohorts = int(values.get("Target-cancer cohorts included", 0))
    boxes = [
        (0.50, 0.88, f"Records identified\nEurope PMC\nn={identified}"),
        (0.50, 0.65, f"Target, cancer, IHC,\nand quantitative signal\nn={eligible}"),
        (0.50, 0.42, f"Reports meeting\nstructured extraction\nn={reports}"),
        (0.50, 0.18, f"Target-cancer cohorts\nin quantitative synthesis\nn={cohorts}"),
    ]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.25, y - 0.085),
                0.50,
                0.17,
                boxstyle="round,pad=0.01,rounding_size=0.025",
                facecolor="white",
                edgecolor=PALETTE["slate"],
                linewidth=0.7,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=6.3, color=PALETTE["ink"])
    for first, second in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (first[0], first[1] - 0.09),
                (second[0], second[1] + 0.09),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.7,
                color=PALETTE["gray"],
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (0.25, 0.65),
            (0.02, 0.65),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.6,
            color=PALETTE["gray"],
        )
    )
    ax.text(
        0.015,
        0.51,
        f"Excluded at title/abstract\nscreening: n={identified - eligible}",
        fontsize=5.8,
        ha="left",
        color=PALETTE["slate"],
    )
    ax.text(
        0.765,
        0.42,
        f"Not extractable or duplicate:\nn={eligible - reports}",
        fontsize=5.8,
        ha="left",
        color=PALETTE["slate"],
    )
    ax.set_title("Search and inclusion flow", loc="left")
    panel(ax, "d", x=-0.05)


def main():
    setup()
    primary = pd.read_csv(RES / "ihc_meta_primary_v6.csv")
    primary["target"] = pd.Categorical(primary["target"], categories=TARGET_ORDER, ordered=True)
    primary = primary.sort_values("target")
    studies = pd.read_csv(LIT / "ihc_study_characteristics.csv")
    estimates = pd.read_csv(LIT / "ihc_prevalence_estimates.csv").merge(
        studies[["study_id", "first_author", "year", "pmid", "doi"]],
        on="study_id",
        how="left",
    )
    estimates["prevalence"] = estimates["n_positive"] / estimates["n_total"]
    subgroups = pd.read_csv(RES / "ihc_meta_subgroups_v6.csv")
    search = pd.read_csv(LIT / "prisma_search_summary_20261102.csv")

    fig = plt.figure(figsize=(7.5, 9.0))
    grid = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1.1, 1.25],
        hspace=0.42,
        wspace=0.5,
    )
    forest(fig.add_subplot(grid[0, 0]), primary.reset_index(drop=True))
    study_distribution(fig.add_subplot(grid[0, 1]), estimates)
    subgroup_forest(fig.add_subplot(grid[1, 0]), subgroups)
    flow(fig.add_subplot(grid[1, 1]), search)
    fig.text(
        0.5,
        0.008,
        "Random effects were fitted on logit prevalence with DerSimonian-Laird variance. "
        "Thresholds differed across studies; subgroup estimates are shown instead of treating them as interchangeable.",
        ha="center",
        fontsize=6.0,
        color=PALETTE["slate"],
    )
    save(fig, str(OUT), "Fig7_IHC_meta")


if __name__ == "__main__":
    main()
