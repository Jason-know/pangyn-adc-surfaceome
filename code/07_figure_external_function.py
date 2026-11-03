from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from fig_style_v6 import CANCER, DIVERGING, PALETTE, SEQ, clean, panel, save, setup


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GENES = [
    "EPCAM",
    "MUC16",
    "VTCN1",
    "CD276",
    "FOLR1",
    "TACSTD2",
    "MSLN",
    "NECTIN4",
    "CLDN4",
    "MAL2",
    "TMPRSS4",
    "CLDN3",
    "MUC1",
    "CLDN7",
    "RAB25",
    "LSR",
]
CANDIDATES = {"CLDN4", "MAL2", "TMPRSS4", "CLDN3", "MUC1", "CLDN7", "RAB25", "LSR"}


def membrane_validation(ax, metrics):
    metrics = metrics.copy()
    metrics["label"] = metrics["annotation"].map(
        {
            "cspa_high_confidence": "CSPA high",
            "cspa_any": "CSPA any",
            "hpa_plasma_membrane": "HPA plasma\nmembrane",
            "hpa_membrane_associated": "HPA membrane\nor junction",
        }
    )
    x = np.arange(len(metrics))
    width = 0.34
    ax.bar(
        x - width / 2,
        metrics["top_decile_rate"],
        width,
        color=PALETTE["blue"],
        label="top decile",
    )
    ax.bar(
        x + width / 2,
        metrics["rest_rate"],
        width,
        color=PALETTE["light_gray"],
        label="rest",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["label"], fontsize=6.2)
    ax.set_ylim(0, max(metrics[["top_decile_rate", "rest_rate"]].max()) * 1.35)
    ax.set_ylabel("Annotation-positive fraction")
    ax.set_title("Public membrane annotation enrichment", loc="left")
    for index, row in enumerate(metrics.itertuples()):
        ax.text(
            index,
            max(row.top_decile_rate, row.rest_rate) + 0.015,
            f"OR {row.odds_ratio:.2f}\nFDR {row.fdr_bh:.3g}",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    clean(ax, grid=True)
    panel(ax, "a")


def candidate_support(ax, candidates):
    columns = [
        "uniprot_surface",
        "cspa_high_confidence",
        "cspa_any",
        "hpa_plasma_membrane",
        "hpa_membrane_associated",
    ]
    matrix = candidates.set_index("gene").reindex(GENES)[columns].astype(float)
    ax.imshow(matrix.to_numpy(), aspect="auto", cmap=SEQ, vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=6.2, color=PALETTE["ink"])
    for index, label in enumerate(ax.get_yticklabels()):
        if matrix.index[index] in CANDIDATES:
            label.set_fontweight("bold")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        ["In silico", "CSPA\nhigh", "CSPA\nany", "HPA\nplasma", "HPA\njunction"],
        fontsize=6.0,
    )
    ax.set_title("Surface annotation support", loc="left")
    panel(ax, "b")


def heatmap(ax, frame, value_column, title, letter, diverging=False):
    matrix = frame.pivot(index="target", columns="cancer", values=value_column).reindex(
        index=GENES,
        columns=["OV", "UCEC", "CESC"],
    )
    if diverging:
        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap=DIVERGING,
            norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5),
        )
    else:
        image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=SEQ, vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=6.0, color=PALETTE["ink"])
    for index, label in enumerate(ax.get_yticklabels()):
        if matrix.index[index] in CANDIDATES:
            label.set_fontweight("bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["OV", "UCEC", "CESC"], fontsize=6.7)
    ax.set_title(title, loc="left")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Expression prevalence" if not diverging else "Mean gene effect", fontsize=6.3)
    colorbar.ax.tick_params(labelsize=5.8, width=0.4, length=1.5)
    panel(ax, letter, x=-0.3)


def payload_associations(ax, data):
    selected = (
        data.sort_values(["fdr_bh", "p"])
        .head(18)
        .sort_values("spearman_rho")
        .copy()
    )
    selected["label"] = (
        selected["target"]
        + " | "
        + selected["cancer"]
        + " | "
        + selected["payload_class"].map(
            {
                "TOP1_payload_proxy": "TOP1",
                "microtubule_payload_proxy": "MT",
            }
        )
    )
    colors = selected["payload_class"].map(
        {
            "TOP1_payload_proxy": PALETTE["green"],
            "microtubule_payload_proxy": PALETTE["orange"],
        }
    )
    y = np.arange(len(selected))
    ax.hlines(y, 0, selected["spearman_rho"], color=PALETTE["grid"], linewidth=0.6)
    ax.scatter(
        selected["spearman_rho"],
        y,
        s=np.where(selected["fdr_bh"] <= 0.05, 38, 22),
        c=colors,
        marker="D",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(selected["label"], fontsize=5.7, color=PALETTE["ink"])
    ax.set_xlim(-0.75, 0.65)
    ax.set_xlabel("Expression-payload response rho")
    ax.set_title("PRISM payload-class proxy", loc="left")
    clean(ax, grid=True)
    panel(ax, "c")


def nested_benchmark(ax, metrics, recovery):
    metrics = metrics.set_index("benchmark")
    labels = [
        "full_3131_with_cellular_specificity",
        "expression_only_3640",
        "OV_expression_only",
        "UCEC_expression_only",
        "CESC_expression_only",
    ]
    display = ["Full", "RNA only", "OV", "UCEC", "CESC"]
    x = np.arange(len(labels))
    ax.plot(x, metrics.loc[labels, "auroc"], marker="o", markersize=4, color=PALETTE["blue"], linewidth=1.0, label="AUROC")
    ax.plot(
        x,
        metrics.loc[labels, "average_precision"],
        marker="D",
        markersize=3.5,
        color=PALETTE["orange"],
        linewidth=1.0,
        label="Average precision",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=6.3)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Known-target benchmark")
    ax.set_title("Nested surfaceome benchmark", loc="left")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    clean(ax, grid=True)
    panel(ax, "d")


def main():
    setup()
    membrane = pd.read_csv(RES / "surface_proteome_benchmark_v6.csv")
    support = pd.read_csv(RES / "surface_proteome_annotation_support_v6.csv")
    depmap = pd.read_csv(RES / "depmap_target_function_v6.csv")
    payload = pd.read_csv(RES / "target_payload_functional_proxy_v6.csv")
    metrics = pd.read_csv(RES / "surfaceome_nested_metrics_v6.csv")
    recovery = pd.read_csv(RES / "surfaceome_nested_recovery_v6.csv")

    candidates = support[support["gene"].isin(GENES)].copy()
    depmap = depmap[depmap["target"].isin(GENES)]

    fig = plt.figure(figsize=(7.5, 9.2))
    grid = gridspec.GridSpec(
        3,
        2,
        width_ratios=[1.15, 1.0],
        height_ratios=[1.0, 1.35, 1.25],
        hspace=0.48,
        wspace=0.55,
    )
    membrane_validation(fig.add_subplot(grid[0, 0]), membrane)
    candidate_support(fig.add_subplot(grid[0, 1]), candidates)
    heatmap(
        fig.add_subplot(grid[1, 0]),
        depmap,
        "fraction_expressed",
        "DepMap target expression",
        "e",
    )
    heatmap(
        fig.add_subplot(grid[1, 1]),
        depmap,
        "median_gene_effect",
        "DepMap CRISPR dependency",
        "f",
        diverging=True,
    )
    payload_associations(fig.add_subplot(grid[2, 0]), payload)
    nested_benchmark(fig.add_subplot(grid[2, 1]), metrics, recovery)
    fig.text(
        0.5,
        0.008,
        "CSPA and HPA are membrane annotations, not paired antibody-accessibility measurements. "
        "DepMap and PRISM are perturbational proxies and do not test ADC killing.",
        ha="center",
        fontsize=6.0,
        color=PALETTE["slate"],
    )
    save(fig, str(OUT), "Fig8_external_function")


if __name__ == "__main__":
    main()
