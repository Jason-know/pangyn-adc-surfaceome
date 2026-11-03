import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


PALETTE = {
    "ink": "#102331",
    "navy": "#102331",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "amber": "#E69F00",
    "vermillion": "#D55E00",
    "coral": "#D55E00",
    "green": "#009E73",
    "teal": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#AEB7BD",
    "slate": "#5F6C74",
    "light": "#F5F7F8",
    "light_gray": "#E7EBEE",
    "grid": "#D9DFE3",
}
CANCER = {"OV": "#0072B2", "UCEC": "#D55E00", "CESC": "#009E73"}
TARGET = {
    "MUC16": "#D55E00",
    "EPCAM": "#0072B2",
    "MSLN": "#E69F00",
    "TACSTD2": "#CC79A7",
    "FOLR1": "#009E73",
}
SEQ = LinearSegmentedColormap.from_list(
    "seq_teal_navy",
    ["#F8FBFC", "#D5ECEE", "#84C7C4", "#2E9B97", "#115E76", "#102331"],
)
SEQ_REVERSED = SEQ.reversed()
DIVERGING = LinearSegmentedColormap.from_list(
    "div_violet_white_orange",
    ["#5B3E8E", "#B8A7D3", "#F7F7F7", "#F2B47A", "#B24A1C"],
)


def target_color(target):
    return TARGET.get(target, PALETTE["gray"])


def setup():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(ax, letter, x=-0.16, y=1.04, size=10):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PALETTE["ink"],
    )


def clean(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)


def save(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"saved {name}")
