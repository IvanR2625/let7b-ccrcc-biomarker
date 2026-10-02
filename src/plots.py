"""
Generate figures for the let-7b ccRCC biomarker paper.

Figure 1 – Box plot: RPM per let-7 member, Normal vs. Cancer (all stages combined).
Figure 2 – Violin plot: log10(let-7b RPM + 1) by Stage (Normal, Stage 1–4).

Reads:  data/processed/let7_rpm.csv
Writes:
    results/figures/fig1_let7_normal_vs_cancer_boxplot.png
    results/figures/fig2_let7b_by_stage_violin.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = Path("data/processed/let7_rpm.csv")
FIG_DIR = Path("results/figures")

STAGE_ORDER = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
CANCER_STAGES = {"Stage 1", "Stage 2", "Stage 3", "Stage 4"}

# Colour palettes
GROUP_PALETTE = {"Normal": "#2196F3", "Cancer": "#F44336"}
STAGE_PALETTE = {
    "Normal": "#2196F3",
    "Stage 1": "#FF9800",
    "Stage 2": "#4CAF50",
    "Stage 3": "#E91E63",
    "Stage 4": "#9C27B0",
}

# Canonical family member order for the x-axis
_SUFFIX_RANK = {
    "a": 1, "a-1": 1, "a-2": 1, "a-3": 1,
    "b": 2, "c": 3, "d": 4, "e": 5,
    "f": 6, "f-1": 6, "f-2": 6,
    "g": 7, "i": 8,
}


def _mirna_sort_key(m: str):
    suffix = m.lower().replace("hsa-let-7", "").lstrip("-")
    return _SUFFIX_RANK.get(suffix, 99), m


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["Group"] = df["Stage"].apply(lambda s: "Normal" if s == "Normal" else "Cancer")
    return df


def get_mirna_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["miRNA"].unique(), key=_mirna_sort_key)


def plot_fig1(df: pd.DataFrame):
    """Box plot of RPM for each let-7 member, grouped by Normal vs. Cancer."""
    mirna_order = get_mirna_order(df)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df,
        x="miRNA",
        y="RPM",
        hue="Group",
        order=mirna_order,
        hue_order=["Normal", "Cancer"],
        palette=GROUP_PALETTE,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        linewidth=0.8,
        ax=ax,
    )

    ax.set_xlabel("miRNA", fontsize=12)
    ax.set_ylabel("RPM (reads per million miRNA mapped)", fontsize=12)
    ax.set_title(
        "Let-7 Family Expression: Normal vs. ccRCC Tumour Tissue\n"
        "(TCGA-KIRC; n = 20 normal, 537 tumour)",
        fontsize=12,
    )
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Tissue type", fontsize=10)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = FIG_DIR / "fig1_let7_normal_vs_cancer_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_fig2(df: pd.DataFrame):
    """Violin plot of log10(let-7b RPM + 1) by stage."""
    let7b = df[df["miRNA"] == "hsa-let-7b"].copy()
    let7b["log10_RPM_p1"] = np.log10(let7b["RPM"] + 1)

    stages_present = [s for s in STAGE_ORDER if s in let7b["Stage"].unique()]
    palette = {s: STAGE_PALETTE[s] for s in stages_present}

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=let7b,
        x="Stage",
        y="log10_RPM_p1",
        order=stages_present,
        palette=palette,
        inner="box",
        cut=0,
        linewidth=0.8,
        ax=ax,
    )

    # Annotate sample counts below each violin
    for i, stage in enumerate(stages_present):
        n = (let7b["Stage"] == stage).sum()
        ax.text(i, ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n={n}", ha="center", va="top", fontsize=8, color="grey")

    ax.set_xlabel("Stage", fontsize=12)
    ax.set_ylabel("log₁₀(RPM + 1)", fontsize=12)
    ax.set_title("hsa-let-7b Expression by Stage in ccRCC (TCGA-KIRC)", fontsize=12)

    fig.tight_layout()
    out = FIG_DIR / "fig2_let7b_by_stage_violin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    plot_fig1(df)
    plot_fig2(df)


if __name__ == "__main__":
    main()
