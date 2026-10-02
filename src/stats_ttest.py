"""
Per-miRNA independent two-sample t-tests (normal vs. cancer) and fold changes.
Also produces pairwise stage comparisons for let-7b.

Reads:  data/processed/let7_rpm.csv
Writes:
    results/tables/ttest_normal_vs_cancer.csv
    results/tables/ttest_let7b_stages.csv
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_FILE = Path("data/processed/let7_rpm.csv")
TABLES_DIR = Path("results/tables")

STAGE_ORDER = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
CANCER_STAGES = {"Stage 1", "Stage 2", "Stage 3", "Stage 4"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["is_cancer"] = df["Stage"].isin(CANCER_STAGES)
    return df


def normal_vs_cancer_ttests(df: pd.DataFrame) -> pd.DataFrame:
    """Welch t-test (equal_var=False) for each let-7 member: normal vs. all cancer stages."""
    normal_df = df[df["Stage"] == "Normal"]
    cancer_df = df[df["is_cancer"]]

    results = []
    for mirna, _ in df.groupby("miRNA"):
        n_vals = normal_df.loc[normal_df["miRNA"] == mirna, "RPM"].dropna().values
        c_vals = cancer_df.loc[cancer_df["miRNA"] == mirna, "RPM"].dropna().values

        if len(n_vals) < 2 or len(c_vals) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(n_vals, c_vals, equal_var=False)
        mean_n = n_vals.mean()
        mean_c = c_vals.mean()
        fold_change = mean_c / mean_n if mean_n != 0 else np.nan
        direction = "up" if mean_c > mean_n else "down"

        results.append(
            {
                "miRNA": mirna,
                "n_normal": len(n_vals),
                "n_cancer": len(c_vals),
                "mean_RPM_normal": round(mean_n, 4),
                "mean_RPM_cancer": round(mean_c, 4),
                "fold_change_cancer_over_normal": round(fold_change, 4),
                "direction": direction,
                "t_statistic": round(t_stat, 4),
                "p_value": p_val,
            }
        )

    result_df = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    result_df["p_value"] = result_df["p_value"].map("{:.4e}".format)
    return result_df


def let7b_stage_ttests(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Welch t-tests between all stage pairs for hsa-let-7b."""
    let7b = df[df["miRNA"] == "hsa-let-7b"]
    stages_present = [s for s in STAGE_ORDER if s in let7b["Stage"].unique()]

    results = []
    for s1, s2 in combinations(stages_present, 2):
        v1 = let7b.loc[let7b["Stage"] == s1, "RPM"].dropna().values
        v2 = let7b.loc[let7b["Stage"] == s2, "RPM"].dropna().values

        if len(v1) < 2 or len(v2) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
        results.append(
            {
                "group_1": s1,
                "group_2": s2,
                "n_1": len(v1),
                "n_2": len(v2),
                "mean_RPM_1": round(v1.mean(), 4),
                "mean_RPM_2": round(v2.mean(), 4),
                "t_statistic": round(t_stat, 4),
                "p_value": "{:.4e}".format(p_val),
            }
        )

    return pd.DataFrame(results)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    nvc = normal_vs_cancer_ttests(df)
    out1 = TABLES_DIR / "ttest_normal_vs_cancer.csv"
    nvc.to_csv(out1, index=False)
    print("Normal vs. cancer t-tests (sorted by p-value):")
    print(nvc.to_string(index=False))
    print(f"\nSaved → {out1}\n")

    stage_df = let7b_stage_ttests(df)
    out2 = TABLES_DIR / "ttest_let7b_stages.csv"
    stage_df.to_csv(out2, index=False)
    print("hsa-let-7b pairwise stage comparisons:")
    print(stage_df.to_string(index=False))
    print(f"\nSaved → {out2}")


if __name__ == "__main__":
    main()
