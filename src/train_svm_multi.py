"""
Multi-feature let-7 family classifier for ccRCC detection.

Extends the single-feature (let-7b) SVM from train_svm.py to use all available
let-7 family members as features and reports per-miRNA contribution.

Pipeline
--------
1. Pivot let7_rpm.csv → wide format (one row per sample, one column per miRNA).
2. Stratified 80/20 train/test split.
3. SMOTE on training set only (avoids data-leakage noted in README Known Limitations).
4. Three classifiers: SVM-RBF, Logistic Regression, Random Forest.
5. Evaluate each on the real held-out test set (accuracy, precision, recall, ROC-AUC).
6. Report permutation feature importance for the best classifier.
7. Compare directly against the single-feature let-7b SVM baseline.

Reads:  data/processed/let7_rpm.csv
Writes:
    results/tables/multi_feature_metrics.csv
    results/tables/feature_importance.csv
    results/figures/fig4_multi_feature_roc.png
    results/figures/fig5_feature_importance.png
    models/best_multi_classifier.joblib
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

DATA_FILE = Path("data/processed/let7_rpm.csv")
MODEL_FILE = Path("models/best_multi_classifier.joblib")
FIG_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")

CANCER_STAGES = {"Stage 1", "Stage 2", "Stage 3", "Stage 4"}
RANDOM_STATE = 42

# Canonical family member order (matches plots.py)
LET7_ORDER = [
    "hsa-let-7a-1", "hsa-let-7a-2", "hsa-let-7a-3",
    "hsa-let-7b", "hsa-let-7c", "hsa-let-7d",
    "hsa-let-7e", "hsa-let-7f-1", "hsa-let-7f-2",
    "hsa-let-7g", "hsa-let-7i",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_wide() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return (X_wide, y) where X_wide is a DataFrame with one row per sample and
    one column per let-7 family member, y is 0=Normal / 1=Cancer.
    Samples with any missing let-7 RPM value are dropped.
    """
    df = pd.read_csv(DATA_FILE)
    df["label"] = df["Stage"].isin(CANCER_STAGES).astype(int)

    # Pivot: index = (CaseID, FileUUID, Stage, label), columns = miRNA
    wide = (
        df.pivot_table(
            index=["CaseID", "FileUUID", "Stage", "label"],
            columns="miRNA",
            values="RPM",
            aggfunc="first",
        )
        .reset_index()
    )

    # Keep only let-7 columns that are actually present
    feature_cols = [c for c in LET7_ORDER if c in wide.columns]
    wide_clean = wide.dropna(subset=feature_cols).copy()

    if wide_clean.empty:
        print("ERROR: no complete samples found after pivot.", file=sys.stderr)
        sys.exit(1)

    X = wide_clean[feature_cols]
    y = wide_clean["label"].values

    print(
        f"Loaded {len(y):,} samples  (Normal={(y==0).sum()}, Cancer={(y==1).sum()})"
    )
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    return X, y, feature_cols


# ── Classifier builders ───────────────────────────────────────────────────────

def make_svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE, C=1.0), ensemble=False)),
    ])


def make_logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs", C=1.0
        )),
    ])


def make_rf():
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])


CLASSIFIERS = {
    "SVM-RBF": make_svm,
    "LogisticRegression": make_logreg,
    "RandomForest": make_rf,
}


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_evaluate(X_df: pd.DataFrame, y: np.ndarray, feature_cols: list[str]):
    X = X_df.values

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(y_train):,}  (N={(y_train==0).sum()}, C={(y_train==1).sum()})")
    print(f"Test:  {len(y_test):,}   (N={(y_test==0).sum()}, C={(y_test==1).sum()})")

    # ── SMOTE on train only ────────────────────────────────────────────────
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        f"After SMOTE → {len(y_train_res):,} train  "
        f"(N={(y_train_res==0).sum()}, C={(y_train_res==1).sum()})"
    )

    # ── Train & score all classifiers ─────────────────────────────────────
    metrics_rows = []
    trained = {}
    roc_curves = {}

    for name, factory in CLASSIFIERS.items():
        pipe = factory()
        pipe.fit(X_train_res, y_train_res)
        trained[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)

        metrics_rows.append({
            "Classifier": name,
            "Accuracy":   round(acc, 3),
            "Precision (Cancer)": round(prec, 3),
            "Recall (Cancer)":    round(rec, 3),
            "ROC-AUC":    round(auc, 3),
        })
        roc_curves[name] = roc_curve(y_test, y_prob)

        print(f"\n── {name} ──")
        print(f"  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  AUC={auc:.3f}")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Cancer"]))

    # ── Also run single-feature let-7b baseline for direct comparison ──────
    let7b_idx = feature_cols.index("hsa-let-7b") if "hsa-let-7b" in feature_cols else None
    if let7b_idx is not None:
        X_tr_1d = X_train_res[:, [let7b_idx]]
        X_te_1d = X_test[:, [let7b_idx]]
        base_pipe = make_svm()
        base_pipe.fit(X_tr_1d, y_train_res)
        y_pred_b = base_pipe.predict(X_te_1d)
        y_prob_b = base_pipe.predict_proba(X_te_1d)[:, 1]
        metrics_rows.append({
            "Classifier": "SVM-let7b-only (baseline)",
            "Accuracy":   round(accuracy_score(y_test, y_pred_b), 3),
            "Precision (Cancer)": round(precision_score(y_test, y_pred_b, pos_label=1, zero_division=0), 3),
            "Recall (Cancer)":    round(recall_score(y_test, y_pred_b, pos_label=1, zero_division=0), 3),
            "ROC-AUC":    round(roc_auc_score(y_test, y_prob_b), 3),
        })
        roc_curves["SVM-let7b-only (baseline)"] = roc_curve(y_test, y_prob_b)
        print(f"\n── SVM (let-7b only, baseline) ──")
        print(f"  Acc={metrics_rows[-1]['Accuracy']:.3f}  AUC={metrics_rows[-1]['ROC-AUC']:.3f}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("ROC-AUC", ascending=False)
    print("\n── Summary ──")
    print(metrics_df.to_string(index=False))

    return metrics_df, trained, roc_curves, X_train_res, y_train_res, X_test, y_test


# ── Feature importance ────────────────────────────────────────────────────────

def compute_feature_importance(
    best_pipe: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Permutation importance on the held-out test set."""
    print("\nComputing permutation feature importance …")
    result = permutation_importance(
        best_pipe, X_test, y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    imp_df = pd.DataFrame({
        "miRNA": feature_cols,
        "importance_mean": result.importances_mean.round(4),
        "importance_std":  result.importances_std.round(4),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    print(imp_df.to_string(index=False))
    return imp_df


# ── Plots ─────────────────────────────────────────────────────────────────────

_COLORS = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]


def plot_roc_curves(roc_curves: dict, metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    auc_map = dict(zip(metrics_df["Classifier"], metrics_df["ROC-AUC"]))

    for (name, (fpr, tpr, _)), color in zip(roc_curves.items(), _COLORS):
        auc = auc_map.get(name, 0)
        lw = 2.5 if "baseline" not in name else 1.2
        ls = "--" if "baseline" in name else "-"
        ax.plot(fpr, tpr, lw=lw, linestyle=ls, color=color,
                label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – Multi-Feature Let-7 Classifiers\n(Normal vs. ccRCC)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    out = FIG_DIR / "fig4_multi_feature_roc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nROC curves saved → {out}")


def plot_feature_importance(imp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E91E63" if m == "hsa-let-7b" else "#2196F3"
              for m in imp_df["miRNA"]]
    ax.barh(
        imp_df["miRNA"][::-1],
        imp_df["importance_mean"][::-1],
        xerr=imp_df["importance_std"][::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Permutation Importance (ΔROC-AUC)", fontsize=11)
    ax.set_title("Per-miRNA Feature Importance\n(Best Multi-Feature Classifier)", fontsize=11)
    ax.axvline(0, color="grey", lw=0.8, linestyle="--")
    fig.tight_layout()
    out = FIG_DIR / "fig5_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not DATA_FILE.exists():
        print(
            f"ERROR: {DATA_FILE} not found.\n"
            "Run  python src/extract_let7.py  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    X_df, y, feature_cols = load_wide()

    metrics_df, trained, roc_curves, X_tr, y_tr, X_te, y_te = train_evaluate(
        X_df, y, feature_cols
    )

    # Save metrics
    out_metrics = TABLES_DIR / "multi_feature_metrics.csv"
    metrics_df.to_csv(out_metrics, index=False)
    print(f"\nMetrics saved → {out_metrics}")

    # Best classifier by ROC-AUC (excluding baseline)
    non_base = metrics_df[~metrics_df["Classifier"].str.contains("baseline")]
    best_name = non_base.iloc[0]["Classifier"]
    best_pipe = trained[best_name]
    print(f"\nBest classifier: {best_name}")

    # Feature importance for the best model
    imp_df = compute_feature_importance(best_pipe, X_te, y_te, feature_cols)
    out_imp = TABLES_DIR / "feature_importance.csv"
    imp_df.to_csv(out_imp, index=False)
    print(f"Feature importance saved → {out_imp}")

    # Plots
    plot_roc_curves(roc_curves, metrics_df)
    plot_feature_importance(imp_df)

    # Save best model
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_FILE)
    print(f"Best model saved → {MODEL_FILE}")


if __name__ == "__main__":
    main()
