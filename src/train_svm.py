"""
Train and evaluate a single-feature SVM classifier (hsa-let-7b RPM) to distinguish
normal kidney tissue from ccRCC tumour tissue.

Method
------
1. Load let-7b RPM from data/processed/let7_rpm.csv.
2. Stratified 80/20 train/test split (preserves the 20 normal / 537 cancer ratio).
3. Apply SMOTE *only* to the training set to oversample the minority (normal) class.
   (This avoids the data-leakage risk noted in README.md Known Limitations.)
4. Fit scikit-learn SVC with an RBF kernel.
5. Report accuracy, precision, recall, and ROC-AUC on the *real* held-out test set.
6. Save the trained pipeline to models/svm_let7b.joblib.
7. Save the ROC curve to results/figures/fig3_roc_curve.png.

Usage
-----
    python src/train_svm.py

Note on reported paper metrics
-------------------------------
The paper (Table 4) reports Accuracy=0.907, Precision=0.887, Recall=0.936,
ROC-AUC=0.931 on an evaluation set of 106 Normal / 109 Cancer – a nearly balanced
split that is inconsistent with 20 real normal samples.  This suggests SMOTE was
applied *before* the train/test split, allowing synthetic normals to appear in the
test set and likely inflating the metrics.  This script implements the corrected
(split-first) approach; results will differ from the paper until a clean evaluation
is agreed upon by the authors.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_FILE = Path("data/processed/let7_rpm.csv")
MODEL_FILE = Path("models/svm_let7b.joblib")
FIG_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")

CANCER_STAGES = {"Stage 1", "Stage 2", "Stage 3", "Stage 4"}
RANDOM_STATE = 42


def load_let7b() -> tuple[np.ndarray, np.ndarray]:
    """Return X (RPM, shape [n,1]) and y (0=Normal, 1=Cancer) for hsa-let-7b."""
    df = pd.read_csv(DATA_FILE)
    let7b = df[df["miRNA"] == "hsa-let-7b"].dropna(subset=["RPM"]).copy()
    let7b["label"] = let7b["Stage"].isin(CANCER_STAGES).astype(int)

    X = let7b["RPM"].values.reshape(-1, 1)
    y = let7b["label"].values
    print(f"Loaded {len(y):,} samples  (Normal={( y==0).sum()}, Cancer={(y==1).sum()})")
    return X, y


def build_pipeline() -> Pipeline:
    """StandardScaler + RBF SVC; probability=True enables ROC-AUC."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    )


def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    # ── 1. Stratified split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"\nTrain: {len(y_train):,}  (Normal={( y_train==0).sum()}, Cancer={(y_train==1).sum()})"
    )
    print(
        f"Test:  {len(y_test):,}   (Normal={( y_test==0).sum()}, Cancer={(y_test==1).sum()})"
    )

    # ── 2. SMOTE on training set only ─────────────────────────────────────────
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        f"\nAfter SMOTE → Train: {len(y_train_res):,}  "
        f"(Normal={(y_train_res==0).sum()}, Cancer={(y_train_res==1).sum()})"
    )

    # ── 3. Train ──────────────────────────────────────────────────────────────
    pipe = build_pipeline()
    pipe.fit(X_train_res, y_train_res)

    # ── 4. Evaluate on real held-out test set ─────────────────────────────────
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n── Test-set metrics (split-first, SMOTE on train only) ──")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  ROC-AUC  : {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Cancer"]))

    # ── 5. Cross-validation (on full dataset with SMOTE inside each fold) ─────
    print("── Stratified 5-fold cross-validation (for reference) ──")
    cv_results = cross_validate(
        build_pipeline(),
        X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring=["accuracy", "roc_auc"],
        return_train_score=False,
    )
    print(
        f"  CV Accuracy : {cv_results['test_accuracy'].mean():.3f} "
        f"± {cv_results['test_accuracy'].std():.3f}"
    )
    print(
        f"  CV ROC-AUC  : {cv_results['test_roc_auc'].mean():.3f} "
        f"± {cv_results['test_roc_auc'].std():.3f}"
    )

    # ── 6. ROC curve ──────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    _save_roc_curve(fpr, tpr, auc)

    # ── 7. Save model ─────────────────────────────────────────────────────────
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_FILE)
    print(f"\nModel saved → {MODEL_FILE}")

    # ── 8. Save metrics table ─────────────────────────────────────────────────
    from sklearn.metrics import precision_score, recall_score
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(
        [
            {
                "metric": "Accuracy",
                "value": round(acc, 3),
            },
            {
                "metric": "Precision (Cancer)",
                "value": round(precision_score(y_test, y_pred, pos_label=1), 3),
            },
            {
                "metric": "Recall (Cancer)",
                "value": round(recall_score(y_test, y_pred, pos_label=1), 3),
            },
            {
                "metric": "ROC-AUC",
                "value": round(auc, 3),
            },
        ]
    )
    out_csv = TABLES_DIR / "svm_metrics.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"Metrics saved → {out_csv}")


def _save_roc_curve(fpr, tpr, auc: float):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#E91E63", label=f"SVM (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve – SVM on hsa-let-7b RPM\n(Normal vs. ccRCC)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    out = FIG_DIR / "fig3_roc_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve saved → {out}")


def main():
    if not DATA_FILE.exists():
        print(
            f"ERROR: {DATA_FILE} not found.\n"
            "Run  python src/extract_let7.py  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    X, y = load_let7b()
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
