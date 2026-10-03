# let7b-ccrcc-biomarker

**Exploring the prognostic potential of let-7 microRNA in clear cell renal cell carcinoma (ccRCC)**

Code, analysis, and supplementary material for the paper *"Exploring the Prognostic Potential of Let-7 MicroRNA in Clear Cell Renal Cancer"* (Raizada I., Raizada S., Cai L.), submitted to the *Journal of Emerging Investigators* (manuscript JEI-26-058).

> Paper: _link / DOI to be added once published_

---

## Overview

Existing studies disagree on whether let-7 microRNAs are up- or down-regulated in ccRCC. Using miRNA-Seq data from the TCGA-KIRC project (via the NCI Genomic Data Commons), this project:

1. Quantifies expression (reads per million miRNA mapped, RPM) of all 11 let-7 family members in normal kidney tissue (n = 20) and ccRCC tumour tissue (n = 537, Stages 1-4).
2. Tests normal-vs-cancer differences with an independent two-sample t-test.
3. Identifies **let-7b** as the member with the largest and most significant difference (p = 1.18 x 10^-7; ~2.63-fold higher on average in cancer).
4. Trains a single-feature **Support Vector Machine (SVM)** classifier on let-7b RPM to separate normal from cancerous tissue.

### Key findings

- let-7 expression **mostly increased** in ccRCC tissue (contrary to the hypothesis that let-7 would decrease); only let-7c and let-7e decreased.
- let-7b showed the largest normal-vs-cancer difference at every stage; differences between cancer stages were small.
- SVM on let-7b alone:

| Metric | Value |
|---|---|
| Accuracy | 0.907 |
| Precision (Cancer) | 0.887 |
| Recall (Cancer) | 0.936 |
| ROC-AUC | 0.931 |

See the paper (Tables 1-4, Figures 1-2) for full results and the Known Limitations section below for important caveats.

---

## Repository structure

```
let7b-ccrcc-biomarker/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md          # how to download / organise the GDC data
│   ├── raw/               # GDC files (NOT committed; see .gitignore)
│   └── processed/         # let-7 RPM table (CSV) used for analysis
├── src/
│   ├── extract_let7.py    # replaces the manual grep -> spreadsheet step
│   ├── stats_ttest.py     # per-miRNA t-tests and fold changes
│   ├── plots.py           # box plot (Fig. 1) and violin plot (Fig. 2)
│   └── train_svm.py       # SVM training and evaluation
├── models/                # saved .joblib model(s)
├── results/
│   ├── figures/
│   └── tables/
└── paper/                 # manuscript PDF / supplementary files
```

Adjust file names to match what is actually in the repo; files marked above are the recommended layout.

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/IvanR2625/let7b-ccrcc-biomarker.git
cd let7b-ccrcc-biomarker
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

The data are open-access TCGA-KIRC miRNA-Seq files from the [GDC Data Portal](https://portal.gdc.cancer.gov/). Raw files are not stored in this repo. See [`data/README.md`](data/README.md) for the exact filters and folder layout.

### 3. Run the analysis

```bash
python src/extract_let7.py       # build data/processed/let7_rpm.csv
python src/stats_ttest.py        # p-values and fold changes
python src/plots.py              # figures
python src/train_svm.py          # train + evaluate the SVM
```

---

## Methods summary

- **Data source:** GDC, project TCGA-KIRC; transcriptome profiling, miRNA-Seq, *miRNA Expression Quantification*; open access; primary site kidney; diagnosis ccRCC. 557 samples: 20 normal (solid tissue normal) and 537 tumour.
- **Expression metric:** RPM (reads per million miRNA mapped). Violin plots use log10(RPM + 1).
- **Statistics:** independent two-sample t-test per let-7 member (normal vs. cancer, and pairwise between stages for let-7b).
- **Classifier:** scikit-learn `SVC` on let-7b RPM (single feature), with SMOTE oversampling of the minority (normal) class, saved with `joblib`; predictions via `.predict` and `.predict_proba`.

---

## Known limitations

These are discussed in the paper and are repeated here so users of the code are aware of them:

- **Small normal group and class imbalance:** 20 normal vs. 537 tumour samples.
- **Single cohort:** only TCGA-KIRC; no independent validation cohort.
- **Single feature, single algorithm:** only let-7b and only SVM were tested.
- **Tissue, not urine:** the model uses kidney tissue expression. A non-invasive test would need to be retrained and validated on urine supernatant.
- **Specificity:** let-7b may also be altered in other RCC subtypes and cancers.
- **Possible data leakage from SMOTE (please review):** the evaluation set contains 215 samples (106 Normal / 109 Cancer). This is roughly 20% of a fully balanced dataset (~537 + 537), which suggests oversampling may have been applied *before* the train/test split. If so, synthetic normal samples (interpolated from only 20 real ones) could appear in the test set and inflate the metrics. Recommended fix: split first, apply SMOTE (or `class_weight="balanced"`) to the training set only, and report metrics on real held-out samples, ideally with stratified cross-validation. Treat the reported metrics as preliminary until this is checked.

---

## Roadmap

- [ ] Replace manual grep/spreadsheet extraction with a scripted pipeline
- [ ] Split-before-oversample evaluation with stratified cross-validation
- [ ] Compare SVM with logistic regression and random forest
- [ ] Multi-miRNA models
- [ ] Validate on an independent cohort (e.g., GEO) and on urine supernatant data
- [ ] Check specificity against other RCC subtypes

---

## Citation

If you use this work, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

> Raizada S., Raizada I., Cai L. *Exploring the Prognostic Potential of Let-7 MicroRNA in Clear Cell Renal Cancer.* Journal of Emerging Investigators (submitted/in press).

## Data and ethics

All data are de-identified, open-access TCGA data from the NCI GDC. Please follow the [GDC data use policies](https://gdc.cancer.gov/access-data/data-access-processes-and-tools).

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). TCGA/GDC data remain subject to their original terms.

## Acknowledgements

Thanks to Dr. Garcia for manuscript review, Jingyi Xiang for mentoring during the related science fair project, and the Ask-A-Scientist program (Linfeng Cai) for advice and insights.

## Disclaimer

This is a research prototype for education and exploration. It is **not** a validated diagnostic tool and must not be used for clinical decision-making.
