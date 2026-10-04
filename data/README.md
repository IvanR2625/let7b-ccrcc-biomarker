# Data

Raw data are **not** included in this repository. They are open-access and can be downloaded from the [NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/).

## Download steps

1. Go to the GDC Data Portal -> **Repository**.
2. Apply filters:
   - **Project:** TCGA-KIRC
   - **Data Category:** Transcriptome Profiling
   - **Data Type:** miRNA Expression Quantification
   - **Experimental Strategy:** miRNA-Seq
   - **Access:** open
   - **Primary Site:** kidney; **Diagnosis:** clear cell renal cell carcinoma
3. Add the tumour files to the cart; download the manifest and metadata (sample sheet).
4. Repeat with the sample type filter set to **Solid Tissue Normal** for the 20 normal controls.
5. Download with the [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool):

```bash
gdc-client download -m gdc_manifest.txt -d data/raw/
```

## Expected counts

| Group | Samples |
|---|---|
| Normal (solid tissue normal) | 20 |
| ccRCC tumour (Stages 1-4) | 537 |
| **Total** | **557** |

Stage counts for let-7b (Figure 1): Stage 1 = 242, Stage 2 = 65, Stage 3 = 130, Stage 4 = 100.

## Folder layout used in the project

```
data/raw/FULL_DATA/
└── <Stage>/                 # Normal, Stage 1, Stage 2, Stage 3, Stage 4
    └── <CaseID>/            # unzipped sample folder, renamed to CaseID
        └── <FileUUID>/
            └── <FileName>.txt   # miRNA expression quantification file
```

Each `.txt` file has one line per miRNA (e.g. `hsa-let-7b`), so let-7 values can be pulled with:

```bash
grep -r "hsa-let-7b" data/raw/FULL_DATA/
```

(`src/extract_let7.py` replaces this manual step and writes `data/processed/let7_rpm.csv`.)

## File format

GDC miRNA Expression Quantification files are tab-separated with a header row:

```
miRNA_ID	read_count	reads_per_million_miRNA_mapped	cross-mapped
hsa-let-7a-1	12345	1234.56	N
...
```

The column used for analysis is `reads_per_million_miRNA_mapped` (RPM).

## Full let-7 spreadsheet

The compiled spreadsheet used in the paper is linked in the manuscript appendix. Consider exporting it to CSV and committing it as `data/processed/let7_rpm.csv` so results can be reproduced without re-downloading from GDC.
