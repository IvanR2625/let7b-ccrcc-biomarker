"""
Extract let-7 family miRNA RPM values from GDC miRNA Expression Quantification files.

Supported folder layouts
------------------------
Layout A (canonical, full TCGA download):
    <base>/<Stage>/<CaseID>/<FileUUID>/<FileName>.txt
    Stage names: "Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4"

Layout B (trial data, compact download):
    <base>/<Stage>/<CaseID>/[<sample_idx>/]<FileUUID>/<FileName>.txt
    Stage names: "Normal", "Stage1", "Stage2", "Stage3", "Stage4"
    (CaseID directories may have a leading space; sample_idx 1/2 for duplicates)

Usage
-----
    # canonical data
    python src/extract_let7.py

    # trial / alternate data root
    python src/extract_let7.py --data-dir user-data/Trial_1_Project_Data

Each .txt file must be a tab-separated GDC miRNA quantification file with header:
    miRNA_ID  read_count  reads_per_million_miRNA_mapped  cross-mapped

Writes: data/processed/let7_rpm.csv
    Columns: CaseID, FileUUID, Stage, miRNA, RPM
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path("data/raw/FULL_DATA")
OUT_FILE = Path("data/processed/let7_rpm.csv")

LET7_PATTERN = re.compile(r"^hsa-let-7", re.IGNORECASE)

# Normalise any stage folder name to the canonical "Stage N" / "Normal" label
_STAGE_NORM = {
    "normal": "Normal",
    "stage1": "Stage 1",
    "stage2": "Stage 2",
    "stage3": "Stage 3",
    "stage4": "Stage 4",
    "stage 1": "Stage 1",
    "stage 2": "Stage 2",
    "stage 3": "Stage 3",
    "stage 4": "Stage 4",
}


def normalise_stage(raw: str) -> str | None:
    return _STAGE_NORM.get(raw.strip().lower())


def parse_mirna_file(
    filepath: Path, case_id: str, file_uuid: str, stage: str
) -> list[dict]:
    """Return let-7 rows from one GDC miRNA quantification file."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for record in reader:
            mirna_id = record.get("miRNA_ID", "").strip()
            if not LET7_PATTERN.match(mirna_id):
                continue
            try:
                rpm = float(record["reads_per_million_miRNA_mapped"])
            except (KeyError, ValueError):
                rpm = float("nan")
            rows.append(
                {
                    "CaseID": case_id,
                    "FileUUID": file_uuid,
                    "Stage": stage,
                    "miRNA": mirna_id,
                    "RPM": rpm,
                }
            )
    return rows


def collect_all_rows(raw_dir: Path) -> list[dict]:
    """
    Recursively find all *.quantification.txt files under raw_dir.
    Infer Stage from the first path component and CaseID from the second.
    """
    all_rows = []
    if not raw_dir.exists():
        print(
            f"ERROR: {raw_dir} does not exist.\n"
            "Download the GDC data first (see data/README.md) or pass --data-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    for quant_file in sorted(raw_dir.rglob("*.quantification.txt")):
        rel = quant_file.relative_to(raw_dir)
        parts = rel.parts  # e.g. ('Stage1', 'TCGA-B0-5088', '<UUID>', '<file>.txt')

        if len(parts) < 3:
            continue

        stage_raw = parts[0]
        stage = normalise_stage(stage_raw)
        if stage is None:
            print(f"  Skipping unknown stage dir: {stage_raw}", file=sys.stderr)
            continue

        # CaseID is parts[1]; strip any accidental leading/trailing whitespace
        case_id = parts[1].strip()

        # FileUUID is the directory immediately above the .txt file
        file_uuid = quant_file.parent.name

        rows = parse_mirna_file(quant_file, case_id, file_uuid, stage)
        all_rows.extend(rows)

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Extract let-7 RPM from GDC files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Root of the stage folders (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_FILE,
        help=f"Output CSV path (default: {OUT_FILE})",
    )
    args = parser.parse_args()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {args.data_dir} …")

    rows = collect_all_rows(args.data_dir)
    if not rows:
        print("No let-7 rows found – check the data directory.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["CaseID", "FileUUID", "Stage", "miRNA", "RPM"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} rows → {args.out}")

    df = pd.read_csv(args.out)
    summary = (
        df.groupby(["Stage", "miRNA"])["RPM"]
        .count()
        .unstack("miRNA")
        .fillna(0)
        .astype(int)
    )
    print("\nSample counts per stage × miRNA:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
