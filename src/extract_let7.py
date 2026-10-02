"""
Extract let-7 family miRNA RPM values from GDC miRNA Expression Quantification files.

Expected folder layout:
    data/raw/FULL_DATA/<Stage>/<CaseID>/<FileUUID>/<FileName>.txt

Each .txt file is a tab-separated GDC miRNA quantification file with header:
    miRNA_ID  read_count  reads_per_million_miRNA_mapped  cross-mapped

Writes: data/processed/let7_rpm.csv
    Columns: CaseID, FileUUID, Stage, miRNA, RPM
"""

import re
import csv
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/FULL_DATA")
OUT_FILE = Path("data/processed/let7_rpm.csv")

LET7_PATTERN = re.compile(r"^hsa-let-7", re.IGNORECASE)
VALID_STAGES = {"Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4"}


def parse_mirna_file(
    filepath: Path, case_id: str, file_uuid: str, stage: str
) -> list[dict]:
    """Return let-7 rows from one GDC miRNA quantification .txt file."""
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


def collect_all_rows() -> list[dict]:
    all_rows = []
    if not RAW_DIR.exists():
        print(
            f"ERROR: {RAW_DIR} does not exist. "
            "Download the GDC data first (see data/README.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    for stage_dir in sorted(RAW_DIR.iterdir()):
        stage = stage_dir.name
        if not stage_dir.is_dir():
            continue
        if stage not in VALID_STAGES:
            print(f"  Skipping unexpected directory: {stage_dir}", file=sys.stderr)
            continue

        for case_dir in sorted(stage_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name

            for uuid_dir in sorted(case_dir.iterdir()):
                if not uuid_dir.is_dir():
                    continue
                file_uuid = uuid_dir.name

                txt_files = list(uuid_dir.glob("*.txt"))
                if not txt_files:
                    print(f"  WARNING: no .txt in {uuid_dir}", file=sys.stderr)
                    continue

                rows = parse_mirna_file(txt_files[0], case_id, file_uuid, stage)
                all_rows.extend(rows)

    return all_rows


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {RAW_DIR} …")

    rows = collect_all_rows()
    if not rows:
        print(
            "No let-7 rows found – check that RAW_DIR points to the correct location.",
            file=sys.stderr,
        )
        sys.exit(1)

    fieldnames = ["CaseID", "FileUUID", "Stage", "miRNA", "RPM"]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} rows → {OUT_FILE}")

    df = pd.read_csv(OUT_FILE)
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
