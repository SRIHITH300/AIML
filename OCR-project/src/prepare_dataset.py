"""
Dataset preparation module.

Walks through dataset/raw/<category>/ folders, runs OCR on each document,
and produces a structured CSV at dataset/processed/documents.csv.

Columns: filename, category, text, text_length
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr import IMAGE_EXTENSIONS, PDF_EXTENSIONS, clean_text, extract_text_from_file

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
CSV_PATH = PROCESSED_DIR / "documents.csv"

SUPPORTED_EXTS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def discover_categories(raw_dir: Path) -> list:
    """Return sorted list of category folder names."""
    categories = []
    for entry in sorted(raw_dir.iterdir()):
        if entry.is_dir():
            categories.append(entry.name)
    return categories


def list_documents(raw_dir: Path) -> list:
    """Return list of (category, filepath) tuples for all supported documents."""
    docs = []
    for cat_dir in sorted(raw_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for fpath in sorted(cat_dir.iterdir()):
            if fpath.is_file() and fpath.suffix.lower() in SUPPORTED_EXTS:
                docs.append((category, fpath))
    return docs


def load_existing_csv(csv_path: Path) -> dict:
    """Load existing CSV into a dict keyed by (category, filename) -> row."""
    existing = {}
    if not csv_path.exists():
        return existing
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("category", ""), row.get("filename", ""))
            existing[key] = row
    return existing


def file_hash(filepath: Path) -> str:
    """Compute a short hash of file content for duplicate detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def build_dataset(limit: int = 0, resume: bool = True) -> int:
    """
    Build the dataset CSV.

    Args:
        limit: Max documents per category (0 = all).
        resume: Skip files already present in the CSV.
    Returns:
        Number of new rows added.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    categories = discover_categories(RAW_DIR)
    if not categories:
        print(f"No category folders found in {RAW_DIR}")
        return 0

    print(f"Categories found: {categories}")
    docs = list_documents(RAW_DIR)
    print(f"Total documents found: {len(docs)}")

    existing = load_existing_csv(CSV_PATH) if resume else {}
    print(f"Existing rows in CSV: {len(existing)}")

    seen_hashes = set()
    new_rows = []
    skipped = 0
    empty = 0
    duplicates = 0

    for category, fpath in docs:
        fname = fpath.name
        key = (category, fname)

        if resume and key in existing:
            skipped += 1
            continue

        try:
            raw_text = extract_text_from_file(str(fpath))
        except Exception as e:
            print(f"  ERROR extracting {category}/{fname}: {e}")
            raw_text = ""

        text = clean_text(raw_text)

        if not text:
            empty += 1
            continue

        fh = file_hash(fpath)
        if fh in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(fh)

        new_rows.append({
            "filename": fname,
            "category": category,
            "text": text,
            "text_length": len(text),
        })

    if new_rows:
        write_header = not CSV_PATH.exists() or not resume
        mode = "a" if (resume and CSV_PATH.exists()) else "w"
        with open(CSV_PATH, mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "category", "text", "text_length"])
            if write_header:
                writer.writeheader()
            for row in new_rows:
                writer.writerow(row)

    print(f"\nDataset build complete:")
    print(f"  New rows added: {len(new_rows)}")
    print(f"  Skipped (already in CSV): {skipped}")
    print(f"  Empty text (dropped): {empty}")
    print(f"  Duplicates (dropped): {duplicates}")
    print(f"  CSV path: {CSV_PATH}")

    return len(new_rows)


def summarize_csv(csv_path: Path) -> None:
    """Print a summary of the CSV."""
    if not csv_path.exists():
        print("CSV does not exist yet.")
        return

    counts = {}
    total = 0
    total_len = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            counts[cat] = counts.get(cat, 0) + 1
            total += 1
            total_len += int(row.get("text_length", 0))

    print(f"\nDataset summary ({csv_path.name}):")
    print(f"  Total documents: {total}")
    for cat, cnt in sorted(counts.items()):
        print(f"    {cat}: {cnt}")
    if total:
        print(f"  Average text length: {total_len // total} chars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build document classification dataset from OCR.")
    parser.add_argument("--limit", type=int, default=0, help="Max docs per category (0 = all).")
    parser.add_argument("--no-resume", action="store_true", help="Rebuild from scratch.")
    args = parser.parse_args()

    added = build_dataset(limit=args.limit, resume=not args.no_resume)
    summarize_csv(CSV_PATH)
