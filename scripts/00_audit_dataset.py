"""Stream the source CSV and write a structural audit.

No tweet text is included in the output report.
This does not reconstruct conversations or create evaluation labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def audit_dataset(source: Path, brand: str) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"CSV not found: {source}")

    csv.field_size_limit(10_000_000)

    start = time.perf_counter()
    rows = 0
    brand_rows = 0
    brand_rows_with_parent = 0

    missing_fields = Counter()
    inbound_counts = Counter()
    brand_inbound_counts = Counter()
    spotify_authors = Counter()

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        columns = reader.fieldnames or []

        if len(columns) != len(set(columns)):
            raise ValueError("CSV contains duplicate column names.")

        missing_columns = REQUIRED_COLUMNS - set(columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"Malformed CSV record near line {reader.line_num}"
                )

            rows += 1

            for column in REQUIRED_COLUMNS:
                if not row[column].strip():
                    missing_fields[column] += 1

            author = row["author_id"].strip()
            inbound = row["inbound"].strip().lower()
            inbound_counts[inbound or "<blank>"] += 1

            if "spotify" in author.lower():
                spotify_authors[author] += 1

            # Exact matching is also what our extraction will use.
            if author == brand:
                brand_rows += 1
                brand_inbound_counts[inbound or "<blank>"] += 1

                if row["in_response_to_tweet_id"].strip():
                    brand_rows_with_parent += 1

            if rows % 500_000 == 0:
                print(
                    f"Scanned {rows:,} rows; "
                    f"found {brand_rows:,} {brand} tweets.",
                    flush=True,
                )

    if rows == 0:
        raise ValueError("CSV contains no data rows.")

    warnings = []

    if brand_rows == 0:
        warnings.append(
            f"No exact author match for {brand!r}. "
            "Inspect spotify_author_counts."
        )

    unexpected_inbound = set(inbound_counts) - {"true", "false"}

    if unexpected_inbound:
        warnings.append(
            f"Unexpected inbound encodings: {sorted(unexpected_inbound)}"
        )

    return {
        "audit_version": 1,
        "source_filename": source.name,
        "source_size_bytes": source.stat().st_size,
        "columns": columns,
        "total_rows": rows,
        "brand": brand,
        "brand_rows": brand_rows,
        "brand_rows_with_parent_reference": brand_rows_with_parent,
        "inbound_counts": dict(inbound_counts),
        "brand_inbound_counts": dict(brand_inbound_counts),
        "spotify_author_counts": dict(spotify_authors),
        "blank_field_counts": {
            key: missing_fields[key]
            for key in sorted(REQUIRED_COLUMNS)
        },
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "warnings": warnings,
        "limitations": [
            "Tweet rows are not conversation counts.",
            "Missing parent fields can be legitimate root tweets.",
            "Referenced tweet existence has not yet been checked.",
            "Duplicate IDs have not yet been checked.",
            "No model-input or evaluation splits exist yet.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/twcs.csv",
    )
    parser.add_argument("--brand", default="SpotifyCares")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/dataset_audit.json",
    )
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        print("ERROR: Input and output paths must differ.")
        return 1

    try:
        report = audit_dataset(args.input, args.brand)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nAudit complete")
    print(f"Total rows: {report['total_rows']:,}")
    print(f"Brand rows: {report['brand_rows']:,}")
    print(f"Report: {args.output}")

    for warning in report["warnings"]:
        print(f"WARNING: {warning}")

    return 0 if report["brand_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())