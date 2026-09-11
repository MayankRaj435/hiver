"""Extract all tweets connected to a selected support brand.

Memory strategy:
- Stream the full CSV.
- Store tweets and reply edges in SQLite.
- Export selected tweets without loading them all into Python memory.

This script does NOT create model inputs or evaluation splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/twcs.csv",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data/processed/twitter.sqlite",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts/extraction_audit.json",
    )
    parser.add_argument("--brand", default="SpotifyCares")
    parser.add_argument("--rebuild", action="store_true")

    return parser.parse_args()


def import_csv(connection, source):
    """Import records and bidirectional grouping edges in small batches."""
    csv.field_size_limit(10_000_000)

    connection.executescript(
        """
        CREATE TABLE tweets (
            tweet_id TEXT PRIMARY KEY,
            author_id TEXT NOT NULL,
            inbound INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            text TEXT NOT NULL,
            response_tweet_id TEXT NOT NULL,
            in_response_to_tweet_id TEXT
        );

        CREATE TABLE edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            PRIMARY KEY (source_id, target_id)
        ) WITHOUT ROWID;
        """
    )

    tweet_batch = []
    edge_batch = []
    total = 0

    def flush():
        with connection:
            # Fail on duplicate tweet IDs instead of silently overwriting.
            connection.executemany(
                "INSERT INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?)",
                tweet_batch,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO edges VALUES (?, ?)",
                edge_batch,
            )

        tweet_batch.clear()
        edge_batch.clear()

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        header = reader.fieldnames or []

        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV column names.")

        missing = set(COLUMNS) - set(header)

        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(
                    f"Malformed CSV near physical line {reader.line_num}"
                )

            tweet_id = row["tweet_id"].strip()
            inbound = row["inbound"].strip().lower()
            parent_id = row["in_response_to_tweet_id"].strip() or None
            response_field = row["response_tweet_id"].strip()

            if not tweet_id:
                raise ValueError("Encountered a missing tweet ID.")

            if inbound not in {"true", "false"}:
                raise ValueError(f"Unexpected inbound value: {inbound!r}")

            tweet_batch.append(
                (
                    tweet_id,
                    row["author_id"].strip(),
                    int(inbound == "true"),
                    row["created_at"],
                    row["text"],
                    response_field,
                    parent_id,
                )
            )

            neighbor_ids = [
                item.strip()
                for item in response_field.split(",")
                if item.strip()
            ]

            if parent_id is not None:
                neighbor_ids.append(parent_id)

            for neighbor_id in neighbor_ids:
                if neighbor_id != tweet_id:
                    # Bidirectional edges are for grouping only.
                    # They do not establish valid prior context.
                    edge_batch.append((tweet_id, neighbor_id))
                    edge_batch.append((neighbor_id, tweet_id))

            total += 1

            if len(tweet_batch) >= 20_000:
                flush()

            if total % 250_000 == 0:
                print(f"Imported {total:,} tweets...", flush=True)

        if tweet_batch:
            flush()

    if total == 0:
        raise ValueError("CSV contains no data records.")

    with connection:
        connection.execute(
            "CREATE INDEX author_idx ON tweets(author_id)"
        )

    return total


def select_connected_tweets(connection, brand):
    """Select graph nodes reachable from any tweet by the brand."""
    brand_count = connection.execute(
        "SELECT COUNT(*) FROM tweets WHERE author_id = ?",
        (brand,),
    ).fetchone()[0]

    if brand_count == 0:
        raise ValueError(f"No exact author match for {brand!r}.")

    print(
        f"Found {brand_count:,} {brand} tweets. "
        "Following reply links...",
        flush=True,
    )

    with connection:
        connection.execute(
            "CREATE TABLE selected_nodes (tweet_id TEXT PRIMARY KEY)"
        )

        # UNION deduplicates nodes, so cycles do not cause infinite traversal.
        connection.execute(
            """
            INSERT INTO selected_nodes(tweet_id)
            WITH RECURSIVE reachable(tweet_id) AS (
                SELECT tweet_id
                FROM tweets
                WHERE author_id = ?

                UNION

                SELECT e.target_id
                FROM edges AS e
                JOIN reachable AS r ON e.source_id = r.tweet_id
            )
            SELECT tweet_id FROM reachable
            """,
            (brand,),
        )

    return brand_count


def export_data(connection, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    tweets_path = output_dir / "brand_connected_tweets.jsonl"
    missing_path = output_dir / "missing_referenced_tweet_ids.jsonl"

    exported = 0
    inbound_count = 0

    query = """
        SELECT t.tweet_id, t.author_id, t.inbound, t.created_at,
               t.text, t.response_tweet_id, t.in_response_to_tweet_id
        FROM tweets AS t
        JOIN selected_nodes AS s ON s.tweet_id = t.tweet_id
        ORDER BY t.tweet_id
    """

    with tweets_path.open("w", encoding="utf-8") as handle:
        for row in connection.execute(query):
            record = dict(zip(COLUMNS, row))
            record["inbound"] = bool(record["inbound"])

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            exported += 1
            inbound_count += int(record["inbound"])

    missing_count = 0

    with missing_path.open("w", encoding="utf-8") as handle:
        for (tweet_id,) in connection.execute(
            """
            SELECT s.tweet_id
            FROM selected_nodes AS s
            LEFT JOIN tweets AS t ON t.tweet_id = s.tweet_id
            WHERE t.tweet_id IS NULL
            ORDER BY s.tweet_id
            """
        ):
            handle.write(json.dumps({"tweet_id": tweet_id}) + "\n")
            missing_count += 1

    return exported, inbound_count, missing_count


def main():
    args = parse_args()

    if not args.input.is_file() or args.input.stat().st_size == 0:
        raise SystemExit(f"Missing or empty dataset: {args.input}")

    if args.input.resolve() == args.db.resolve():
        raise SystemExit("CSV and database paths must differ.")

    if args.db.exists():
        if not args.rebuild:
            raise SystemExit(
                f"Database already exists: {args.db}\n"
                "To recreate this generated database, run with --rebuild."
            )
        args.db.unlink()

    args.db.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    connection = sqlite3.connect(args.db)

    # Approximately 32 MiB for SQLite's page cache.
    connection.execute("PRAGMA cache_size = -32768")
    connection.execute("PRAGMA temp_store = FILE")

    try:
        total = import_csv(connection, args.input)
        brand_count = select_connected_tweets(connection, args.brand)

        exported, inbound_count, missing_count = export_data(
            connection,
            args.output_dir,
        )

        audit = {
            "brand": args.brand,
            "source_filename": args.input.name,
            "source_size_bytes": args.input.stat().st_size,
            "source_rows": total,
            "brand_authored_rows": brand_count,
            "connected_existing_tweets": exported,
            "connected_inbound_tweets": inbound_count,
            "missing_referenced_tweets": missing_count,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "duplicate_id_policy": "Fail on duplicate tweet IDs.",
            "graph_policy": (
                "Follow both reply-reference columns in both directions "
                "for grouping; retain missing referenced IDs."
            ),
            "limitations": [
                "These are tweet counts, not conversation counts.",
                "Connected graphs may contain multiple brands.",
                "Connected graphs may contain branches.",
                "Exported source records contain future replies.",
                "No model inputs, labels, or evaluation splits exist yet.",
            ],
        }

        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(
            json.dumps(audit, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\nExtraction complete")
        print(f"Connected existing tweets: {exported:,}")
        print(f"Missing referenced tweets: {missing_count:,}")
        print(f"Exports: {args.output_dir}")
        print(f"Audit: {args.audit}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()