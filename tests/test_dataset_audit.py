import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/00_audit_dataset.py"


def test_dataset_audit(tmp_path):
    source = tmp_path / "sample.csv"
    output = tmp_path / "audit.json"

    columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]

    rows = [
        [
            "1", "customer_1", "True", "2017-01-01",
            "My music stopped.", "2", "",
        ],
        [
            "2", "SpotifyCares", "False", "2017-01-01",
            "Which device are you using?", "", "1",
        ],
        [
            "3", "AnotherBrand", "False", "2017-01-01",
            "Unrelated reply.", "", "",
        ],
    ]

    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input", str(source),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["total_rows"] == 3
    assert report["brand_rows"] == 1
    assert report["brand_rows_with_parent_reference"] == 1
    assert report["inbound_counts"] == {"true": 1, "false": 2}


def test_missing_dataset(tmp_path):
    output = tmp_path / "audit.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input", str(tmp_path / "missing.csv"),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "CSV not found" in result.stderr
    assert not output.exists()