import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/01_extract_brand.py"


def test_brand_extraction(tmp_path):
    source = tmp_path / "sample.csv"
    output = tmp_path / "processed"
    audit = tmp_path / "audit.json"

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
        # A multi-turn conversation.
        ["1", "customer_a", "True", "2017-01-01",
         "Music stopped.", "2", ""],
        ["2", "SpotifyCares", "False", "2017-01-01",
         "Which device?", "3", "1"],
        ["3", "customer_a", "True", "2017-01-01",
         "My laptop.", "", "2"],

        # Unrelated tweet must not be exported.
        ["4", "OtherBrand", "False", "2017-01-01",
         "Unrelated.", "", ""],

        # Missing parent must be recorded.
        ["5", "SpotifyCares", "False", "2017-01-01",
         "A reply.", "", "999"],

        # Connection exists only in response_tweet_id.
        ["6", "customer_b", "True", "2017-01-01",
         "Another question.", "7", ""],
        ["7", "SpotifyCares", "False", "2017-01-01",
         "Another reply.", "", ""],
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
            "--db", str(tmp_path / "test.sqlite"),
            "--output-dir", str(output),
            "--audit", str(audit),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    records = [
        json.loads(line)
        for line in (output / "brand_connected_tweets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert {r["tweet_id"] for r in records} == {
        "1", "2", "3", "5", "6", "7"
    }

    report = json.loads(audit.read_text(encoding="utf-8"))

    assert report["source_rows"] == 7
    assert report["brand_authored_rows"] == 3
    assert report["connected_existing_tweets"] == 6
    assert report["missing_referenced_tweets"] == 1