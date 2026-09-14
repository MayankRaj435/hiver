"""Run both baselines on a chosen split.

Trains only on development labels.
Produces predictions in the same schema the evaluator will consume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import PREDICTORS, build_simple, build_trivial

LABEL_FIELDS = [
    "primary_intent",
    "required_action",
    "risk_flags",
    "acceptable_response_actions",
    "forbidden_claims",
    "label_rationale",
    "annotation_confidence",
]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_labels():
    path = ROOT / "data/processed/prepared/annotation.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    complete = {
        row["example_id"]: row
        for row in rows
        if all(row.get(field, "").strip() for field in LABEL_FIELDS)
    }
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=[
            "dev", "representative_test", "stress_test", "all",
        ],
        default="dev",
    )
    parser.add_argument("--systems", default="trivial,simple")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    for name in systems:
        if name not in PREDICTORS:
            raise SystemExit(f"Unknown system: {name}")

    prepared = ROOT / "data/processed/prepared"
    inputs_path = prepared / "evaluation_inputs.jsonl"

    audit = json.loads(
        (ROOT / "artifacts/preparation_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if sha256(inputs_path) != audit["evaluation_inputs_sha256"]:
        raise SystemExit("Frozen evaluation input hash changed.")

    labels = load_labels()
    dev_labels = [
        label for label in labels.values()
        if label["split"] == "dev"
    ]

    if len(dev_labels) < 30:
        raise SystemExit(
            f"Need 30 complete development labels; found {len(dev_labels)}. "
            "Finish development annotation first."
        )

    states = {}
    if "trivial" in systems:
        states["trivial"] = build_trivial(dev_labels)
    if "simple" in systems:
        states["simple"] = build_simple(dev_labels)

    retriever = joblib.load(ROOT / "artifacts/retrieval/index.joblib")

    # Verify no overlap between retrieval index and evaluation inputs.
    evaluation_conversations = {
        item["conversation_id"]
        for item in read_jsonl(inputs_path)
    }
    evidence_conversations = {
        item["conversation_id"] for item in retriever.records
    }
    if evaluation_conversations & evidence_conversations:
        raise SystemExit("Index overlaps evaluation conversations.")

    manifest = {
        item["example_id"]: item
        for item in read_jsonl(prepared / "sampling_manifest.jsonl")
    }

    inputs = read_jsonl(inputs_path)
    selected = [
        item
        for item in inputs
        if args.split == "all"
        or manifest[item["example_id"]]["split"] == args.split
    ]

    if args.limit > 0:
        selected = selected[: args.limit]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out_dir = ROOT / "artifacts/baselines" / run_id
    out_dir.mkdir(parents=True)

    metadata = {
        "run_id": run_id,
        "split": args.split,
        "systems": systems,
        "dev_labels_used": len(dev_labels),
        "examples": len(selected),
        "note": (
            "If split is dev, intent results are IN-SAMPLE and are only a "
            "pipeline smoke test, not a performance result."
        ),
        "evaluation_inputs_sha256": sha256(inputs_path),
        "retrieval_index_sha256": sha256(
            ROOT / "artifacts/retrieval/index.joblib"
        ),
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    with (out_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for example in selected:
            for system in systems:
                record = PREDICTORS[system](
                    states[system], example, retriever
                )
                record["example_id"] = example["example_id"]
                record["split"] = manifest[example["example_id"]]["split"]
                handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    # Print summary.
    for system in systems:
        preds = [
            r for r in read_jsonl(out_dir / "predictions.jsonl")
            if r["system"] == system
        ]
        auto = sum(1 for r in preds if r.get("send_eligible"))
        print(
            f"{system}: {len(preds)} examples, "
            f"{auto} auto-handled ({auto / max(len(preds), 1):.1%})"
        )

    print(f"\nSaved: {out_dir}")

    if args.split == "dev":
        print(
            "NOTE: dev is in-sample for the simple baseline. "
            "This run is a smoke test only."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
