"""Validate annotation structure without generating or changing human labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INTENTS = {
    "account_access",
    "billing_subscription",
    "playback_technical",
    "library_content",
    "product_how_to",
    "other_unclear",
}

ACTIONS = {"AUTO_HANDLE", "ESCALATE"}

RISK_FLAGS = {
    "account_specific_action",
    "financial_dispute",
    "possible_account_compromise",
    "sensitive_information",
    "multi_issue",
    "insufficient_context",
    "requires_live_information",
    "language_not_understood",
}

LABEL_FIELDS = {
    "primary_intent",
    "required_action",
    "risk_flags",
    "acceptable_response_actions",
    "forbidden_claims",
    "label_rationale",
    "annotation_confidence",
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "all"],
        default="dev",
    )
    args = parser.parse_args()

    directory = ROOT / "data/processed/prepared"
    input_path = directory / "evaluation_inputs.jsonl"

    audit = json.loads(
        (ROOT / "artifacts/preparation_audit.json").read_text(
            encoding="utf-8"
        )
    )

    actual_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()

    if actual_hash != audit["evaluation_inputs_sha256"]:
        raise SystemExit(
            "ERROR: Frozen evaluation input hash changed. "
            "Investigate before continuing."
        )

    inputs = {
        record["example_id"]: record
        for record in load_jsonl(input_path)
    }

    manifest = {
        record["example_id"]: record
        for record in load_jsonl(directory / "sampling_manifest.jsonl")
    }

    with (directory / "annotation.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)

        required_columns = LABEL_FIELDS | {
            "example_id",
            "split",
            "conversation_id",
            "customer_message",
            "prior_context",
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise SystemExit(
                f"ERROR: Missing columns: {sorted(missing_columns)}"
            )

        rows = list(reader)

    errors = []
    seen = set()
    completed = 0
    selected_count = 0

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            errors.append(f"CSV row {row_number}: malformed record")
            continue

        example_id = row["example_id"]

        if example_id in seen:
            errors.append(f"{example_id}: duplicate annotation row")

        seen.add(example_id)

        if example_id not in inputs:
            errors.append(f"{example_id}: unknown example ID")
            continue

        original = inputs[example_id]
        expected_split = manifest[example_id]["split"]

        if row["split"] != expected_split:
            errors.append(f"{example_id}: split was changed")

        if row["conversation_id"] != original["conversation_id"]:
            errors.append(f"{example_id}: conversation ID was changed")

        if normalized_newlines(row["customer_message"]) != (
            normalized_newlines(original["customer_message"])
        ):
            errors.append(f"{example_id}: customer message was changed")

        expected_context = "\n".join(
            f"{turn['role'].upper()}: {turn['text']}"
            for turn in original["prior_context"]
        )

        if normalized_newlines(row["prior_context"]) != (
            normalized_newlines(expected_context)
        ):
            errors.append(f"{example_id}: prior context was changed")

        if args.split == "dev" and expected_split != "dev":
            continue

        selected_count += 1

        missing_labels = [
            field for field in sorted(LABEL_FIELDS)
            if not row[field].strip()
        ]

        if missing_labels:
            errors.append(
                f"{example_id}: missing labels {missing_labels}"
            )
            continue

        completed += 1

        if row["primary_intent"].strip() not in INTENTS:
            errors.append(f"{example_id}: invalid primary_intent")

        if row["required_action"].strip() not in ACTIONS:
            errors.append(f"{example_id}: invalid required_action")

        if row["annotation_confidence"].strip() not in {
            "high", "medium", "low"
        }:
            errors.append(f"{example_id}: invalid annotation_confidence")

        flags = {
            flag.strip()
            for flag in row["risk_flags"].split(";")
            if flag.strip()
        }

        if flags != {"none"} and not flags.issubset(RISK_FLAGS):
            errors.append(f"{example_id}: invalid risk_flags {sorted(flags)}")

    missing_ids = set(inputs) - seen

    if missing_ids:
        errors.append(f"Missing annotation rows: {sorted(missing_ids)}")

    if len(rows) != 150:
        errors.append(f"Expected 150 rows, found {len(rows)}")

    print(f"Selected split: {args.split}")
    print(f"Rows with all label fields filled: {completed}/{selected_count}")

    if errors:
        print(f"\nValidation issues: {len(errors)}")
        for error in errors[:40]:
            print(f"- {error}")

        if len(errors) > 40:
            print(f"... and {len(errors) - 40} additional issues")

        raise SystemExit(1)

    print("Validation passed.")
    print("This validates structure, not the correctness of human judgments.")


if __name__ == "__main__":
    main()