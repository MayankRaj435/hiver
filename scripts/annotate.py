"""Keyboard-driven annotation tool.

Presents one example at a time and records YOUR labels.
It never proposes, predicts, or auto-fills any label value.
Future brand responses are never displayed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data/processed/prepared"
CSV_PATH = PREPARED / "annotation.csv"

INTENTS = [
    "account_access",
    "billing_subscription",
    "playback_technical",
    "library_content",
    "product_how_to",
    "other_unclear",
]

ACTIONS = ["AUTO_HANDLE", "ESCALATE"]

RISK_FLAGS = [
    "account_specific_action",
    "financial_dispute",
    "possible_account_compromise",
    "sensitive_information",
    "multi_issue",
    "insufficient_context",
    "requires_live_information",
    "language_not_understood",
]

CONFIDENCE = ["high", "medium", "low"]

LABEL_FIELDS = [
    "primary_intent",
    "required_action",
    "risk_flags",
    "acceptable_response_actions",
    "forbidden_claims",
    "label_rationale",
    "annotation_confidence",
]


class Quit(Exception):
    pass


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def choose(prompt, options, allow_multiple=False):
    """Require explicit, valid selections without suggesting labels."""
    while True:
        print(f"\n{prompt}")

        for number, option in enumerate(options, start=1):
            print(f"  {number}. {option}")

        if allow_multiple:
            print("  0. none")
            print("  Enter numbers separated by commas.")

        raw = input("> ").strip().lower()

        if raw in {"q", "quit"}:
            raise Quit

        if allow_multiple and raw == "0":
            return "none"

        parts = raw.split(",") if allow_multiple else [raw]

        try:
            numbers = [int(part.strip()) for part in parts]
        except ValueError:
            print("Enter valid option numbers.")
            continue

        if not numbers or any(
            number < 1 or number > len(options)
            for number in numbers
        ):
            print(f"Choose numbers between 1 and {len(options)}.")
            continue

        selected = [options[number - 1] for number in numbers]

        if allow_multiple:
            return ";".join(dict.fromkeys(selected))

        return selected[0]

def free_text(prompt, minimum_length=3):
    while True:
        print(f"\n{prompt}")
        raw = input("> ").strip()

        if raw.lower() in {"q", "quit"}:
            raise Quit

        if len(raw) < minimum_length:
            print(f"Please write at least {minimum_length} characters.")
            continue

        # Keep CSV values single-line for predictable round-tripping.
        return " ".join(raw.split())


def load_rows():
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def save_rows(rows, fieldnames):
    """Write to a temporary file first so an interruption cannot truncate."""
    temp_path = CSV_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(CSV_PATH)


def backup():
    directory = ROOT / "data/annotation_backups"
    directory.mkdir(parents=True, exist_ok=True)

    count = len(list(directory.glob("session_*.csv")))
    destination = directory / f"session_{count:03d}.csv"

    shutil.copy2(CSV_PATH, destination)
    return destination


def is_complete(row):
    return all(row.get(field, "").strip() for field in LABEL_FIELDS)


def annotate(row, position, total, remaining):
    clear_screen()

    print("=" * 72)
    print(f"Example {position} of {total}   |   Remaining: {remaining}")
    print(f"ID: {row['example_id']}   Split: {row['split']}")
    print("=" * 72)

    context = row.get("prior_context", "").strip()

    if context:
        print("\nPRIOR CONTEXT")
        print("-" * 72)
        print(context)

    print("\nCUSTOMER MESSAGE")
    print("-" * 72)
    print(row["customer_message"])
    print("-" * 72)
    print("\nType q at any prompt to save and exit.")

    intent = choose("Primary intent:", INTENTS)
    action = choose("Required action:", ACTIONS)
    flags = choose("Risk flags:", RISK_FLAGS, allow_multiple=True)

    acceptable = free_text(
        "Acceptable response actions "
        "(for example: ask which device; explain secure support)."
    )

    forbidden = free_text(
        "Forbidden claims for this case "
        "(type none_specific if nothing case-specific applies)."
    )

    rationale = free_text("Why did you choose this action?", 10)
    confidence = choose("Your label confidence:", CONFIDENCE)

    row.update(
        {
            "primary_intent": intent,
            "required_action": action,
            "risk_flags": flags,
            "acceptable_response_actions": acceptable,
            "forbidden_claims": forbidden,
            "label_rationale": rationale,
            "annotation_confidence": confidence,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "representative_test", "stress_test", "all"],
        default="dev",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"Annotation sheet not found: {CSV_PATH}")

    rows, fieldnames = load_rows()

    if not rows:
        raise SystemExit("Annotation sheet is empty.")

    targets = [
        row
        for row in rows
        if (args.split == "all" or row["split"] == args.split)
        and not is_complete(row)
    ]

    if args.limit > 0:
        targets = targets[: args.limit]

    if not targets:
        print(f"No unlabelled rows remain for split: {args.split}")
        return

    total = len(targets)
    completed = 0

    try:
        for position, row in enumerate(targets, start=1):
            annotate(row, position, total, total - position + 1)

            # Save after every example so nothing is lost.
            save_rows(rows, fieldnames)
            completed += 1

    except (Quit, KeyboardInterrupt):
        print("\nStopping early.")

    finally:
        save_rows(rows, fieldnames)
        destination = backup()

        finished = sum(
            1
            for row in rows
            if (args.split == "all" or row["split"] == args.split)
            and is_complete(row)
        )

        print(f"\nLabelled this session: {completed}")
        print(f"Total complete in split {args.split}: {finished}")
        print(f"Saved: {CSV_PATH}")
        print(f"Backup: {destination}")


if __name__ == "__main__":
    main()