"""Fast-mode annotation: one line per example, same schema, same CSV.

Every value is YOUR judgment — nothing is auto-suggested or pre-filled.

Format:  intent|action|flags|acceptable|forbidden|rationale|confidence

Example: 3|1|0|acknowledge screenshot and troubleshoot|none_specific|safe technical followup|1

Keys:
  intent:     1-6 (account billing playback library howto unclear)
  action:     1=AUTO  2=ESCALATE
  flags:      0=none, or digits like 1,3,5
  acceptable: short text
  forbidden:  short text or none_specific
  rationale:  short text
  confidence: 1=high 2=medium 3=low
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/processed/prepared/annotation.csv"

INTENTS = {
    1: "account_access",
    2: "billing_subscription",
    3: "playback_technical",
    4: "library_content",
    5: "product_how_to",
    6: "other_unclear",
}

ACTIONS = {1: "AUTO_HANDLE", 2: "ESCALATE"}
CONFIDENCE = {1: "high", 2: "medium", 3: "low"}

RISK_FLAGS = {
    1: "account_specific_action",
    2: "financial_dispute",
    3: "possible_account_compromise",
    4: "sensitive_information",
    5: "multi_issue",
    6: "insufficient_context",
    7: "requires_live_information",
    8: "language_not_understood",
}

LABEL_FIELDS = [
    "primary_intent", "required_action", "risk_flags",
    "acceptable_response_actions", "forbidden_claims",
    "label_rationale", "annotation_confidence",
]

GUIDE = """
 INTENT                    FLAGS                         ACTION
 1 account_access          1 account_specific_action     1 = AUTO_HANDLE
 2 billing_subscription    2 financial_dispute            2 = ESCALATE
 3 playback_technical      3 possible_account_compromise
 4 library_content         4 sensitive_information       CONFIDENCE
 5 product_how_to          5 multi_issue                 1 = high
 6 other_unclear           6 insufficient_context        2 = medium
                           7 requires_live_information   3 = low
                           8 language_not_understood

 FORMAT: intent|action|flags|acceptable|forbidden|rationale|confidence
 EXAMPLE: 3|1|0|ask device and app version|none_specific|safe playback clarification|1
 Type q to save and quit.
"""


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def load():
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def save(rows, fields):
    tmp = CSV_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(CSV_PATH)


def backup():
    d = ROOT / "data/annotation_backups"
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("fast_*.csv")))
    dst = d / f"fast_{n:03d}.csv"
    shutil.copy2(CSV_PATH, dst)
    return dst


def done(row):
    return all(row.get(k, "").strip() for k in LABEL_FIELDS)


def parse_flags(raw):
    raw = raw.strip()
    if raw in ("0", "", "none"):
        return "none"
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for p in parts:
        n = int(p)
        if n not in RISK_FLAGS:
            raise ValueError(f"flag {n} not in 1-8")
        out.append(RISK_FLAGS[n])
    return ";".join(dict.fromkeys(out))


def parse_line(raw):
    """Parse one-line input into label dict. Raises ValueError on bad input."""
    parts = raw.split("|")
    if len(parts) != 7:
        raise ValueError(f"Need 7 fields separated by |, got {len(parts)}")

    i = int(parts[0].strip())
    a = int(parts[1].strip())
    c = int(parts[6].strip())

    if i not in INTENTS:
        raise ValueError(f"intent must be 1-6, got {i}")
    if a not in ACTIONS:
        raise ValueError(f"action must be 1 or 2, got {a}")
    if c not in CONFIDENCE:
        raise ValueError(f"confidence must be 1-3, got {c}")

    flags = parse_flags(parts[2])
    acceptable = " ".join(parts[3].split())
    forbidden = " ".join(parts[4].split())
    rationale = " ".join(parts[5].split())

    if len(acceptable) < 3:
        raise ValueError("acceptable too short (min 3 chars)")
    if len(forbidden) < 3:
        raise ValueError("forbidden too short (min 3 chars, or 'none_specific')")
    if len(rationale) < 5:
        raise ValueError("rationale too short (min 5 chars)")

    return {
        "primary_intent": INTENTS[i],
        "required_action": ACTIONS[a],
        "risk_flags": flags,
        "acceptable_response_actions": acceptable,
        "forbidden_claims": forbidden,
        "label_rationale": rationale,
        "annotation_confidence": CONFIDENCE[c],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev",
                        choices=["dev", "representative_test", "stress_test", "all"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows, fields = load()
    targets = [
        r for r in rows
        if (args.split == "all" or r["split"] == args.split) and not done(r)
    ]
    if args.limit > 0:
        targets = targets[:args.limit]

    if not targets:
        print(f"Nothing left to label in {args.split}.")
        return

    total = len(targets)
    session_done = 0
    last_line = ""

    try:
        for i, row in enumerate(targets, 1):
            clear()
            print(f"{'=' * 72}")
            print(f"  {i}/{total}  |  {row['example_id']}  [{row['split']}]")
            print(f"{'=' * 72}")

            ctx = (row.get("prior_context") or "").strip()
            if ctx:
                print("\nCONTEXT")
                print(ctx[:800])

            print("\nMESSAGE")
            print((row.get("customer_message") or "").strip()[:800])

            print(GUIDE)

            if last_line:
                print(f"  (last: {last_line})")

            while True:
                raw = input("> ").strip()
                if raw.lower() in ("q", "quit"):
                    raise KeyboardInterrupt

                try:
                    labels = parse_line(raw)
                    break
                except (ValueError, KeyError) as e:
                    print(f"  Error: {e}. Try again.")

            row.update(labels)
            save(rows, fields)
            session_done += 1
            last_line = raw
            print(f"  Saved. ({session_done} this session)")

    except (KeyboardInterrupt, EOFError):
        print("\nStopping.")

    finally:
        save(rows, fields)
        backup()
        filled = sum(
            1 for r in rows
            if (args.split == "all" or r["split"] == args.split) and done(r)
        )
        print(f"\nSession: {session_done} | Total complete in {args.split}: {filled}")
        print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
