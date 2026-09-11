"""Build a local retrieval index without reading human labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval import HistoricalRetriever


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=5000)
    args = parser.parse_args()

    if args.max_cases < 1:
        raise SystemExit("--max-cases must be positive.")

    prepared = ROOT / "data/processed/prepared"
    evaluation_path = prepared / "evaluation_inputs.jsonl"

    preparation_audit = json.loads(
        (ROOT / "artifacts/preparation_audit.json").read_text(
            encoding="utf-8"
        )
    )

    digest = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()

    if digest != preparation_audit["evaluation_inputs_sha256"]:
        raise SystemExit("Frozen evaluation input hash changed.")

    historical = read_jsonl(
        prepared / "historical_evidence_pool.jsonl"
    )
    evaluation = read_jsonl(evaluation_path)

    evaluation_conversations = {
        case["conversation_id"] for case in evaluation
    }
    historical_conversations = {
        case["conversation_id"] for case in historical
    }

    overlap = evaluation_conversations & historical_conversations

    if overlap:
        raise SystemExit(
            f"Conversation leakage detected: {len(overlap)} overlaps."
        )

    # Check historical source content ends before the temporal cutoff.
    cutoff = preparation_audit["historical_dev_cutoff"]

    if any(case["component_end"] >= cutoff for case in historical):
        raise SystemExit("Historical pool violates the temporal cutoff.")

    # Deterministically choose one historical case per conversation.
    candidates = sorted(historical, key=lambda case: case["example_id"])
    random.Random(42).shuffle(candidates)

    selected = []
    seen_conversations = set()

    for case in candidates:
        conversation_id = case["conversation_id"]

        if conversation_id in seen_conversations:
            continue

        if not case["historical_responses"]:
            continue

        if not case["customer_message"].strip():
            continue

        seen_conversations.add(conversation_id)
        selected.append(case)

        if len(selected) >= args.max_cases:
            break

    print(f"Building TF-IDF index from {len(selected):,} cases...")

    retriever = HistoricalRetriever().fit(selected)

    output = ROOT / "artifacts/retrieval"
    output.mkdir(parents=True, exist_ok=True)

    # Local generated artifact only. Never load joblib files from
    # untrusted sources: pickle-based files can execute code.
    joblib.dump(retriever, output / "index.joblib", compress=3)

    selected_ids = [case["example_id"] for case in selected]

    (output / "selected_evidence_ids.json").write_text(
        json.dumps(selected_ids, indent=2) + "\n",
        encoding="utf-8",
    )

    audit = {
        "seed": 42,
        "historical_pool_cases": len(historical),
        "indexed_cases": len(selected),
        "indexed_conversations": len(seen_conversations),
        "vocabulary_size": len(retriever.vectorizer.vocabulary_),
        "evaluation_conversation_overlap": len(overlap),
        "evaluation_inputs_sha256": digest,
        "indexed_text": "Historical customer message only.",
        "selection": (
            "Seeded shuffle, one case per historical conversation, "
            "up to the configured maximum."
        ),
        "limitations": [
            "Lexical similarity is not calibrated confidence.",
            "Historical replies are not verified successful resolutions.",
            "Acknowledgement and handoff replies remain in the index.",
            "Evidence applicability requires separate assessment.",
            "No public-data redistribution or remote-upload approval "
            "is implied by creating this local index.",
        ],
    }

    audit_path = ROOT / "artifacts/retrieval_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Index saved: {output / 'index.joblib'}")
    print(f"Audit saved: {audit_path}")


if __name__ == "__main__":
    main()