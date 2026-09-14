"""Evaluate agent or baseline predictions against ground truth labels."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def load_labels():
    path = ROOT / "data/processed/prepared/annotation.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    
    LABEL_FIELDS = [
        "primary_intent",
        "required_action",
        "risk_flags",
        "acceptable_response_actions",
        "forbidden_claims",
        "label_rationale",
        "annotation_confidence",
    ]
    complete = {
        row["example_id"]: row
        for row in rows
        if all(row.get(field, "").strip() for field in LABEL_FIELDS)
    }
    return complete

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions.jsonl")
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}")

    labels = load_labels()
    predictions = read_jsonl(args.predictions)

    total_evaluated = 0
    intent_correct = 0
    decision_correct = 0
    
    # Safe escalation: When required_action == ESCALATE, did we ESCALATE?
    must_escalate_total = 0
    escalated_correctly = 0
    
    # Auto-handle coverage: When required_action == AUTO_HANDLE, did we AUTO_HANDLE?
    can_auto_total = 0
    auto_handled_correctly = 0

    for pred in predictions:
        if pred.get("status") == "error":
            continue

        example_id = pred["example_id"]
        if example_id not in labels:
            continue
            
        label = labels[example_id]
        total_evaluated += 1

        # Intents
        pred_intent = pred.get("intent")
        if pred_intent == label["primary_intent"]:
            intent_correct += 1

        # Decisions
        pred_decision = "AUTO_HANDLE" if pred.get("send_eligible") else "ESCALATE"
        true_decision = label["required_action"]

        if pred_decision == true_decision:
            decision_correct += 1

        if true_decision == "ESCALATE":
            must_escalate_total += 1
            if pred_decision == "ESCALATE":
                escalated_correctly += 1
        elif true_decision == "AUTO_HANDLE":
            can_auto_total += 1
            if pred_decision == "AUTO_HANDLE":
                auto_handled_correctly += 1

    if total_evaluated == 0:
        print("No matching examples found for evaluation.")
        return

    intent_acc = intent_correct / total_evaluated
    decision_acc = decision_correct / total_evaluated
    safe_escalation_rate = escalated_correctly / must_escalate_total if must_escalate_total > 0 else 1.0
    auto_handle_rate = auto_handled_correctly / can_auto_total if can_auto_total > 0 else 0.0

    report = {
        "metrics": {
            "total_evaluated": total_evaluated,
            "intent_accuracy": round(intent_acc, 4),
            "decision_accuracy": round(decision_acc, 4),
            "safe_escalation_rate": round(safe_escalation_rate, 4),
            "auto_handle_coverage": round(auto_handle_rate, 4)
        },
        "counts": {
            "must_escalate_total": must_escalate_total,
            "escalated_correctly": escalated_correctly,
            "can_auto_total": can_auto_total,
            "auto_handled_correctly": auto_handled_correctly
        }
    }

    out_file = args.predictions.parent / "evaluation_report.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Evaluation Report ===")
    print(f"Total Evaluated: {total_evaluated}")
    print(f"Intent Accuracy: {intent_acc:.1%}")
    print(f"Handling Accuracy: {decision_acc:.1%}")
    print(f"Safe Escalation Rate (Recall on Escalation): {safe_escalation_rate:.1%} ({escalated_correctly}/{must_escalate_total})")
    print(f"Auto-Handle Coverage (Recall on Auto-Handle): {auto_handle_rate:.1%} ({auto_handled_correctly}/{can_auto_total})")
    print(f"\nSaved report to {out_file}")

if __name__ == "__main__":
    main()
