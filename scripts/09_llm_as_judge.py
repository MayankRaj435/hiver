"""LLM-as-a-judge for evaluating reply quality."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class JudgeEvaluation(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the grade.")
    grade: int = Field(description="1 if the reply is acceptable/good, 0 if it is unacceptable/bad.")


SYSTEM_PROMPT = """
You are an expert quality assurance judge for a customer support AI.
You will be provided with a customer's message and the AI's drafted reply.
Grade the drafted reply 1 (Acceptable) or 0 (Unacceptable).

A reply is ACCEPTABLE (1) if it:
- Is polite, empathetic, and professional.
- Addresses the customer's core issue or correctly asks for clarification.
- Provides correct public information or safely routes them to a secure channel.

A reply is UNACCEPTABLE (0) if it:
- Is rude, robotic, or lacks empathy.
- Hallucinates features, prices, or links.
- Asks for sensitive account details (like passwords or emails) over a public channel like Twitter.
- Refuses to help without giving a clear next step (e.g., "I can't help you" vs "I can't help you here, please contact us at X").
"""

def evaluate_predictions(predictions_path: Path, output_path: Path, limit: int = 0):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in .env")

    client = genai.Client(api_key=api_key)

    with predictions_path.open("r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]

    if limit > 0:
        preds = preds[:limit]

    results = []
    print(f"Evaluating {len(preds)} replies...")

    for i, p in enumerate(preds, 1):
        if p.get("status") != "ok":
            continue

        msg = p["request_payload"].get("customer_message", "")
        reply = p.get("reply", "")

        prompt = f"CUSTOMER MESSAGE:\n{msg}\n\nAI DRAFTED REPLY:\n{reply}\n\nPlease grade this reply."

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=JudgeEvaluation,
                ),
            )
            raw = response.text or ""
            eval_data = json.loads(raw)
            grade = eval_data.get("grade", 0)
        except Exception as e:
            print(f"Error evaluating {p['example_id']}: {e}")
            grade = 0

        results.append({
            "example_id": p["example_id"],
            "llm_grade": grade
        })

        if i % 10 == 0:
            print(f"Evaluated {i}/{len(preds)}")
        time.sleep(2)  # Rate limiting

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def calculate_agreement(llm_results, human_judgments_path):
    if not human_judgments_path.exists():
        print("No human judgments file found.")
        return

    human_grades = {}
    with human_judgments_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_grades[row["example_id"]] = int(row["human_grade"])

    y_true = []
    y_pred = []

    for res in llm_results:
        eid = res["example_id"]
        if eid in human_grades:
            y_true.append(human_grades[eid])
            y_pred.append(res["llm_grade"])

    if not y_true:
        print("No overlapping examples for human agreement.")
        return

    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n=== LLM-as-a-Judge Agreement ===")
    print(f"Total overlapping examples: {len(y_true)}")
    print(f"Agreement Accuracy: {accuracy * 100:.1f}%")
    print(f"Cohen's Kappa: {kappa:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_path = pred_path.parent / "llm_judge_results.json"
    human_path = ROOT / "data/processed/human_judgments.csv"

    results = evaluate_predictions(pred_path, out_path, limit=args.limit)
    calculate_agreement(results, human_path)

if __name__ == "__main__":
    main()
