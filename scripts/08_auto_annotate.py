"""Automated data annotation script using Gemini."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/processed/prepared/annotation.csv"

# Load dot env to get API keys
load_dotenv(ROOT / ".env")

class AnnotationDraft(BaseModel):
    # model_config = ConfigDict(extra="forbid")
    
    primary_intent: str = Field(description="Must be one of: account_access, billing_subscription, playback_technical, library_content, product_how_to, other_unclear")
    required_action: str = Field(description="Must be one of: AUTO_HANDLE, ESCALATE")
    risk_flags: str = Field(description="Semicolon-separated list of flags: account_specific_action, financial_dispute, possible_account_compromise, sensitive_information, multi_issue, insufficient_context, requires_live_information, language_not_understood. Or 'none' if empty.")
    acceptable_response_actions: str = Field(description="What a useful, permissible reply should do")
    forbidden_claims: str = Field(description="What the agent must not claim (or 'none_specific')")
    label_rationale: str = Field(description="Why this label was chosen")
    annotation_confidence: str = Field(description="Must be one of: high, medium, low")


SYSTEM_PROMPT = """
You are an expert customer support data annotator for Spotify.
Your job is to read customer messages and assign the ground-truth labels for a support agent evaluating them.

INTENTS:
account_access: login, credentials, access, suspected compromise.
billing_subscription: charges, payments, plans, cancellation, refunds.
playback_technical: playback errors, crashes, connectivity, malfunction.
library_content: playlists, saved music, tracks, content availability.
product_how_to: other general feature-use questions.
other_unclear: ambiguous, unsupported, non-actionable, or unassignable.

ROUTING ACTIONS (AUTO_HANDLE vs ESCALATE):
- Escalate account-specific actions, financial disputes, suspected compromise, and requests requiring live verification.
- Escalate if language_not_understood.
- Escalate if you're not sure it's safe to auto-handle.
- AUTO_HANDLE for simple playback questions, library questions, product how-to, or basic clarifications.

RISK FLAGS:
account_specific_action, financial_dispute, possible_account_compromise, sensitive_information, multi_issue, insufficient_context, requires_live_information, language_not_understood.
If none apply, output 'none'. If multiple apply, separate with semicolons.

Please provide concise strings for acceptable_response_actions, forbidden_claims, and label_rationale.
"""

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
    n = len(list(d.glob("auto_*.csv")))
    dst = d / f"auto_{n:03d}.csv"
    shutil.copy2(CSV_PATH, dst)
    return dst

LABEL_FIELDS = [
    "primary_intent", "required_action", "risk_flags",
    "acceptable_response_actions", "forbidden_claims",
    "label_rationale", "annotation_confidence",
]

def done(row):
    return all(row.get(k, "").strip() for k in LABEL_FIELDS)

def annotate_example(client, model, row):
    ctx = (row.get("prior_context") or "").strip()
    msg = (row.get("customer_message") or "").strip()
    
    prompt = f"PRIOR CONTEXT:\n{ctx}\n\nCUSTOMER MESSAGE:\n{msg}\n\nPlease label this."
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=AnnotationDraft,
        ),
    )
    
    raw_text = response.text or ""
    try:
        draft = json.loads(raw_text)
        return draft
    except json.JSONDecodeError:
        print("Failed to decode JSON:", raw_text)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="representative_test", choices=["dev", "representative_test", "stress_test", "all"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=5)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not api_key or not model:
        raise SystemExit("Set GEMINI_API_KEY and GEMINI_MODEL in .env.")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
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

    print(f"Found {len(targets)} examples to annotate in split '{args.split}'.")
    
    session_done = 0
    try:
        for i, row in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] Annotating {row['example_id']}...")
            labels = annotate_example(client, model, row)
            
            if labels:
                for k in LABEL_FIELDS:
                    row[k] = labels.get(k, "")
                save(rows, fields)
                session_done += 1
                print(f"  -> Saved {row['primary_intent']} / {row['required_action']}")
            
            if i < len(targets):
                time.sleep(args.delay_seconds)
                
    except KeyboardInterrupt:
        print("\nStopping early.")
    finally:
        backup()
        print(f"Successfully auto-annotated {session_done} examples.")

if __name__ == "__main__":
    main()
