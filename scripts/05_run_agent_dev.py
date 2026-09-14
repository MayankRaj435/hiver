"""Run a small development-only agent smoke test.

Never reads human labels or runs test examples.
Requires explicit permission before sending redacted text to Gemini.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import (
    SYSTEM_PROMPT,
    apply_gate,
    build_payload,
    generate_draft,
)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay-seconds", type=float, default=10)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--split", default="dev", choices=["dev", "representative_test", "stress_test", "all"])
    args = parser.parse_args()

    if not args.allow_remote:
        raise SystemExit(
            "Remote upload is disabled.  Review dataset/provider terms "
            "and the redaction limitations, then pass --allow-remote "
            "if you choose to proceed."
        )

    if args.limit < 0:
        raise SystemExit("--limit cannot be negative.")

    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds cannot be negative.")

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    if not api_key or not model:
        raise SystemExit("Set GEMINI_API_KEY and GEMINI_MODEL in .env.")

    prepared = ROOT / "data/processed/prepared"
    inputs_path = prepared / "evaluation_inputs.jsonl"

    audit = json.loads(
        (ROOT / "artifacts/preparation_audit.json").read_text(
            encoding="utf-8"
        )
    )

    if sha256(inputs_path) != audit["evaluation_inputs_sha256"]:
        raise SystemExit("Frozen evaluation input hash changed.")

    manifest = read_jsonl(prepared / "sampling_manifest.jsonl")
    inputs = {
        item["example_id"]: item for item in read_jsonl(inputs_path)
    }

    run_ids = [
        item["example_id"]
        for item in manifest
        if args.split == "all" or item["split"] == args.split
    ]
    if args.limit > 0:
        run_ids = run_ids[: args.limit]

    index_path = ROOT / "artifacts/retrieval/index.joblib"
    retriever = joblib.load(index_path)

    # Verify the loaded index does not overlap evaluation conversations.
    evaluation_conversations = {
        item["conversation_id"] for item in inputs.values()
    }
    evidence_conversations = {
        item["conversation_id"] for item in retriever.records
    }
    if evaluation_conversations & evidence_conversations:
        raise SystemExit(
            "Loaded index overlaps evaluation conversations."
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = ROOT / "artifacts/agent_dev" / run_id
    output_dir.mkdir(parents=True)

    metadata = {
        "run_id": run_id,
        "model": model,
        "split": args.split,
        "requested_examples": len(run_ids),
        "temperature": 0,
        "retrieval_k": 3,
        "evaluation_inputs_sha256": sha256(inputs_path),
        "retrieval_index_sha256": sha256(index_path),
        "agent_source_sha256": sha256(ROOT / "src/agent.py"),
        "prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "note": "Development smoke test, not headline test results.",
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "system_prompt.txt").write_text(
        SYSTEM_PROMPT, encoding="utf-8",
    )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )

    try:
        with (output_dir / "predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for number, example_id in enumerate(run_ids, start=1):
                example = inputs[example_id]
                evidence = retriever.search(
                    example["customer_message"], k=3
                )
                payload = build_payload(example, evidence)

                started = time.perf_counter()
                try:
                    draft, raw_text, usage = generate_draft(
                        client,
                        model,
                        json.dumps(payload, ensure_ascii=False),
                    )
                    result = {
                        "example_id": example_id,
                        "split": args.split,
                        "status": "ok",
                        "model": model,
                        **apply_gate(draft, evidence),
                        "raw_model_output": raw_text,
                        "usage": usage,
                        "request_payload": payload,
                        "retrieval_scores": [
                            {
                                "evidence_id": item["evidence_id"],
                                "similarity": item["similarity"],
                            }
                            for item in evidence
                        ],
                        "latency_seconds": round(
                            time.perf_counter() - started, 3
                        ),
                    }
                except Exception as exc:
                    result = {
                        "example_id": example_id,
                        "split": args.split,
                        "status": "error",
                        "decision": "ESCALATE",
                        "send_eligible": False,
                        "error_type": type(exc).__name__,
                        "error_code": str(getattr(exc, "code", "")),
                        "latency_seconds": round(
                            time.perf_counter() - started, 3
                        ),
                    }
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
                    print(
                        f"Stopped: {result['error_type']} "
                        f"code={result['error_code']}"
                    )
                    print(f"Artifacts: {output_dir}")
                    return 1

                handle.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                handle.flush()

                print(
                    f"[{number}/{len(run_ids)}] "
                    f"{example_id}: {result['intent']} | "
                    f"{result['decision']}"
                )

                if number < len(run_ids):
                    time.sleep(args.delay_seconds)
    finally:
        client.close()

    print(f"\nSaved development artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
