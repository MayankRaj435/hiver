"""Minimal Gemini connectivity check.

Uses only synthetic text. No customer-support data is uploaded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    if not api_key:
        print("ERROR: GEMINI_API_KEY is missing from .env")
        return 1

    if not model:
        print("ERROR: GEMINI_MODEL is missing from .env")
        return 1

    print(f"Testing model: {model}")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly CONNECTION_OK",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=32,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        text = (response.text or "").strip()

        print(f"Response: {text!r}")

        if response.usage_metadata:
            usage = response.usage_metadata.model_dump(exclude_none=True)

            print(
                "Token usage:",
                {
                    "prompt_token_count": usage.get("prompt_token_count"),
                    "candidates_token_count": usage.get(
                        "candidates_token_count"
                    ),
                    "total_token_count": usage.get("total_token_count"),
                },
            )

        if "CONNECTION_OK" not in text:
            print("ERROR: Model responded, but not as expected.")
            return 2

        print("API connectivity: OK")
        return 0

    except Exception as exc:
        print(f"ERROR TYPE: {type(exc).__name__}")

        status = getattr(exc, "code", None)

        if status is not None:
            print(f"STATUS: {status}")

        # This can contain useful model-not-found diagnostics.
        # Never print the API key itself.
        message = str(exc)

        if api_key in message:
            message = message.replace(api_key, "[REDACTED]")

        print(f"MESSAGE: {message}")

        return 1

    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())