"""Unit tests for agent gate logic and input construction.

No Gemini API calls are made.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import AgentDraft, apply_gate, build_payload, redact_text


def _draft(**changes):
    values = {
        "intent": "playback_technical",
        "reply": "Which device are you using?",
        "response_kind": "clarification",
        "decision": "AUTO_HANDLE",
        "reason": "A relevant clarification is safe.",
        "risk_flags": ["insufficient_context"],
        "evidence_ids": [],
        "evidence_sufficient": False,
    }
    values.update(changes)
    return AgentDraft(**values)


def test_safe_clarification_can_pass():
    result = apply_gate(_draft(), [])
    assert result["decision"] == "AUTO_HANDLE"


def test_account_action_is_escalated():
    result = apply_gate(
        _draft(risk_flags=["account_specific_action"]),
        [],
    )
    assert result["decision"] == "ESCALATE"
    assert "HIGH_RISK_REQUEST" in result["gate_reason_codes"]


def test_invalid_citation_is_escalated():
    result = apply_gate(
        _draft(evidence_ids=["invented_case"]),
        [],
    )
    assert "INVALID_EVIDENCE_ID" in result["gate_reason_codes"]


def test_guidance_without_evidence_is_escalated():
    result = apply_gate(
        _draft(response_kind="general_guidance"),
        [],
    )
    assert "INSUFFICIENT_EVIDENCE_FOR_GUIDANCE" in (
        result["gate_reason_codes"]
    )


def test_future_response_is_not_in_payload():
    example = {
        "customer_message": "Music stopped.",
        "prior_context": [],
        "historical_responses": [
            {"text": "FUTURE_RESPONSE_SENTINEL"}
        ],
    }
    payload = build_payload(example, [])
    assert "FUTURE_RESPONSE_SENTINEL" not in str(payload)


def test_common_identifiers_are_redacted():
    text = redact_text(
        "@customer Email me at person@example.com "
        "or visit https://example.com"
    )
    assert "@customer" not in text
    assert "person@example.com" not in text
    assert "https://example.com" not in text


def test_false_action_claim_is_caught():
    result = apply_gate(
        _draft(reply="We have checked your account and reset your password."),
        [],
    )
    assert "POSSIBLE_FALSE_ACTION_CLAIM" in result["gate_reason_codes"]
    assert result["decision"] == "ESCALATE"


def test_url_in_reply_is_caught():
    result = apply_gate(
        _draft(reply="Please visit https://spotify.com/help for more."),
        [],
    )
    assert "UNAPPROVED_URL" in result["gate_reason_codes"]
    assert result["decision"] == "ESCALATE"
