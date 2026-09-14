"""Two baselines for the Spotify support-agent evaluation.

Neither baseline reads test labels.
Neither baseline calls a language model.

Baseline A (trivial):
    Majority development intent, generic acknowledgement, always escalate.

Baseline B (simple):
    TF-IDF intent classifier trained on development labels only,
    sanitized nearest historical reply, conservative routing rules.
"""

from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INTENTS = [
    "account_access",
    "billing_subscription",
    "playback_technical",
    "library_content",
    "product_how_to",
    "other_unclear",
]

GENERIC_ACK = (
    "Thanks for reaching out. We're looking into this and will follow up."
)

# Documented, conservative routing keywords for the simple baseline.
# This is a deliberately simple rule, not a safety guarantee.
HIGH_RISK_KEYWORDS = [
    "refund",
    "charged",
    "charge",
    "billing",
    "payment",
    "invoice",
    "hacked",
    "stolen",
    "unauthorized",
    "unauthorised",
    "password",
    "compromise",
    "cancel my",
    "delete my",
    "close my",
    "suspend",
    "banned",
    "lawyer",
    "legal",
    "court",
    "sue",
]

# Documented a priori.  NOT tuned on test results.
SIMILARITY_THRESHOLD = 0.10


def sanitize_reply(text):
    """Light sanitization of a reused historical reply."""
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return " ".join(text.split())


def build_trivial(dev_labels):
    """Build trivial baseline state from development labels."""
    majority = Counter(
        label["primary_intent"] for label in dev_labels
    ).most_common(1)[0][0]
    return {"system": "trivial", "majority_intent": majority}


def build_simple(dev_labels):
    """Build simple baseline: TF-IDF + LogisticRegression on dev only."""
    texts = [label["customer_message"] for label in dev_labels]
    labels = [label["primary_intent"] for label in dev_labels]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(vectorizer.fit_transform(texts), labels)

    return {"system": "simple", "vectorizer": vectorizer, "model": model}


def predict_trivial(state, example, retriever):
    """Trivial baseline: majority intent, always escalate."""
    return {
        "system": "trivial",
        "intent": state["majority_intent"],
        "reply": GENERIC_ACK,
        "decision": "ESCALATE",
        "send_eligible": False,
        "reason": "Trivial baseline always escalates.",
        "evidence_ids": [],
    }


def predict_simple(state, example, retriever):
    """Simple baseline: TF-IDF intent + nearest historical reply + keyword routing."""
    message = example["customer_message"]

    intent = state["model"].predict(
        state["vectorizer"].transform([message])
    )[0]

    evidence = retriever.search(message, k=1)
    if evidence:
        raw_reply = evidence[0]["historical_responses"][0]["text"]
        reply = sanitize_reply(raw_reply) or GENERIC_ACK
        evidence_ids = [evidence[0]["evidence_id"]]
        retrieval_score = evidence[0]["similarity"]
    else:
        reply = GENERIC_ACK
        evidence_ids = []
        retrieval_score = 0.0

    lowered = message.casefold()
    matched = [kw for kw in HIGH_RISK_KEYWORDS if kw in lowered]

    if matched or retrieval_score < SIMILARITY_THRESHOLD:
        decision = "ESCALATE"
    else:
        decision = "AUTO_HANDLE"

    return {
        "system": "simple",
        "intent": intent,
        "reply": reply,
        "decision": decision,
        "send_eligible": decision == "AUTO_HANDLE",
        "reason": f"Keyword routing; matched: {matched or 'none'}. "
                  f"Similarity: {retrieval_score:.3f}.",
        "evidence_ids": evidence_ids,
        "retrieval_score": retrieval_score,
    }


PREDICTORS = {
    "trivial": predict_trivial,
    "simple": predict_simple,
}
