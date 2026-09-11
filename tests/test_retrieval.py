import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval import HistoricalRetriever


def record(example_id, message, reply):
    return {
        "example_id": example_id,
        "conversation_id": f"thread_{example_id}",
        "customer_message": message,
        "prior_context": [],
        "historical_responses": [
            {"tweet_id": f"reply_{example_id}", "text": reply}
        ],
    }


def test_retrieves_matching_customer_issue():
    retriever = HistoricalRetriever().fit(
        [
            record("a", "My playlist disappeared", "Please explain further."),
            record("b", "I was charged twice", "An account review is needed."),
        ]
    )

    results = retriever.search("charged twice", k=1)

    assert results[0]["evidence_id"] == "b"
    assert results[0]["similarity"] > 0


def test_reply_text_is_not_indexed():
    retriever = HistoricalRetriever().fit(
        [
            record("a", "My playlist disappeared", "secretanswerword"),
        ]
    )

    assert retriever.search("secretanswerword") == []


def test_unknown_vocabulary_returns_no_evidence():
    retriever = HistoricalRetriever().fit(
        [
            record("a", "My playlist disappeared", "Please explain further."),
        ]
    )

    assert retriever.search("zzzzunknownword") == []