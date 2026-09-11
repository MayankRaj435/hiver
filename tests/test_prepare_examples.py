import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "prepare_examples",
    ROOT / "scripts/02_prepare_examples.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def tweet(
    tweet_id,
    author,
    inbound,
    minute,
    text,
    parent=None,
    responses="",
):
    return {
        "tweet_id": tweet_id,
        "author_id": author,
        "inbound": inbound,
        "created_at": f"2017-01-01T10:{minute:02d}:00+00:00",
        "text": text,
        "in_response_to_tweet_id": parent,
        "response_tweet_id": responses,
    }


def test_only_parent_ancestry_enters_input():
    records = [
        tweet("1", "customer_a", True, 0, "Music stopped.", responses="2"),
        tweet(
            "2", "SpotifyCares", False, 1,
            "Which device?", parent="1", responses="3,5",
        ),
        tweet(
            "3", "customer_a", True, 3,
            "My laptop.", parent="2", responses="4",
        ),
        tweet(
            "4", "SpotifyCares", False, 4,
            "FUTURE_RESPONSE_SENTINEL", parent="3",
        ),
        # Earlier in time but a sibling, not an ancestor of tweet 3.
        tweet(
            "5", "customer_a", True, 2,
            "SIBLING_SENTINEL", parent="2",
        ),
    ]

    cases, _ = module.build_cases(records)
    case = next(c for c in cases if c["customer_tweet_id"] == "3")

    assert [t["tweet_id"] for t in case["prior_context"]] == ["1", "2"]
    assert case["historical_responses"][0]["tweet_id"] == "4"

    exposed = module.model_input(case)

    assert "historical_responses" not in exposed
    assert "FUTURE_RESPONSE_SENTINEL" not in str(exposed)
    assert "SIBLING_SENTINEL" not in str(exposed)


def test_missing_ancestry_is_excluded():
    records = [
        tweet(
            "1", "customer_a", True, 1,
            "@SpotifyCares Please help.", parent="missing",
        ),
    ]

    cases, audit = module.build_cases(records)

    assert cases == []
    assert audit["excluded_incoming_turns"]["missing_ancestor"] == 1


def test_future_dated_parent_is_excluded():
    records = [
        tweet("1", "SpotifyCares", False, 5, "A brand message."),
        tweet(
            "2", "customer_a", True, 1,
            "A customer reply.", parent="1",
        ),
    ]

    cases, audit = module.build_cases(records)

    assert cases == []
    assert audit["excluded_incoming_turns"]["parent_after_child"] == 1