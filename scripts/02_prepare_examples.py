"""Build prior-context inputs and sample a manual annotation set.

Source records may contain future responses.
Exported annotation/model inputs do not contain those responses.

Sampling unit: one eligible incoming turn per connected conversation.
Representative results therefore describe this sampling population,
not the raw distribution of all tweet turns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import timezone
from pathlib import Path

from dateutil.parser import parse as parse_datetime


ROOT = Path(__file__).resolve().parents[1]
BRAND = "SpotifyCares"
SEED = 42


def timestamp(value):
    parsed = parse_datetime(value)

    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value!r}")

    return parsed.astimezone(timezone.utc).isoformat()


def response_ids(value):
    return [
        item.strip()
        for item in (value or "").split(",")
        if item.strip()
    ]


def normalize_message(text):
    """Conservative normalized-exact duplicate key, not semantic dedup."""
    text = text.casefold()
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+", "<url>", text)
    return " ".join(text.split())


def model_input(case):
    """Explicit allowlist: future response fields cannot pass through."""
    return {
        "example_id": case["example_id"],
        "conversation_id": case["conversation_id"],
        "customer_tweet_id": case["customer_tweet_id"],
        "timestamp": case["timestamp"],
        "customer_message": case["customer_message"],
        "prior_context": case["prior_context"],
    }


def build_cases(records):
    tweets = {}

    for record in records:
        item = dict(record)
        tweet_id = item["tweet_id"]

        if tweet_id in tweets:
            raise ValueError(f"Duplicate tweet ID: {tweet_id}")

        item["created_at"] = timestamp(item["created_at"])
        tweets[tweet_id] = item

    # Union-find includes absent referenced IDs. Two observed branches
    # referencing the same missing parent still stay in one component.
    parents = {}

    def find(node):
        parents.setdefault(node, node)

        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]

        return node

    def union(left, right):
        a, b = find(left), find(right)

        if a != b:
            small, large = sorted((a, b))
            parents[large] = small

    for tweet_id, tweet in tweets.items():
        find(tweet_id)

        neighbors = response_ids(tweet.get("response_tweet_id"))
        parent_id = tweet.get("in_response_to_tweet_id")

        if parent_id:
            neighbors.append(parent_id)

        for neighbor in neighbors:
            union(tweet_id, neighbor)

    grouped = defaultdict(list)

    for tweet_id in tweets:
        grouped[find(tweet_id)].append(tweet_id)

    component_meta = {}

    for root, ids in grouped.items():
        times = [tweets[tid]["created_at"] for tid in ids]

        foreign_brand = any(
            not tweets[tid]["inbound"]
            and tweets[tid]["author_id"] != BRAND
            for tid in ids
        )

        component_meta[root] = {
            "start": min(times),
            "end": max(times),
            "foreign_brand": foreign_brand,
        }

    # Only explicit parent links establish response ancestry.
    children = defaultdict(list)

    for tweet in tweets.values():
        if tweet.get("in_response_to_tweet_id"):
            children[tweet["in_response_to_tweet_id"]].append(tweet)

    cases = []
    excluded = Counter()

    for tweet_id in sorted(tweets):
        tweet = tweets[tweet_id]

        if not tweet["inbound"]:
            continue

        root = find(tweet_id)
        meta = component_meta[root]

        if meta["foreign_brand"]:
            excluded["multi_brand_component"] += 1
            continue

        if not tweet["text"].strip():
            excluded["empty_message"] += 1
            continue

        parent = tweets.get(tweet.get("in_response_to_tweet_id"))
        direct_brand_replies = [
            child
            for child in children[tweet_id]
            if child["author_id"] == BRAND and not child["inbound"]
        ]

        # Require a clear connection between this incoming turn and Spotify.
        directly_addressed = (
            (parent is not None and parent["author_id"] == BRAND)
            or bool(direct_brand_replies)
            or "@spotifycares" in tweet["text"].casefold()
        )

        if not directly_addressed:
            excluded["not_clearly_addressed_to_brand"] += 1
            continue

        # Follow parents only. Never use arbitrary chronological neighbors.
        ancestors = []
        seen = {tweet_id}
        parent_id = tweet.get("in_response_to_tweet_id")
        child_time = tweet["created_at"]
        invalid_reason = None

        while parent_id:
            if parent_id in seen:
                invalid_reason = "ancestry_cycle"
                break

            seen.add(parent_id)
            ancestor = tweets.get(parent_id)

            if ancestor is None:
                invalid_reason = "missing_ancestor"
                break

            if ancestor["created_at"] > child_time:
                invalid_reason = "parent_after_child"
                break

            ancestors.append(ancestor)
            child_time = ancestor["created_at"]
            parent_id = ancestor.get("in_response_to_tweet_id")

            if len(ancestors) > 100:
                invalid_reason = "ancestry_over_100_turns"
                break

        if invalid_reason:
            excluded[invalid_reason] += 1
            continue

        ancestors.reverse()

        prior_context = [
            {
                "tweet_id": ancestor["tweet_id"],
                "role": "customer" if ancestor["inbound"] else "brand",
                "text": ancestor["text"],
            }
            for ancestor in ancestors
        ]

        # These are historical responses, NOT proof of successful resolution.
        historical_responses = [
            {
                "tweet_id": reply["tweet_id"],
                "text": reply["text"],
                "timestamp": reply["created_at"],
            }
            for reply in sorted(
                direct_brand_replies,
                key=lambda item: (item["created_at"], item["tweet_id"]),
            )
            if reply["created_at"] >= tweet["created_at"]
        ]

        cases.append(
            {
                "example_id": f"spotify_{tweet_id}",
                "conversation_id": f"thread_{root}",
                "customer_tweet_id": tweet_id,
                "timestamp": tweet["created_at"],
                "customer_message": tweet["text"],
                "prior_context": prior_context,
                "historical_responses": historical_responses,
                "component_start": meta["start"],
                "component_end": meta["end"],
            }
        )

    return cases, {
        "connected_components": len(grouped),
        "eligible_cases_before_splitting": len(cases),
        "excluded_incoming_turns": dict(excluded),
    }


def split_cases(cases):
    """Earlier retrieval, middle development, later test.

    Components crossing temporal cutoffs are excluded.
    The percentages define time cutoffs, not guaranteed final set sizes.
    """
    components = {
        case["conversation_id"]: case["component_start"]
        for case in cases
    }

    starts = sorted(components.values())

    if len(starts) < 100:
        raise ValueError("Too few eligible conversations for this sampling plan.")

    first_cutoff = starts[int(len(starts) * 0.70)]
    second_cutoff = starts[int(len(starts) * 0.80)]

    if first_cutoff >= second_cutoff:
        raise ValueError("Temporal cutoffs are not distinct.")

    pools = {"historical": [], "dev": [], "test": []}
    crossing = 0

    for case in cases:
        start = case["component_start"]
        end = case["component_end"]

        if end < first_cutoff:
            pools["historical"].append(case)
        elif start >= first_cutoff and end < second_cutoff:
            pools["dev"].append(case)
        elif start >= second_cutoff:
            pools["test"].append(case)
        else:
            crossing += 1

    # Remove later cases repeating normalized customer messages from an
    # earlier partition. This is intentionally conservative.
    seen_earlier = set()
    duplicate_removals = {}

    for name in ["historical", "dev", "test"]:
        original = pools[name]

        kept = [
            case
            for case in original
            if normalize_message(case["customer_message"]) not in seen_earlier
        ]

        duplicate_removals[name] = len(original) - len(kept)
        pools[name] = kept

        seen_earlier.update(
            normalize_message(case["customer_message"])
            for case in original
        )

    return pools, {
        "historical_dev_cutoff": first_cutoff,
        "dev_test_cutoff": second_cutoff,
        "crossing_component_cases_excluded": crossing,
        "normalized_exact_duplicates_removed": duplicate_removals,
        "pool_sizes": {name: len(pool) for name, pool in pools.items()},
    }


def one_case_per_conversation(cases, rng):
    grouped = defaultdict(list)

    for case in cases:
        grouped[case["conversation_id"]].append(case)

    selected = []

    for conversation_id in sorted(grouped):
        options = sorted(
            grouped[conversation_id],
            key=lambda case: case["example_id"],
        )
        selected.append(rng.choice(options))

    return selected


def stress_tags(case):
    """Sampling heuristics only. These are NOT human labels."""
    text = case["customer_message"].casefold()
    tags = []

    if re.search(
        r"\b(refund|charged|billing|payment|hacked|stolen|"
        r"unauthorized|unauthorised|password)\b",
        text,
    ):
        tags.append("account_or_financial_keyword")

    if len(re.sub(r"@\w+", "", text).split()) <= 6:
        tags.append("short_message")

    if re.search(
        r"\b(still|again|already|nothing works|not working)\b",
        text,
    ):
        tags.append("possible_repeated_failure")

    if len(case["prior_context"]) >= 4:
        tags.append("longer_conversation")

    return tags


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_outputs(cases, build_audit, output, audit_path):
    pools, split_audit = split_cases(cases)
    rng = random.Random(SEED)

    dev_candidates = one_case_per_conversation(pools["dev"], rng)
    test_candidates = one_case_per_conversation(pools["test"], rng)

    if len(dev_candidates) < 30 or len(test_candidates) < 120:
        raise ValueError("Insufficient conversations for 150 annotations.")

    dev = rng.sample(dev_candidates, 30)

    # Sample representative cases FIRST so stress selection cannot bias them.
    representative = rng.sample(test_candidates, 90)
    representative_ids = {
        case["conversation_id"] for case in representative
    }

    remaining = [
        case
        for case in test_candidates
        if case["conversation_id"] not in representative_ids
    ]

    stress_candidates = [case for case in remaining if stress_tags(case)]

    if len(stress_candidates) < 30:
        raise ValueError("Fewer than 30 heuristic-matched stress candidates.")

    stress = rng.sample(stress_candidates, 30)

    assignments = (
        [(case, "dev") for case in dev]
        + [(case, "representative_test") for case in representative]
        + [(case, "stress_test") for case in stress]
    )

    output.mkdir(parents=True, exist_ok=True)

    # Never overwrite human work by silently recreating the annotation sheet.
    annotation_path = output / "annotation.csv"

    if annotation_path.exists():
        raise FileExistsError(
            f"{annotation_path} already exists. "
            "Back up annotations before creating a new sample."
        )

    inputs = [model_input(case) for case, _ in assignments]
    write_jsonl(output / "evaluation_inputs.jsonl", inputs)

    historical = [
        case for case in pools["historical"]
        if case["historical_responses"]
    ]

    write_jsonl(
        output / "historical_evidence_pool.jsonl",
        historical,
    )

    fields = [
        "example_id",
        "split",
        "conversation_id",
        "customer_message",
        "prior_context",
        "primary_intent",
        "required_action",
        "risk_flags",
        "acceptable_response_actions",
        "forbidden_claims",
        "label_rationale",
        "annotation_confidence",
    ]

    with annotation_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for case, split in assignments:
            context = "\n".join(
                f"{turn['role'].upper()}: {turn['text']}"
                for turn in case["prior_context"]
            )

            writer.writerow(
                {
                    "example_id": case["example_id"],
                    "split": split,
                    "conversation_id": case["conversation_id"],
                    "customer_message": case["customer_message"],
                    "prior_context": context,
                }
            )

    manifest = [
        {
            "example_id": case["example_id"],
            "conversation_id": case["conversation_id"],
            "split": split,
            "sampling_tags": (
                stress_tags(case) if split == "stress_test" else []
            ),
        }
        for case, split in assignments
    ]

    write_jsonl(output / "sampling_manifest.jsonl", manifest)

    # Assert partition-level component separation.
    component_sets = {
        name: {case["conversation_id"] for case in pool}
        for name, pool in pools.items()
    }

    assert component_sets["historical"].isdisjoint(component_sets["dev"])
    assert component_sets["historical"].isdisjoint(component_sets["test"])
    assert component_sets["dev"].isdisjoint(component_sets["test"])

    selected_ids = [case["example_id"] for case, _ in assignments]
    assert len(selected_ids) == len(set(selected_ids)) == 150

    input_hash = hashlib.sha256(
        (output / "evaluation_inputs.jsonl").read_bytes()
    ).hexdigest()

    audit = {
        **build_audit,
        **split_audit,
        "sampling_seed": SEED,
        "annotation_counts": {
            "dev": 30,
            "representative_test": 90,
            "stress_test": 30,
        },
        "historical_cases_with_brand_response": len(historical),
        "evaluation_inputs_sha256": input_hash,
        "sampling_unit": (
            "One randomly selected eligible turn per conversation, "
            "followed by conversation-level sampling."
        ),
        "stress_sampling": (
            "Random sample from remaining cases matching documented "
            "lexical, short-message, or context-length heuristics."
        ),
        "limitations": [
            "No semantic near-duplicate detection has been performed.",
            "Customer-author overlap has not been measured.",
            "Language is not automatically filtered.",
            "Incomplete ancestry and multi-brand components are excluded.",
            "Historical replies are not verified successful resolutions.",
            "Stress heuristics are not human risk or intent labels.",
            "Representative sampling is not raw tweet-turn traffic sampling.",
        ],
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nPreparation complete")
    print(f"Eligible cases: {len(cases):,}")
    print(f"Historical evidence cases: {len(historical):,}")
    print(f"Manual annotation sheet: {annotation_path}")
    print(f"Audit: {audit_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/processed/brand_connected_tweets.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/prepared",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts/preparation_audit.json",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Missing extraction: {args.input}")

    if (args.output_dir / "annotation.csv").exists():
        raise SystemExit(
            "Annotation sheet already exists. Refusing to overwrite it."
        )

    with args.input.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    cases, audit = build_cases(records)
    prepare_outputs(cases, audit, args.output_dir, args.audit)


if __name__ == "__main__":
    main()