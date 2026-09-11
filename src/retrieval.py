"""Small, deterministic TF-IDF retriever over historical customer messages.

Similarity measures lexical overlap, not confidence, safety, or correctness.
Historical replies are evidence candidates, not verified resolutions.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_query(text):
    text = text.casefold()
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return " ".join(text.split())


class HistoricalRetriever:
    def __init__(self):
        self.records = []
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize_query,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
            max_features=30_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.matrix = None

    def fit(self, records):
        if not records:
            raise ValueError("Cannot build retrieval from an empty corpus.")

        self.records = list(records)

        # Index customer wording only. Do not index the historical reply
        # as if its answer text were part of the incoming customer query.
        self.matrix = self.vectorizer.fit_transform(
            record["customer_message"]
            for record in self.records
        )

        return self

    def search(self, customer_message, k=3):
        if self.matrix is None:
            raise RuntimeError("Call fit() before search().")

        if k <= 0:
            return []

        query = self.vectorizer.transform([customer_message])

        # TF-IDF vectors are L2 normalized, so this is cosine similarity.
        scores = (self.matrix @ query.T).toarray().ravel()

        ranked = np.argsort(-scores, kind="stable")[:k]
        results = []

        for index in ranked:
            score = float(scores[index])

            # No shared vocabulary means no useful lexical match.
            if score <= 0:
                continue

            record = self.records[int(index)]

            results.append(
                {
                    "evidence_id": record["example_id"],
                    "conversation_id": record["conversation_id"],
                    "similarity": score,
                    "customer_message": record["customer_message"],
                    "prior_context": record["prior_context"],
                    "historical_responses": record["historical_responses"],
                }
            )

        return results