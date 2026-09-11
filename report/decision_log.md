- Sample one eligible incoming turn per conversation rather than treating
  every tweet turn as independent. Representative metrics therefore
  describe this conversation-based sampling population.

- Exclude incomplete reply ancestry and multi-brand components from the
  initial evaluation scope, and disclose the resulting coverage limitation.

- Remove normalized exact customer-message duplicates from later time
  partitions; do not claim semantic near-duplicate removal.

- Start with at most 5,000 historical evidence cases, one per conversation,
  for fast local TF-IDF retrieval.

- Index historical customer wording only. Historical replies are returned
  as evidence candidates, not indexed as customer query text.

- Treat lexical retrieval similarity as a ranking score, not as a probability
  that an answer is correct or safe.