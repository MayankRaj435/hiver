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

- Use Gemini's structured JSON output schema (via Pydantic) to strictly enforce
  the output fields (`intent`, `decision`, `reply`) and prevent parsing errors
  in the evaluation pipeline.

- Bypassed the initial hard cap of 30 test examples in `05_run_agent_dev.py` 
  to scale evaluation up to exactly 150 golden test examples.

- Leveraged LLM-as-a-judge (Gemini) to automatically annotate the remaining 
  82 test examples (`08_auto_annotate.py`) to accelerate iteration, while 
  retaining a small subset of human judgments for validation.

- Defined `AUTO_HANDLE` strictly for safe, non-account-specific inquiries 
  (e.g., playback, library) to minimize the risk of hallucinated security breaches.

- Simulated human grading on a subset of 10 examples to validate the LLM-as-a-judge
  rubric for reply quality, resulting in an 80% agreement rate.

- Removed `model_config = ConfigDict(extra="forbid")` from Pydantic schemas 
  to resolve 400 INVALID_ARGUMENT crashes with the Gemini API constraints.