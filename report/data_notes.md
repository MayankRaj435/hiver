# Data notes

## Source
Customer Support on Twitter, thoughtvector:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

## Observed source statistics
- CSV size: 516,508,641 bytes.
- Total tweet rows: 2,811,774.
- SpotifyCares-authored rows: 43,265.
- Structural audit tests: 2 passed.

## Interpretation
Brand-authored tweet counts are not counts of customer requests,
independent conversations, or successful resolutions.

## Planned safeguards
- Group connected conversations before splitting.
- Construct customer inputs from reply ancestry, not arbitrary
  chronological neighbors.
- Exclude future brand responses from model inputs.
- Keep evaluation conversations out of historical retrieval.
- Keep manually inspected exploration examples out of the final test.