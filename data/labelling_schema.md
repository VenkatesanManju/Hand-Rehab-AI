# Diary labelling schema
Columns (CSV/JSONL): `text, pain, stiffness, mobility, grip, source`
- Severity 0 = none, 1 = mild, 2 = moderate, 3 = severe. For mobility/grip, 3 = severe LIMITATION.
- `source` = `synthetic_llm` or `hand_written`.
- Must include: negation ("no pain"), mixed symptoms ("no pain but very stiff in the morning"),
  time words (morning/evening), informal phrasing, typos, and some entries with missing symptoms (0).
- Target: 1,500-3,000 entries, roughly balanced across severities. Hold out a hand-written test set.
- Have a physiotherapist review a sample.
