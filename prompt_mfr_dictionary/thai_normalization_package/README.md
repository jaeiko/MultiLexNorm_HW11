# Thai Normalization Package

This package contains a Thai-specific lexical normalization module for MultiLexNorm-style data.

## Files

```text
thai_normalization_package/
  language_rules/
    th.py
  mfr_dictionaries/
    th_mfr_dictionary.json
    th_mfr.py
  th_summary.csv
  README.md
```

## Strategy

Thai is written without spaces between words in standard orthography and Thai social media misspellings can encode emotion, politeness, stance, or identity. Therefore, this package uses a conservative hybrid pipeline:

```text
Thai input tokens
→ exact high-confidence MFR lookup
→ Thai-specific candidate detection
→ context-aware target-token prompt/model fallback
→ uncertain cases preserve original token
```

## Validation summary

```text
Train tokens: 140423
Train changed tokens: 5560
Train changed ratio: 0.039595
Validation tokens: 19872
Validation changed tokens: 786
Validation changed ratio: 0.039553
High-confidence MFR pairs: 214
Validation ERR with high-confidence MFR only: 0.372774
TP / FP / FN: 378 / 85 / 408
```

## Usage

```python
from mfr_dictionaries.th_mfr import load_th_mfr_dictionary, apply_th_mfr_to_tokens
from language_rules.th import candidate_indices, build_th_target_prompt

th_dict = load_th_mfr_dictionary("mfr_dictionaries/th_mfr_dictionary.json")
tokens = ["เค้า", "ชอบ", "จริงๆ", "มากกก", "555"]
first_pass = apply_th_mfr_to_tokens(tokens, th_dict)
# ["เขา", "ชอบ", "จริง ๆ", "มาก", "555"]

cands = candidate_indices(tokens)
prompt = build_th_target_prompt(tokens, cands[0])
```

## Important notes

- Apply only `high_confidence_pairs` automatically.
- Do not automatically rewrite `review_pairs` or `ambiguous_pairs`.
- Preserve laughter such as `555` by default.
- Repeated characters are candidates, not always direct replacements, because Thai misspellings can carry sentiment and pragmatic information.
- Politeness particles such as `คะ`, `ค่ะ`, `ครับ`, `คับ`, `จ้า`, `น้า`, and `นะ` are context-sensitive.
