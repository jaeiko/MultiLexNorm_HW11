# Vietnamese Normalization Package

This package contains a Vietnamese-specific rule prompt module and a train-derived MFR dictionary.

## Files

- `language_rules/vi.py`: Vietnamese-specific detection guidance, few-shot examples, and candidate heuristics.
- `mfr_dictionaries/vi_mfr_dictionary.json`: train-derived Vietnamese MFR dictionary.
- `mfr_dictionaries/vi_mfr.py`: safe lookup helper for the Vietnamese MFR dictionary.

## Current project VI data summary

- Train VI tokens: 101,793
- Train changed tokens: 16,287 (16.00%)
- Validation VI tokens: 13,651
- Validation changed tokens: 2,133 (15.63%)
- High-confidence MFR pairs: 545
- Review pairs: 64
- Ambiguous pairs: 90

## Recommended integration

1. Use `lookup_vi_mfr` first for `high_confidence_pairs` only.
2. Send review/ambiguous/OOV tokens to target-token LLM detection using `language_rules.vi` guidance.
3. Preserve uncertain tokens to avoid overnormalization.
4. Keep paper-derived `paper_seed_pairs` disabled by default unless external resources are allowed.

## Example

```python
from language_rules.vi import build_vi_detection_prompt, is_vi_likely_candidate
from mfr_dictionaries.vi_mfr import load_vi_mfr_dictionary, apply_vi_mfr_to_tokens

tokens = ["t", "ko", "biết", "ngta", "zui"]
d = load_vi_mfr_dictionary("mfr_dictionaries/vi_mfr_dictionary.json")
print(apply_vi_mfr_to_tokens(tokens, d))
print([i for i, tok in enumerate(tokens) if is_vi_likely_candidate(tok)])
print(build_vi_detection_prompt(tokens))
```
