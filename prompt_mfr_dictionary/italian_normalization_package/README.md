# Italian normalization package

Data-driven Italian (`it`) module for MultiLexNorm-style lexical normalization.

## Files

- `prompts/common_prompt.py`: shared target-only prompt and protected-token guard.
- `language_rules/it.py`: Italian rule block, few-shot examples, candidate selector, prompt wrappers.
- `mfr_dictionaries/it_mfr_dictionary.json`: train-derived MFR dictionary split into high-confidence / review / ambiguous pairs.
- `mfr_dictionaries/it_mfr.py`: loader and MFR application utilities.
- `it_summary.csv`: internal-dev MFR diagnostics.
- `it_top_pairs.csv`: top observed raw→norm pairs.

## Data summary

- Train sentences: 593
- Train tokens: 12645
- Train changed tokens: 926
- Train changed ratio: 0.0732
- Official validation rows: 0

Because the official validation split contains no Italian rows, diagnostics use a deterministic 90/10 split of the train rows.

## Recommended pipeline

```python
from language_rules.it import candidate_indices, build_it_target_normalization_prompt
from mfr_dictionaries.it_mfr import load_it_mfr_dictionary, apply_it_mfr_to_tokens

it_dict = load_it_mfr_dictionary()
tokens = ["nn", "so", "perche'", "e'", "successo", "#Roma2024"]
raw_sentence = "nn so perche' e' successo #Roma2024"

first_pass = apply_it_mfr_to_tokens(tokens, it_dict, mode="conservative")
cands = candidate_indices(first_pass)
prompt = build_it_target_normalization_prompt(first_pass, cands[0], raw_sentence=raw_sentence)
```

Use `high_confidence_pairs` for automatic replacement. Send `review_pairs`, `ambiguous_pairs`, and case-sensitive proper/entity-like tokens to target-token prompt/model fallback.

## Notes

- Accent/apostrophe restoration is central: `e'→è`, `perche'→perché`, `puo'→può`, `pero'→però`.
- Abbreviation restoration includes `nn→non`, `ke→che`, `cmq→comunque`, `dx→destra`, `sx→sinistra`.
- Casing is frequent but risky; avoid global titlecasing or lowercasing.
- Preserve hashtags, mentions, URLs, numbers, emojis, acronyms, product names, political parties, organizations, usernames, and named entities.
