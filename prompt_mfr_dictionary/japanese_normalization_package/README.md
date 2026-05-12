# Japanese Normalization Package

This package adds Japanese-specific lexical-normalization support for the MultiLexNorm-style project.

## Files

```text
japanese_normalization_package/
  language_rules/
    ja.py
  mfr_dictionaries/
    ja_mfr_dictionary.json
    ja_mfr.py
  ja_summary.csv
  README.md
```

## Why Japanese needs a special path

Japanese is not normally space-delimited, and lexical normalization often depends on boundary/span decisions. The package therefore uses conservative direct MFR and sends contextual cases to target-token prompts.

## Recommended pipeline

```text
Japanese tokens
→ apply high-confidence MFR only
→ identify remaining candidate tokens with language_rules.ja.candidate_indices
→ send ambiguous/OOV candidates to target-token prompt/model
→ preserve original token if uncertain
```

## Quick usage

```python
from mfr_dictionaries.ja_mfr import load_ja_mfr_dictionary, apply_ja_mfr_to_tokens
from language_rules.ja import candidate_indices, build_ja_target_prompt

d = load_ja_mfr_dictionary("mfr_dictionaries/ja_mfr_dictionary.json")
tokens = ["やっぱり", "スマホ", "便利", "です"]
first_pass = apply_ja_mfr_to_tokens(tokens, d)
# ["やはり", "スマートフォン", "便利", "です"]

cands = candidate_indices(tokens)
prompt = build_ja_target_prompt(tokens, cands[0]) if cands else None
```

## Conservative direct-MFR rule

The direct dictionary uses only deterministic train mappings and excludes common function tokens and sentence-final punctuation insertions. In validation, this strict dictionary achieved a small positive ERR with zero false positives in the local check:

```text
ERR: 0.060029
TP: 41
FP: 0
FN: 642
```

Use `review_pairs` and `ambiguous_pairs` for analysis/prompting only; do not directly replace them.
