# Indonesian Normalization Package

This package adds Indonesian (`id`) and Indonesian-English code-mixed (`iden`) lexical-normalization support for the MultiLexNorm-style project.

## Files

```text
indonesian_normalization_package/
  language_rules/
    id.py
  mfr_dictionaries/
    id_mfr_dictionary.json
    id_mfr.py
  id_summary.csv
  README.md
```

## Why Indonesian needs a special path

Indonesian social media text has many high-frequency colloquial abbreviations and slang forms. The same raw token may normalize differently in Indonesian-only and Indonesian-English code-mixed data; for example, `ga/gak` maps mostly to `enggak` in `id`, but to `tidak` in `iden` in the current training split.

## Recommended pipeline

```text
Indonesian/id-en tokens
→ apply language-specific high-confidence MFR only
→ detect remaining candidates with language_rules.id.candidate_indices
→ send ambiguous/OOV candidates to target-token prompt/model
→ preserve original token if uncertain
```

## Quick usage

```python
from mfr_dictionaries.id_mfr import load_id_mfr_dictionary, apply_id_mfr_to_tokens
from language_rules.id import candidate_indices, build_id_target_prompt

d = load_id_mfr_dictionary("mfr_dictionaries/id_mfr_dictionary.json")
tokens = ["aku", "gak", "liat", "yg", "baru"]
first_pass = apply_id_mfr_to_tokens(tokens, d, lang="id")
# ["aku", "enggak", "lihat", "yang", "baru"]

cands = candidate_indices(tokens, lang="id")
prompt = build_id_target_prompt(tokens, cands[0], lang="id") if cands else None
```

For `iden`:

```python
tokens = ["i'm", "gak", "sure", "yg", "itu"]
first_pass = apply_id_mfr_to_tokens(tokens, d, lang="iden")
# ["i am", "tidak", "sure", "yang", "itu"]  # when high-confidence entries exist
```

## Local validation summary

Using direct high-confidence MFR only:

```text
id   ERR: 0.545941, TP: 1128, FP: 5,  FN: 929
iden ERR: 0.499145, TP: 308,  FP: 16, FN: 277
```

Use `review_pairs` and `ambiguous_pairs` for analysis/prompting only; do not directly replace them.
