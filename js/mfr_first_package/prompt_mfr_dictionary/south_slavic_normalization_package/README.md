# South Slavic normalization package (sr/hr/sl)

This package extends the common target-only lexical-normalization prompt to Serbian, Croatian, and Slovenian.

## Files

```text
prompts/common_prompt.py
language_rules/south_slavic.py
language_rules/sr.py
language_rules/hr.py
language_rules/sl.py
mfr_dictionaries/south_slavic_mfr.py
mfr_dictionaries/sr_mfr_dictionary.json
mfr_dictionaries/hr_mfr_dictionary.json
mfr_dictionaries/sl_mfr_dictionary.json
mfr_dictionaries/sr_mfr.py
mfr_dictionaries/hr_mfr.py
mfr_dictionaries/sl_mfr.py
south_slavic_summary.csv
```

## Recommended pipeline

1. Compute protected indices with `find_protected_indices()`.
2. Apply high-confidence MFR replacements. Use `skip_context_sensitive=True` for the safest mode, or `False` for stronger MFR-only behavior.
3. Run `candidate_indices()` from `sr.py`, `hr.py`, or `sl.py` on remaining tokens.
4. Remove protected indices from candidate indices.
5. Use target-token detection and normalization prompts.
6. If output is uncertain, invalid JSON, or attempts to change protected tokens, preserve the original token.

## Example

```python
from language_rules.hr import candidate_indices, build_hr_target_normalization_prompt
from mfr_dictionaries.hr_mfr import load_hr_mfr_dictionary, apply_hr_mfr_to_tokens

raw_sentence = "ak neš ne mogu smislit #karla_photography"
tokens = raw_sentence.split()

d = load_hr_mfr_dictionary()
first_pass = apply_hr_mfr_to_tokens(tokens, d, skip_context_sensitive=True)
indices = candidate_indices(first_pass)
prompt = build_hr_target_normalization_prompt(first_pass, indices[0], raw_sentence=raw_sentence)
```
