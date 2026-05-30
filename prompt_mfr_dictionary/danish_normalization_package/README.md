# Danish normalization package

This package adds a data-driven Danish (`da`) lexical-normalization module.

## Files

```text
danish_normalization_package/
  prompts/common_prompt.py
  language_rules/da.py
  mfr_dictionaries/da_mfr_dictionary.json
  mfr_dictionaries/da_mfr.py
  da_summary.csv
  da_top_pairs.csv
```

## Dataset pattern summary

- Train sentences: 719
- Train tokens: 16448
- Train changed tokens: 1521
- Train changed ratio: 9.25%
- Official validation rows: 0 Danish rows
- Diagnostic split: deterministic 90/10 internal split, seed=42

Top patterns include Danish letter restoration (`vaere→være`, `ogsa→også`, `paa/pa→på`, `sa/saa→så`), q-style substitutions (`jeq→jeg`, `oq→og`, `diq→dig`, `miq→mig`), abbreviation restoration (`ikk/ik/ek→ikke`, `pga→pga.`, `fx→f.eks.`), and split normalization (`idag→i dag`).

## Recommended pipeline

1. Apply `apply_da_mfr_to_tokens(..., mode="conservative")`.
2. Use `language_rules.da.candidate_indices()` for remaining candidates.
3. Send ambiguous candidates to target-token prompt/model fallback.
4. Keep protected tokens unchanged via `common_prompt.is_protected_token()`.

## Warning

Very short forms such as `p`, `s`, `a`, `r`, `gr`, `fr`, `fa`, `pa`, `sa`, `ma`, `n`, and `t` are context-sensitive. Avoid global replacement unless the specific pair is in `high_confidence_pairs` and validation/internal-dev ablation confirms low FP.
