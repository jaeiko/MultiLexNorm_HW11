# Spanish Normalization Package

Data-driven Spanish module for MultiLexNorm-style lexical normalization.

## Files

```text
spanish_normalization_package/
  prompts/common_prompt.py
  language_rules/es.py
  mfr_dictionaries/es_mfr_dictionary.json
  mfr_dictionaries/es_mfr.py
  es_summary.csv
  es_top_pairs.csv
```

## Data summary

- Train sentences: 568
- Train tokens: 7189
- Train changed tokens: 553
- Train changed ratio: 7.69%
- Official validation sentences: 0
- Official validation changed ratio: 0.00%

The provided official validation split contains no Spanish rows, so `es_summary.csv` reports diagnostics on a deterministic 90/10 internal split of the train rows.

## Recommended pipeline

1. Apply `high_confidence_pairs` with `apply_es_mfr_to_tokens(..., mode="conservative")`.
2. Run `language_rules.es.candidate_indices()` on remaining tokens.
3. Exclude protected indices.
4. Use target-token detection and normalization prompts.
5. If uncertain or parsing fails, preserve the original token.

## Typical stable patterns

```text
tambien → también
despues → después
aqui → aquí
noo → no
sii → sí
cn → con
kiero → quiero
jajajaj → ja
```

## Context-sensitive forms

```text
q/k/ke/qe → que or qué
si → si or sí
pa → para, but may be context-sensitive
to → todo/toda
```

These should generally go to target-token prompt/model fallback unless the exact high-confidence pair is present.
