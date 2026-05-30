# English Normalization Package

Data-driven English module for MultiLexNorm-style lexical normalization.

## Files

```text
english_normalization_package/
  prompts/common_prompt.py
  language_rules/en.py
  mfr_dictionaries/en_mfr_dictionary.json
  mfr_dictionaries/en_mfr.py
  en_summary.csv
  en_top_pairs.csv
```

## Data summary

- Train sentences: 2360
- Train tokens: 35216
- Train changed tokens: 2666
- Train changed ratio: 7.57%
- Validation sentences: 590
- Validation tokens: 9169
- Validation changed tokens: 633
- Validation changed ratio: 6.90%

## Recommended pipeline

1. Apply `high_confidence_pairs` with `apply_en_mfr_to_tokens(..., mode="conservative")`.
2. Run `language_rules.en.candidate_indices()` on remaining tokens.
3. Exclude protected indices except context-sensitive numeric shorthand when explicitly targeted.
4. Use target-token detection and normalization prompts.
5. If uncertain or parsing fails, preserve the original token.

## Notes

English has relatively low changed ratio compared with Dutch/Indonesian/Turkish, but it contains many stable high-frequency forms:

```text
u → you
im → i'm
dont → don't
pls → please
ppl → people
gonna → going to
tryna → trying to
yall → y'all
```

Numeric shorthand such as `2 → to` and `4 → for` appears in the data, but pure numeric tokens are protected by default, so they should be handled by context-aware prompt/model rather than direct MFR.
