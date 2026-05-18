# Turkish / Turkish-German normalization package

This package adds language-specific prompt rules and MFR dictionaries for `tr` and `trde`.

## Structure

```text
turkish_normalization_package/
  prompts/common_prompt.py
  language_rules/turkish_common.py
  language_rules/tr.py
  language_rules/trde.py
  mfr_dictionaries/tr_mfr_dictionary.json
  mfr_dictionaries/trde_mfr_dictionary.json
  mfr_dictionaries/turkish_mfr.py
  turkish_summary.csv
  tr_top_pairs.csv
  trde_top_pairs.csv
```

## Recommended pipeline

1. Build `raw_sentence` and `tokens`.
2. Apply protected-token guard.
3. Apply high-confidence MFR only.
4. Run language-specific `candidate_indices()` on remaining tokens.
5. Remove protected indices from candidates.
6. Use target-token detection prompt.
7. If label is 1, use target-token normalization prompt.
8. If parsing fails or model is uncertain, preserve the original token.

## Turkish strategy

`tr` focuses on deasciification, vowel restoration, accent/spoken suffix normalization, clitic spacing, proper-noun apostrophe restoration, and repeated-character reduction.

## Turkish-German strategy

`trde` is code-switching-aware. Turkish tokens should be normalized using Turkish rules, German tokens using German rules, and mixed/ambiguous tokens should be handled conservatively. Do not translate Turkish into German or German into Turkish.

## MFR policy

Only `high_confidence_pairs` are intended for automatic replacement. `review_pairs` and `ambiguous_pairs` should be used for analysis, prompt examples, or model fallback only.
