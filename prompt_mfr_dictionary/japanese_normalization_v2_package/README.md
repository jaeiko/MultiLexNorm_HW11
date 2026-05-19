# Japanese Normalization v2 Package

This package contains a Japanese-specific lexical-normalization module for MultiLexNorm-style data.
It is designed to be conservative and boundary-aware.

## Files

```text
japanese_normalization_v2_package/
  prompts/common_prompt.py
  language_rules/ja.py
  mfr_dictionaries/ja_mfr_dictionary.json
  mfr_dictionaries/ja_mfr.py
  ja_v2_summary.csv
  ja_v2_top_pairs.csv
  README.md
```

## Design

Japanese lexical normalization is not just token replacement. Because Japanese is normally written without spaces,
normalization often requires boundary-aware target span detection and conversion.

The module follows this runtime order:

1. Protect hashtags, mentions, URLs, numbers, dates, times, IDs, and other social-media artifacts.
2. Apply `high_confidence_pairs` from `ja_mfr_dictionary.json` in conservative mode.
3. Use `language_rules/ja.py::candidate_indices()` and optionally `candidate_spans()` to select remaining candidates.
4. Run target-only detection and normalization prompts.
5. Apply `safe_normalization_result()` from `common_prompt.py`.
6. Preserve the original target if uncertain.

## Production vs ablation

Use only `high_confidence_pairs` for automatic replacement. `balanced_pairs_for_analysis`, `review_pairs`,
`ambiguous_pairs`, and `context_sensitive_tokens` are for ablation, prompt few-shot design, and model fallback.

## Key risk controls

- Do not translate or paraphrase.
- Do not rewrite full sentences.
- Do not normalize emotional/pronunciation variation only because it looks informal.
- Do not expand named entities or product/group names without strong evidence.
- Treat punctuation insertion and sentence-final particles as context-dependent.
