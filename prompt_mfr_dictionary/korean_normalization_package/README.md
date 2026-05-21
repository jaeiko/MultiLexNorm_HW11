# Korean Normalization v2 Package

This package improves the Korean module using the uploaded Korean noisy-text,
K-MT tokenizer, and KOLD papers.

## Files

- `prompts/common_prompt.py`: common target-only prompt and protected-token guards.
- `language_rules/ko.py`: Korean-specific rule block, few-shot examples, and candidate selector.
- `mfr_dictionaries/ko_mfr_dictionary.json`: train-derived Korean MFR dictionary with paper-informed metadata.
- `mfr_dictionaries/ko_mfr.py`: Korean MFR loader and guarded application utility.
- `ko_v2_summary.csv`: Korean data and dictionary statistics.
- `ko_v2_top_pairs.csv`: project pattern examples and high-confidence pairs.

## Recommended pipeline

1. Protect tokens with `common_prompt.find_protected_indices`.
2. Apply `apply_ko_mfr_to_tokens(..., mode="conservative")` only to high-confidence pairs.
3. Run `language_rules.ko.candidate_indices()` on remaining tokens.
4. For each candidate, use target-only detection and normalization prompts.
5. Use `common_prompt.safe_normalization_result()` after parsing LLM output.

## Key policy

Korean noisy-looking text should not be blindly normalized. Preserve laughter,
emotion/backchannel compatibility-jamo tokens, hashtags, mentions, proper nouns,
coinages, named entities, and community terms unless project-data evidence and
local context support a target-level normalization.
