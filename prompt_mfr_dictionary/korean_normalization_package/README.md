# Korean Normalization Package

This package contains a Korean-specific lexical normalization module for MultiLexNorm-style data.

## Files

```text
korean_normalization_package/
  language_rules/
    ko.py
  mfr_dictionaries/
    ko_mfr_dictionary.json
    ko_mfr.py
  ko_summary.csv
  README.md
```

## Strategy

Korean social-media text contains intentional text gaming, memes, profanity avoidance, compatibility-jamo abbreviations, leetspeak-like digit/Latin mixing, and expressive particles. This package therefore uses a conservative hybrid pipeline:

```text
Korean input tokens
→ strict high-confidence MFR lookup
→ Korean-specific candidate detection
→ context-aware target-token prompt/model fallback
→ uncertain cases preserve original token
```

## Validation summary

```text
Train tokens: 13130
Train changed tokens: 958
Train changed ratio: 0.072963
Validation tokens: 1880
Validation changed tokens: 166
Validation changed ratio: 0.088298
High-confidence MFR pairs: 24
Validation ERR with high-confidence MFR only: 0.102410
TP / FP / FN: 21 / 4 / 145
```

## Usage

```python
from mfr_dictionaries.ko_mfr import load_ko_mfr_dictionary, apply_ko_mfr_to_tokens
from language_rules.ko import candidate_indices, build_ko_target_prompt

ko_dict = load_ko_mfr_dictionary("mfr_dictionaries/ko_mfr_dictionary.json")
tokens = ["ㄹㅇ", "존나", "귀엽다", "ㅋㅋ"]
first_pass = apply_ko_mfr_to_tokens(tokens, ko_dict)
# ["진짜", "존나", "귀엽다", "ㅋㅋ"]
# 존나 is intentionally not direct-replaced because it is context-dependent in train data.

cands = candidate_indices(tokens)
prompt = build_ko_target_prompt(tokens, cands[0])
```

## Important notes

- Apply only `high_confidence_pairs` automatically.
- Do not automatically rewrite `review_pairs` or `ambiguous_pairs`.
- Preserve laughter/emotion tokens such as `ㅋㅋ`, `ㅎㅎ`, `ㅠㅠ`, `ㅜㅜ` by default.
- Treat intensifiers such as `존나`, `개`, `씹`, `좆` as context-dependent candidates rather than direct replacements.
- Compatibility-jamo abbreviations such as `ㄹㅇ`, `ㅅㅂ`, `ㅈㄴ`, `ㅇㅈㄹ` are high-value candidates, but output should follow the exact project annotation style.
