# Common Prompt v2 for MultiLexNorm-style Normalization

This package contains a consolidated `common_prompt.py` that should replace the duplicate `prompts/common_prompt.py` files inside the language-specific packages.

## Core changes

- Target-token/span only: the model must not rewrite the full sentence.
- Raw sentence + indexed tokenized sentence are both provided.
- Strong protected-token policy for hashtags, mentions, URLs, emails, emojis, numbers, dates, times, separators, punctuation, alphanumeric entities, and named entities.
- Explicit anti-translation and anti-paraphrase rules.
- Code-switching rule: normalize the target surface form in its own language; do not translate across languages.
- Capitalization caution for German/Italian/Dutch/English-style sentence-initial or entity-related casing.
- Short-token ambiguity rule: if uncertain, preserve.
- Output parsing and post-processing helpers.

## Recommended usage

```python
from prompts.common_prompt import (
    build_common_detection_prompt,
    build_common_normalization_prompt,
    find_protected_indices,
    safe_normalization_result,
)

prompt = build_common_normalization_prompt(
    lang="th",
    sentence_tokens=["ศิริปันนา", "แฟมิลี่", "แฟร์", "_", "#BNK48"],
    target_index=1,
    raw_sentence="ศิริปันนา แฟมิลี่ แฟร์ _ #BNK48",
    language_rule_block="Thai-specific guidance...",
    fewshot_examples=[]
)
```

## Pipeline recommendation

1. Compute protected indices.
2. Apply high-confidence MFR only, never to protected tokens.
3. Run language-specific candidate selector.
4. Remove protected indices from candidates.
5. Run target detection prompt.
6. Run target normalization prompt only for label=1.
7. Parse JSON.
8. Apply `safe_normalization_result()`.
9. If parsing fails or model is uncertain, preserve the raw token.
