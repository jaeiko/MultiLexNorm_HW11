# Italian Normalization v2 Package

This package upgrades the previous Italian module using Italian lexical-normalization literature and the existing project data pattern analysis.

## Files

```text
italian_normalization_v2_package/
  prompts/
    common_prompt.py
  language_rules/
    it.py
  mfr_dictionaries/
    it_mfr_dictionary.json
    it_mfr.py
  it_v2_summary.csv
  it_v2_top_pairs.csv
  README.md
```

## Main design change from v1

Italian v1 was intentionally conservative, but its internal-dev MFR recall was low. v2 keeps the production path conservative while improving candidate injection for tokens that XLM-R may miss:

- accent/apostrophe restoration: `e' → è`, `perche' → perché`, `puo' → può`, `pero' → però`
- social-media abbreviations: `nn → non`, `ke → che`, `cmq → comunque`, `cn → con`
- repeated-character candidates: `Beppeee → Beppe`, `ciaoooo → ciao`
- cautious span candidates: `Vabbene → va bene`, `un ultima → un'ultima`
- stronger negative rules for hashtags, usernames, acronyms, named entities, clitics, preposition contractions, phrasal abbreviations, and non-words

## Recommended runtime pipeline

```text
raw_sentence + tokens + lang=it
→ existing trigram
→ apply_it_mfr_to_tokens(mode="conservative")
→ XLM-R detection
→ add language_rules.it.candidate_indices()
→ optionally add language_rules.it.candidate_spans()
→ remove protected candidates
→ LLM only for untouched candidates
→ common_prompt.safe_normalization_result()
```

## MFR modes

```python
from mfr_dictionaries.it_mfr import load_it_mfr_dictionary, apply_it_mfr_to_tokens

dictionary = load_it_mfr_dictionary()
tokens = ["joker", "cmq", "nn", "e'", "nnt", "di", "ke", "!"]

safe = apply_it_mfr_to_tokens(tokens, dictionary, mode="conservative")
```

Recommended production mode: `conservative`.

Available modes:

- `conservative`: uses `high_confidence_pairs` only.
- `accent_abbrev`: adds accent/apostrophe and abbreviation pairs. Use for ablation or if FP is controlled by guard.
- `balanced`: uses broader analysis pairs. Use for experiments only.

## Important policies

- Do not automatically split hashtags or usernames.
- Do not expand `lol`, `omg`, `ahahah`, or similar non-words/interjections unless project data shows an exact gold pair.
- Do not expand Italian clitics: `mi → mi`, `ci → ci`.
- Do not split standard preposition contractions: `del`, `della`, `alla`, `sull'`.
- Be conservative with capitalization. Correct clear ordinary all-caps emphasis, but preserve acronyms and entity-like words.
- Numbers remain protected by the shared prompt. Paper examples like `6 → sei` are candidate seeds, not automatic replacements.

## Development note

The provided official validation split contained no Italian rows in the earlier analysis, so the included metrics are internal-dev diagnostics from a deterministic train split. The next recommended step is to run a language-wise ablation on the current full pipeline:

1. current baseline
2. v2 conservative MFR only
3. v2 candidate injection + XLM-R
4. v2 candidate injection + LLM fallback
5. guard on/off
