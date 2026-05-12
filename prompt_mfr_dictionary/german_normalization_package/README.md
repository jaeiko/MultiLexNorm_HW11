# German (`de`) normalization package

This package adds a data-driven German lexical-normalization module for the project.

## Files

```text
prompts/common_prompt.py
language_rules/de.py
mfr_dictionaries/de_mfr_dictionary.json
mfr_dictionaries/de_mfr.py
de_summary.csv
de_top_pairs.csv
```

## Data summary

- Train sentences: 1,628
- Train tokens: 15,006
- Train changed tokens: 2,578
- Train changed ratio: 17.18%
- Validation sentences: 573
- Validation tokens: 4,860
- Validation changed tokens: 873
- Validation changed ratio: 17.96%

## Main German patterns

The project data shows that German normalization is dominated by capitalization, colloquial spelling, contractions, umlaut restoration, and split/merge normalization:

- `nich/net/ned -> nicht`
- `nix -> nichts`
- `grad/grade -> gerade`
- `heut -> heute`
- `hab -> habe`, `habs -> habe es`
- `gibts -> gibt es`, `gehts -> geht es`
- `fuer/fur -> für`
- `wär -> wäre`, `würd -> würde`, `werd -> werde`
- `zuhause -> zu Hause`, `schonmal -> schon mal`, `naja -> Na ja`, `Achso -> Ach so`
- sentence-initial and noun capitalization, but only when supported by context/evidence

## Pattern counts

- other_substitution: 1188
- case_change: 894
- whitespace_change: 226
- expansion_or_abbreviation_restore: 177
- repeated_char_reduction: 49
- empty_norm_alignment: 29
- digit_related_change: 14
- shortening_or_deletion: 1

## MFR usage

Use only `high_confidence_pairs` for automatic replacement. `review_pairs` and `ambiguous_pairs` should be passed to the target-token prompt or a downstream model.

```python
from mfr_dictionaries.de_mfr import load_de_mfr_dictionary, apply_de_mfr_to_tokens

de_dict = load_de_mfr_dictionary()
tokens = ["ich", "hab", "grad", "nix", "gesehen", "#Tatort2024"]
first_pass = apply_de_mfr_to_tokens(tokens, de_dict, mode="conservative")
```

## Candidate prompt usage

```python
from language_rules.de import candidate_indices, build_de_target_normalization_prompt

cands = candidate_indices(tokens)
prompt = build_de_target_normalization_prompt(tokens, cands[0], raw_sentence=" ".join(tokens))
```

## Validation MFR diagnosis

| mode         |   dictionary_size |   validation_tokens |   validation_changed_tokens |   TP |   FP |   FN |   TN |      ERR |   precision |   recall |
|:-------------|------------------:|--------------------:|----------------------------:|-----:|-----:|-----:|-----:|---------:|------------:|---------:|
| LAI          |                 0 |                4860 |                         873 |    0 |    0 |  873 | 3987 | 0        |    0        | 0        |
| balanced     |               230 |                4860 |                         873 |  221 |   48 |  652 | 3974 | 0.198167 |    0.821561 | 0.25315  |
| conservative |                75 |                4860 |                         873 |  108 |   13 |  765 | 3982 | 0.10882  |    0.892562 | 0.123711 |

The conservative dictionary avoids blind capitalization of high-frequency function words such as `ich`, `das`, `aber`, `wie`, `und`, and `die`. These should be handled by the target-token prompt/model with full sentence context.
