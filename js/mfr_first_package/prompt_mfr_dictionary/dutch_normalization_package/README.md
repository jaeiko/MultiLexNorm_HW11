# Dutch (`nl`) normalization package

This package adds a data-driven Dutch lexical-normalization module for the project.

## Files

```text
prompts/common_prompt.py
language_rules/nl.py
mfr_dictionaries/nl_mfr_dictionary.json
mfr_dictionaries/nl_mfr.py
nl_summary.csv
nl_top_pairs.csv
```

## Data summary

- Train sentences: 907
- Train tokens: 12,381
- Train changed tokens: 3,672
- Train changed ratio: 29.66%
- Validation sentences: 308
- Validation tokens: 3,863
- Validation changed tokens: 1,093
- Validation changed ratio: 28.29%

## Main Dutch patterns

The project data shows that Dutch normalization is dominated by abbreviation/colloquial restoration and clitic splitting:

- `ni/nie/nt -> niet`
- `ff/effe -> even`
- `mss -> misschien`
- `gwn -> gewoon`
- `mn/m'n -> mijn`
- `aant -> aan het`
- `kheb -> ik heb`, `kga -> ik ga`, `kben -> ik ben`
- `da -> dat`, `goe -> goed`, `ma -> maar`, `wa -> wat`
- `prive -> privé`, `Oke/Okee -> Oké`

## MFR usage

Use only `high_confidence_pairs` for automatic replacement. `review_pairs` and `ambiguous_pairs` should be passed to the target-token prompt or a downstream model.

```python
from mfr_dictionaries.nl_mfr import load_nl_mfr_dictionary, apply_nl_mfr_to_tokens

nl_dict = load_nl_mfr_dictionary()
tokens = ["kheb", "da", "ni", "gezien", "#feest"]
first_pass = apply_nl_mfr_to_tokens(tokens, nl_dict, mode="conservative")
```

## Candidate prompt usage

```python
from language_rules.nl import candidate_indices, build_nl_target_normalization_prompt

cands = candidate_indices(tokens)
prompt = build_nl_target_normalization_prompt(tokens, cands[0], raw_sentence=" ".join(tokens))
```

## Validation MFR diagnosis

| mode         |   dictionary_size |   validation_tokens |   validation_changed_tokens |   TP |   FP |   FN |   TN |      ERR |   precision |   recall |
|:-------------|------------------:|--------------------:|----------------------------:|-----:|-----:|-----:|-----:|---------:|------------:|---------:|
| LAI          |                 0 |                3863 |                        1093 |    0 |    0 | 1093 | 2770 | 0        |    0        | 0        |
| balanced     |               297 |                3863 |                        1093 |  256 |   27 |  837 | 2764 | 0.209515 |    0.904594 | 0.234218 |
| conservative |               126 |                3863 |                        1093 |  143 |   10 |  950 | 2768 | 0.121683 |    0.934641 | 0.130833 |

The conservative dictionary is designed to keep false positives low. The balanced dictionary is included for analysis/ablation and should not be used blindly in the final pipeline.
