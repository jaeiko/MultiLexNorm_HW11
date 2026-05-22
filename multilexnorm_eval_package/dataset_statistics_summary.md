# Dataset Statistics Summary

## Files analyzed

- Official validation: `/mnt/data/validation-00000-of-00001.parquet`
- Internal 17-language validation: `dataset_17lang/data/validation-00000-of-00001.parquet`
- Missing-language pseudo validation only: `dataset_17lang/pseudo_validation_missing_langs.parquet`
- Official train reference for official validation OOV/MFR: `/mnt/data/train-00000-of-00001.parquet`
- Internal train reference for internal validation OOV/MFR: `dataset_17lang/data/train-00000-of-00001.parquet`

## Overall dataset statistics

| Dataset | Languages | Samples | Tokens | Changed tokens | Changed ratio | Raw OOV ratio | MFR changed correct ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_validation | 12 | 8,408 | 125,377 | 14,444 | 0.115205 | 0.185704 | 0.535655 |
| internal_validation_17lang | 17 | 9,056 | 136,462 | 16,155 | 0.118385 | 0.199491 | 0.506035 |
| pseudo_missing_5lang | 5 | 648 | 11,085 | 1,711 | 0.154353 | 0.355435 | 0.255991 |

## Built-in baseline sanity check

| Dataset | Baseline | Accuracy | ERR | Macro ERR | Detection Precision | Detection Recall | Detection F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| official_validation | LAI | 0.884795 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| official_validation | MFR | 0.941313 | 0.452922 | 0.396487 | 0.866211 | 0.535655 | 0.661961 |
| internal_validation_17lang | LAI | 0.881615 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| internal_validation_17lang | MFR | 0.936180 | 0.424698 | 0.337539 | 0.861524 | 0.506035 | 0.637576 |
| pseudo_missing_5lang | LAI | 0.845647 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pseudo_missing_5lang | MFR | 0.878124 | 0.186441 | 0.196066 | 0.786355 | 0.255991 | 0.386243 |

## Main interpretation

- The official validation set contains 12 languages; the internal validation set contains 17 languages by appending the held-out pseudo validation examples for `da`, `es`, `it`, `tr`, and `trde`.
- The internal 17-language validation set is only slightly larger overall, but the missing-language-only subset has much higher OOV pressure: raw OOV ratio is 0.355435 and changed-token OOV ratio is 0.638223.
- MFR is a strong sanity-check baseline on official validation, but its macro ERR drops from 0.396487 on official validation to 0.337539 on internal 17-language validation and 0.196066 on the missing-language-only subset.
- For fair held-out evaluation, build dictionaries, n-gram resources, prompt examples, and detector training data from `dataset_17lang/data/train-00000-of-00001.parquet` only.
