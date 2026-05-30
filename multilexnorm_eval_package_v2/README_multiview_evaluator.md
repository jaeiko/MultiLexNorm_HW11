# MultiLexNorm Evaluator v2: 17-lang prediction multi-view evaluation

## Purpose

This version extends the previous evaluator so that one prediction file generated on the internal 17-language validation set can be evaluated from multiple views without rerunning inference.

This is especially useful for LLM experiments, because LLM inference is slow and expensive. The recommended workflow is:

```text
Run LLM inference once on internal_v1/data/validation-00000-of-00001.parquet
        ↓
Save one prediction file in the same sentence order
        ↓
Evaluate the same prediction file as:
  1. all        = all 17 languages
  2. official12 = languages already present in the original official validation split
  3. missing5   = pseudo-held-out languages added from train
```

## Main changes from v1

### 1. Added `--eval_groups`

New option for `evaluate`:

```bash
--eval_groups all official12 missing5
```

Supported values:

| Group | Meaning |
|---|---|
| `all` | Evaluate all sentences in the provided gold parquet |
| `official12` | Evaluate only the 12 languages from the original official validation set |
| `missing5` | Evaluate only the 5 added held-out languages |

Default is:

```bash
--eval_groups all
```

So old commands still work.

### 2. Added default language groups

Default official 12 languages:

```python
{id, ja, ko, th, vi, de, en, hr, iden, nl, sl, sr}
```

Default missing 5 languages:

```python
{da, es, it, tr, trde}
```

These match the current project split.

### 3. Added custom language group override

If the split changes later, override groups from CLI:

```bash
--official_langs id,ja,ko,th,vi,de,en,hr,iden,nl,sl,sr \
--missing_langs da,es,it,tr,trde
```

### 4. Output files now include group suffixes

Example command:

```bash
python multilexnorm_evaluator_v2.py evaluate \
  --gold_parquet internal_v1/data/validation-00000-of-00001.parquet \
  --pred_path predictions_llm_17lang.json \
  --model_name llm_prompt_fewshot \
  --dataset_name internal_validation_17lang \
  --eval_groups all official12 missing5 \
  --out_dir outputs/results
```

Generated files:

```text
outputs/results/
├── internal_validation_17lang_all_llm_prompt_fewshot.json
├── internal_validation_17lang_all_llm_prompt_fewshot_overall.csv
├── internal_validation_17lang_all_llm_prompt_fewshot_language_metrics.csv
├── internal_validation_17lang_official12_llm_prompt_fewshot.json
├── internal_validation_17lang_official12_llm_prompt_fewshot_overall.csv
├── internal_validation_17lang_official12_llm_prompt_fewshot_language_metrics.csv
├── internal_validation_17lang_missing5_llm_prompt_fewshot.json
├── internal_validation_17lang_missing5_llm_prompt_fewshot_overall.csv
├── internal_validation_17lang_missing5_llm_prompt_fewshot_language_metrics.csv
├── internal_validation_17lang_llm_prompt_fewshot_eval_groups.json
└── internal_validation_17lang_llm_prompt_fewshot_eval_groups_overall.csv
```

The last two files combine all requested groups into one summary.

## Important alignment rule

The prediction file must be generated against the full gold parquet in exactly the same sentence order.

Required conditions:

```python
len(predictions) == len(gold_samples)
len(predictions[i]) == len(gold_samples[i]["norm"])
predictions[i] corresponds to gold_samples[i]
```

Do not sort by language or remove samples during LLM inference unless you also restore the original order before saving predictions.

## Prediction format

The evaluator accepts the same formats as v1.

### JSON nested list

```json
[
  ["token1", "token2"],
  ["token3"]
]
```

### JSON object

```json
{
  "predictions": [
    ["token1", "token2"],
    ["token3"]
  ]
}
```

### JSONL

One sentence per line:

```jsonl
["token1", "token2"]
["token3"]
```

### CSV

A column named one of `pred`, `prediction`, `predictions`, `norm`, or `normalized`, where each cell is a JSON list string.

## Common commands

### Evaluate LLM prediction once, three views

```bash
python multilexnorm_evaluator_v2.py evaluate \
  --gold_parquet internal_v1/data/validation-00000-of-00001.parquet \
  --pred_path predictions_llm_17lang.json \
  --model_name llm_prompt_fewshot \
  --dataset_name internal_validation_17lang \
  --eval_groups all official12 missing5 \
  --out_dir outputs/results
```

### Evaluate MFR baseline in the same three views

```bash
python multilexnorm_evaluator_v2.py evaluate \
  --gold_parquet internal_v1/data/validation-00000-of-00001.parquet \
  --train_parquet internal_v1/data/train-00000-of-00001.parquet \
  --baseline mfr \
  --model_name mfr \
  --dataset_name internal_validation_17lang \
  --eval_groups all official12 missing5 \
  --out_dir outputs/results
```

### Evaluate only the full dataset, old behavior

```bash
python multilexnorm_evaluator_v2.py evaluate \
  --gold_parquet validation-00000-of-00001.parquet \
  --pred_path predictions.json \
  --model_name my_model \
  --dataset_name official_validation \
  --out_dir outputs/results
```

## Metrics included

Each result includes:

- token accuracy
- LAI accuracy
- ERR
- macro accuracy
- macro ERR
- language-wise ERR
- TP / FP / FN / TN
- detection precision / recall / F1 based on changed-token decisions
- overnormalization rate
- undernormalization rate

## Report wording suggestion

> Due to the high cost of LLM inference, we generated predictions only once on the internal 17-language validation set. We then evaluated the same prediction file under three views using sentence-level language filtering: all 17 languages, the 12-language subset corresponding to the original official validation languages, and the 5-language held-out subset absent from the official validation set.
