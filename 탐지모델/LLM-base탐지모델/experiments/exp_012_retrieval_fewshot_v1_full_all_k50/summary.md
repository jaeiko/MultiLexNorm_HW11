# LLM Detection Experiment Summary

- Experiment: `exp_012_retrieval_fewshot_v1_full_all_k50`
- Model: `gemma4:26b`
- Prompt template: `prompts/prompt_v1.txt`
- Dataset: `../../multilexnorm2026-dataset/data/validation-00000-of-00001.parquet`
- Language: `all`
- Limit: `None`
- Parse failures: `0`
- Merge method: `language-specific k50 runs sorted by original row_index`

## Metrics

- Total: 8408
- TP: 5145
- TN: 2041
- FP: 805
- FN: 417
- Accuracy: 0.8547
- Precision: 0.8647
- Recall: 0.9250
- F1: 0.8938
- Specificity: 0.7171

## Runtime

- Total output tokens: 214798
- Elapsed time: 82775.1s (1379.6min)
- Throughput: 2.6 tok/s

## Source Languages

- da: rows=0, parse_failures=0, elapsed_seconds=0.0
- de: rows=573, parse_failures=0, elapsed_seconds=3302.6
- en: rows=590, parse_failures=0, elapsed_seconds=5852.6
- es: rows=0, parse_failures=0, elapsed_seconds=0.0
- hr: rows=1588, parse_failures=0, elapsed_seconds=14881.3
- id: rows=431, parse_failures=0, elapsed_seconds=2730.8
- iden: rows=165, parse_failures=0, elapsed_seconds=2119.6
- it: rows=0, parse_failures=0, elapsed_seconds=0.0
- ja: rows=305, parse_failures=0, elapsed_seconds=3006.3
- ko: rows=212, parse_failures=0, elapsed_seconds=1377.0
- nl: rows=308, parse_failures=0, elapsed_seconds=1718.9
- sl: rows=1557, parse_failures=0, elapsed_seconds=12683.5
- sr: rows=1379, parse_failures=0, elapsed_seconds=13570.1
- th: rows=250, parse_failures=0, elapsed_seconds=14164.0
- tr: rows=0, parse_failures=0, elapsed_seconds=0.0
- trde: rows=0, parse_failures=0, elapsed_seconds=0.0
- vi: rows=1050, parse_failures=0, elapsed_seconds=7368.4

- Skipped empty languages: `da, es, it, tr, trde`
