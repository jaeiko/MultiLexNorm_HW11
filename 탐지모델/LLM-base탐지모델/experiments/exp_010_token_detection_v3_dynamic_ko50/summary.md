# LLM Token Detection Experiment Summary

- Experiment: `exp_010_token_detection_v3_dynamic_ko50`
- Model: `gemma4:26b`
- Prompt template: `prompts/prompt_v3.txt`
- Dataset: `../../multilexnorm2026-dataset/data/validation-00000-of-00001.parquet`
- Language: `ko`
- Limit: `None`
- Parse failures: `10`

## Token-Level Metrics

- Total rows: 212
- Total tokens: 1880
- TP: 134
- TN: 1145
- FP: 569
- FN: 32
- Accuracy: 0.6803
- Precision: 0.1906
- Recall: 0.8072
- F1: 0.3084
- Specificity: 0.6680
- Row exact accuracy: 0.4623

## Runtime

- Ollama eval tokens: 11166
- Elapsed time: 1940.3s (32.3min)
- Throughput: 5.8 tok/s
