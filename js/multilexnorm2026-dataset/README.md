---
dataset_info:
  features:
  - name: raw
    sequence: string
  - name: norm
    sequence: string
  - name: lang
    dtype: string
  splits:
  - name: train
    num_bytes: 13157263
    num_examples: 39178
  - name: validation
    num_bytes: 2489517
    num_examples: 8408
  - name: test
    num_bytes: 2853877
    num_examples: 11956
  download_size: 7270705
  dataset_size: 18500657
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
