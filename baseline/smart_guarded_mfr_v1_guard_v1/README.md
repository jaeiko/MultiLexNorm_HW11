# Smart Guarded MFR v1 guard_v1

Validation package for MultiLexNorm2026.

## Model
- Base: train-based `MFR_language_aware`
- Guard: Smart Guard v1 language-specific rules
- Guard signals: confidence, change_rate, margin
- Dictionary source: train split only

## Validation Result
- Overall ERR: 46.12
- Macro ERR: 40.60
- Accuracy: 94.09

## Files
- `mfr_stats.pkl.gz`: train-based MFR dictionary and extended statistics
- `smart_guard_mfr_v1.py`: prediction function
- `config.json`: guard_v1 hyperparameters
- `validation_predictions.json`: validation predictions
- `validation_overall.json`: validation overall result
- `validation_language_metrics.csv`: language-wise metrics

## Usage
```python
import gzip, pickle
from smart_guard_mfr_v1 import predict_smart_guarded_mfr_v1

with gzip.open("mfr_stats.pkl.gz", "rb") as f:
    obj = pickle.load(f)

mfr = obj["mfr"]
stats = obj["stats"]
predictions = predict_smart_guarded_mfr_v1(test_samples, mfr, stats)
```
