# Smart Guarded MFR v2 pth 0.8

Current best validation model for MultiLexNorm2026.

## Model
- Base: train-based MFR_language_aware
- Guard: Smart Guard v2 for protected tokens
- Best threshold: protected_threshold = 0.8

## Validation Result
- Overall ERR: 49.19
- Macro ERR: 43.81
- Accuracy: 94.15

## Files
- mfr_stats.pkl.gz: train-based MFR statistics
- smart_guard_mfr_v2.py: prediction function
- config.json: best hyperparameters
- validation_predictions.json: validation predictions
- validation_overall.json: validation overall result
- validation_language_metrics.csv: language-wise metrics
- test_predictions.json: optional test predictions if generated

## Usage
1. Mount Google Drive.
2. Add common_prompt_v2_package to sys.path if available.
3. Load mfr_stats.pkl.gz.
4. Import predict_smart_guarded_mfr_v2 from smart_guard_mfr_v2.py.
5. Run predictions on validation/test samples.
