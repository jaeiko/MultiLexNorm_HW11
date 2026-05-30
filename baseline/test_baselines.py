"""Baseline integration verification tests.

This script executes simple mock tests to ensure all three baseline models
(LAI, MFR, ByT5) compile, initialize, and perform predictions correctly
with basic social media slang token examples.
"""

from LAI_baseline import LAIBaseline
from MFR_baseline import MFRBaseline
from ByT5_baseline import ByT5Baseline


def run_baseline_tests() -> None:
    """Runs a series of tests to verify the correctness of the three baseline models."""
    sample_sentence = ["I", "lov", "u", "bcause", "ur", "cute"]
    print("=" * 50)
    print(f"Original Input Sentence: {sample_sentence}")
    print("=" * 50)

    # 1. LAI Baseline Test
    print("\n[1. Testing LAI Baseline...]")
    lai_model = LAIBaseline()
    lai_result = lai_model.predict(sample_sentence)
    print(f"-> LAI Result: {lai_result}")

    # 2. MFR Baseline Test
    print("\n[2. Testing MFR Baseline...]")
    mock_train_data = [
        {"raw": ["u", "r", "cute"], "norm": ["you", "are", "cute"]},
        {"raw": ["lov"], "norm": ["love"]},
        {"raw": ["bcause"], "norm": ["because"]},
        {"raw": ["ur"], "norm": ["you", "are"]}
    ]
    mfr_model = MFRBaseline(mock_train_data)
    mfr_result = mfr_model.predict(sample_sentence)
    print(f"-> MFR Result: {mfr_result}")

    # 3. ByT5 Baseline Test
    print("\n[3. Testing ByT5 Baseline...]")
    byt5_model = ByT5Baseline(model_checkpoint="google/byt5-small")
    byt5_result = byt5_model.predict(sample_sentence)
    print(f"-> ByT5 Result: {byt5_result}\n")


if __name__ == "__main__":
    run_baseline_tests()