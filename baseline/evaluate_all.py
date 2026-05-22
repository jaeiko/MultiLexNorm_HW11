"""Unified baseline evaluation dashboard.

This script loads the designated validation split (Official or Internal),
initializes all three baseline models (LAI, MFR, and ByT5), performs token-level
predictions, and outputs a comparative table comparing their evaluation metrics
(Precision, Recall, F1-score, ERR).
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# 1. Standardized project-level imports using centralized paths_config
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths_config
paths_config.setup_imports()

from evaluation import compute_metrics, format_metrics_table, _to_tokens
from LAI_baseline import LAIBaseline
from MFR_baseline import MFRBaseline
from ByT5_baseline import ByT5Baseline


def evaluate_model(
    model: Any,
    raw_sentences: List[List[str]],
    gold_sentences: List[List[str]],
) -> Dict[str, Any]:
    """Evaluates a single corrector model over a list of sentences.

    Args:
        model: Model instance implementing a `.predict(List[str]) -> List[str]` method.
        raw_sentences: List of raw input token sentences.
        gold_sentences: List of gold standard normalized token sentences.

    Returns:
        Dict[str, Any]: Calculated metrics dictionary (TP, FP, FN, TN, precision, recall, f1, err).
    """
    flat_raw = []
    flat_gold = []
    flat_pred = []

    for raw, gold in zip(raw_sentences, gold_sentences):
        # Run prediction on the model
        pred = model.predict(raw)
        
        # Token-level alignment sanity check
        if len(pred) != len(raw):
            aligned_pred = pred[:len(raw)]
            if len(aligned_pred) < len(raw):
                aligned_pred.extend(raw[len(aligned_pred):])
            pred = aligned_pred

        flat_raw.extend(raw)
        flat_gold.extend(gold)
        flat_pred.extend(pred)

    return compute_metrics(flat_pred, flat_gold, flat_raw)


def main() -> None:
    """Runs the unified baseline evaluation dashboard."""
    parser = argparse.ArgumentParser(
        description="Unified baseline evaluation dashboard for MultiLexNorm2026."
    )
    parser.add_argument(
        "--dataset", "-d", choices=["official", "internal"], default="official",
        help="Select validation split to evaluate on (default: official)"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=100,
        help="Limit number of sentences to evaluate to ensure fast execution (default: 100, use -1 for all)"
    )
    args = parser.parse_args()

    # 1. Resolve Dataset Paths
    if args.dataset == "official":
        val_path = paths_config.DATASET_12LANG / "validation-00000-of-00001.parquet"
        train_path = paths_config.DATASET_12LANG / "train-00000-of-00001.parquet"
    else:
        val_path = paths_config.DATASET_17LANG / "data" / "validation-00000-of-00001.parquet"
        train_path = paths_config.DATASET_17LANG / "data" / "train-00000-of-00001.parquet"

    print(f"\n[Dashboard] Starting Unified Baseline Evaluation")
    print(f"  Validation Dataset: {val_path}")
    print(f"  Train Dataset:      {train_path}")
    
    if not val_path.exists():
        print(f"ERROR: Validation file does not exist at {val_path}")
        return

    # 2. Load Validation Dataset
    print("\nLoading validation dataset...")
    df_val = pd.read_parquet(val_path)
    
    if args.limit > 0:
        df_val = df_val.head(args.limit)
        print(f"  Limited evaluation to first {args.limit} sentences.")
        
    raw_sentences = [_to_tokens(row) for row in df_val['raw']]
    gold_sentences = [_to_tokens(row) for row in df_val['norm']]
    print(f"  Loaded {len(raw_sentences)} sentences with {sum(len(s) for s in raw_sentences)} tokens.")

    results = {}

    # 3. Evaluate LAI Baseline (Always works, instant)
    print("\nInitializing LAI Baseline...")
    try:
        lai_model = LAIBaseline()
        t0 = time.time()
        results["LAIBaseline"] = evaluate_model(lai_model, raw_sentences, gold_sentences)
        print(f"  LAI Baseline evaluation completed in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"  [Error] LAI Baseline failed: {e}")

    # 4. Evaluate MFR Baseline (Requires building count statistics from Train parquet)
    print("\nInitializing MFR Baseline...")
    try:
        if not train_path.exists():
            print(f"  [Warning] Train parquet missing at {train_path}. Using empty mock training data.")
            train_data = []
        else:
            print("  Reading training dataset for MFR counting...")
            df_train = pd.read_parquet(train_path)
            # Standardize training dataset into dictionaries
            train_data = []
            for _, row in df_train.iterrows():
                train_data.append({
                    "raw": _to_tokens(row['raw']),
                    "norm": _to_tokens(row['norm'])
                })
            print(f"  Loaded {len(train_data)} training sentences for MFR compilation.")

        t0 = time.time()
        mfr_model = MFRBaseline(train_data)
        results["MFRBaseline"] = evaluate_model(mfr_model, raw_sentences, gold_sentences)
        print(f"  MFR Baseline evaluation completed in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"  [Error] MFR Baseline failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Evaluate ByT5 Baseline (Requires PyTorch & HuggingFace, might download model weights)
    print("\nInitializing ByT5 Baseline...")
    try:
        t0 = time.time()
        byt5_model = ByT5Baseline(model_checkpoint="google/byt5-small")
        results["ByT5Baseline"] = evaluate_model(byt5_model, raw_sentences, gold_sentences)
        print(f"  ByT5 Baseline evaluation completed in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"  [Error] ByT5 Baseline failed (skipping): {e}")
        print("  (This is normal if offline or resource-constrained during deep learning model load)")

    # 6. Output Dashboards and Tables
    if results:
        print("\n" + "="*80)
        print(f"       COMPARATIVE BASELINE PERFORMANCE: {args.dataset.upper()} VAL (Limit={args.limit})")
        print("="*80)
        format_metrics_table(results)
        print("="*80)
    else:
        print("\nERROR: No models were successfully evaluated.")


if __name__ == "__main__":
    main()
