"""MFR and XLM-R Pipeline Validation Experiment Runner.

This script executes a complete evaluation run of Phase 1 (MFR Dictionary)
and Phase 2 (XLM-R Detection) across both Official (12lang) and Internal (17lang)
validation splits. It outputs the predictions to CodaBench-compatible JSON files.
"""

from __future__ import annotations

import os
import gzip
import pickle
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import pandas as pd

# Standardized path initialization
import paths_config
paths_config.setup_imports()

from smart_guard_mfr_v2 import predict_smart_guarded_mfr_v2
from detection import AnomalyDetector


def run_pipeline_experiment() -> None:
    """Executes the pipeline experiment over Official and Internal datasets.

    Loads the stats, initializes the token classification model, processes the
    validation datasets using MFR + XLM-R combinations, and outputs prediction JSONs.
    """
    t0 = time.time()
    xlmr_path = os.environ.get("XLMR_MODEL_PATH", str(paths_config.ROOT_DIR.parent / "xlmr_finetuned_colab"))
    
    print("[Experiment] Initiating Language-Aware MFR + XLM-R Pipeline Validation Experiment.")
    
    # 1. Load Pre-compiled statistical frequency resources
    print(f"  Loading pre-compiled MFR frequencies: {paths_config.MFR_STATS_PATH}")
    with gzip.open(paths_config.MFR_STATS_PATH, "rb") as f:
        mfr_stats: Dict[str, Dict[str, Any]] = pickle.load(f)
        
    print(f"  Initializing XLM-R Token Classifier: {xlmr_path}")
    detector = AnomalyDetector(xlmr_path)
    
    # 2. Target validation datasets configuration
    datasets: Dict[str, Path] = {
        "official_val": paths_config.DATASET_12LANG / "validation-00000-of-00001.parquet",
    }
    
    for name, path in datasets.items():
        if not path.exists():
            print(f"  [Warning] Dataset split missing, skipping: {path}")
            continue
            
        print(f"\nProcessing dataset split: {name} ({path.name})")
        df = pd.read_parquet(path)
        
        # Format tokens and languages lists
        raw_per_row: List[List[str]] = [list(row['raw']) for _, row in df.iterrows()]
        lang_per_row: List[str] = [str(row['lang']) for _, row in df.iterrows()]
        samples: List[Dict[str, Any]] = [{'lang': l, 'raw': r} for l, r in zip(lang_per_row, raw_per_row)]
        
        # Step A: Perform Language-Aware MFR Lookup predictions
        mfr_preds: List[List[str]] = predict_smart_guarded_mfr_v2(samples, mfr_stats)
        
        # Step B: Perform XLM-R Token classification (0=standard, 1=noise)
        xlmr_preds: List[List[int]] = detector.predict_labels(raw_per_row, lang_per_row)
        
        # Step C: Gated integration: apply MFR dictionary only on XLM-R flagged items
        final_sentences: List[List[str]] = []
        for row_idx in range(len(raw_per_row)):
            raw_sent = raw_per_row[row_idx]
            mfr_sent = mfr_preds[row_idx]
            xlmr_sent = xlmr_preds[row_idx]
            
            pred_sent: List[str] = []
            for i in range(len(raw_sent)):
                if xlmr_sent[i] == 1:
                    pred_sent.append(mfr_sent[i])  # Normalize if classified as noise
                else:
                    pred_sent.append(raw_sent[i])  # Preserve standard/clean tokens
            final_sentences.append(pred_sent)
            
        # 3. Export predictions to file
        out_path = paths_config.ROOT_DIR / f"pred_mfr_xlmr_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_sentences, f, ensure_ascii=False)
        print(f"  Successfully exported predictions -> {out_path}")

    # Explicit memory cleanup
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[Experiment] Completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    run_pipeline_experiment()