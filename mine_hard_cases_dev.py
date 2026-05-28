"""Hard Cases Mining Pipeline Stage for LLM Normalization.

This script parses the internal 17-language validation dataset, computes Phase 1
(MFR Dictionary) and optionally Phase 0 (Trigram) baseline outputs, classifies
word standardness using Phase 2 (XLM-R Token Classifier), identifies candidates
that are standard-looking to baselines but flagged as noisy by XLM-R, and saves
them as hard cases for Stage 3 (LLM).

Usage:
    python mine_hard_cases_dev.py              # Pipeline 2: Trigram + MFR baseline
    python mine_hard_cases_dev.py --no-trigram # Pipeline 1: MFR-only baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import gc
import gzip
import pickle
import json
import time
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Set

import pandas as pd
import torch

# Centralized path imports configuration
import paths_config
paths_config.setup_imports()

from trigram_predictor import predict_trigram
from smart_guard_mfr_v2 import predict_smart_guarded_mfr_v2, find_protected_indices, normalize_lang_code
from detection import AnomalyDetector


def mine_hard_cases(args: argparse.Namespace) -> int:
    """Performs hard cases mining from 17-language internal validation set.

    Returns:
        int: System exit code status (0 for success).
    """
    use_trigram: bool = args.use_trigram
    use_mfr: bool = args.use_mfr
    mfr_first: bool = args.mfr_first
    if not use_trigram and not use_mfr:
        print("ERROR: at least one baseline must be enabled (cannot pass both --no-trigram and --no-mfr).")
        return 1
    parts = []
    if mfr_first:
        if use_mfr: parts.append("mfr")
        if use_trigram: parts.append("tri")
    else:
        if use_trigram: parts.append("tri")
        if use_mfr: parts.append("mfr")
    tag = "+".join(parts)
    print(f"[mine val] Starting validation hard case mining — baseline={tag} ...")
    t0 = time.time()

    xlmr_path: str = os.environ.get("XLMR_MODEL_PATH", str(paths_config.XLMR_MODEL_PATH))
    xlmr_threshold_str: str | None = os.environ.get("XLMR_THRESHOLD")

    # 1. Load raw tokens and language arrays from validation parquet
    val_parquet_path: Path = paths_config.DATASET_DIR / "validation-00000-of-00001.parquet"
    if not val_parquet_path.exists():
        print(f"  ERROR: Validation parquet missing at {val_parquet_path}")
        return 1

    df = pd.read_parquet(val_parquet_path)
    print(f"  Loaded validation rows: {len(df)}")

    raw_per_row: List[List[str]] = []
    lang_per_row: List[str] = []
    samples: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_tokens = [str(x) for x in row['raw']]
        lang_str = str(row['lang'])
        raw_per_row.append(raw_tokens)
        lang_per_row.append(lang_str)
        samples.append({'lang': lang_str, 'raw': raw_tokens})

    print(f"  Total tokens across validation set: {sum(len(r) for r in raw_per_row):,}")

    # 2. Load compiled statistical frequency dictionary resources
    print(f"  Loading MFR Statistics: {paths_config.MFR_STATS_PATH.name}")
    with gzip.open(paths_config.MFR_STATS_PATH, "rb") as f:
        mfr_stats: Dict[str, Dict[str, Any]] = pickle.load(f)

    tri_stats: Dict[str, Any] | None = None
    if use_trigram:
        print(f"  Loading Trigram Statistics: {paths_config.TRIGRAM_STATS_PATH.name}")
        with gzip.open(paths_config.TRIGRAM_STATS_PATH, "rb") as f:
            tri_stats = pickle.load(f)

    # 3. Perform XLM-R Token classification inference
    threshold: float = 0.5 if xlmr_threshold_str is None else float(xlmr_threshold_str)
    print(f"[XLMR detection] Initializing detector with threshold={threshold} ...")

    detector = AnomalyDetector(xlmr_path)
    xlmr_preds: List[List[int]] = detector.predict_labels(raw_per_row, lang_per_row, threshold=threshold)

    # Direct CUDA memory cleanup and release context
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  XLM-R token classification phase complete. GPU context released.")

    # 4. Perform MFR and (optionally) Trigram corrections
    mfr_preds: List[List[str]] | None = None
    if use_mfr:
        print("  Calculating MFR baseline predictions...")
        mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)

    tri_preds: List[List[str]] | None = None
    if use_trigram:
        print("  Calculating Trigram baseline predictions...")
        tri_preds, _ = predict_trigram(samples, tri_stats, {
            'variant': 'tri_bi_both', 'conf_min': 0.70, 'protect': 'none'
        })

    # 5. Extract hard cases where baseline remains unchanged but XLM-R flags error
    hard_records: List[Dict[str, Any]] = []
    baseline_per_row: List[List[str]] = []
    cats_counter: Counter = Counter()

    for row_idx, raw in enumerate(raw_per_row):
        lang = lang_per_row[row_idx]
        mfr_row = mfr_preds[row_idx] if use_mfr else None
        xlmr_row = xlmr_preds[row_idx]
        tri_row = tri_preds[row_idx] if use_trigram else None
        protected_indices: Set[int] = set(find_protected_indices(raw))

        baseline_row: List[str] = []
        for i, tok in enumerate(raw):
            # Priority order: --mfr-first reverses default Trigram -> MFR -> Leave-As-Is
            if mfr_first:
                if use_mfr and mfr_row[i] != tok:
                    baseline_pred = mfr_row[i]
                elif use_trigram and tri_row[i] != tok:
                    baseline_pred = tri_row[i]
                else:
                    baseline_pred = tok
            elif use_trigram and tri_row[i] != tok:
                baseline_pred = tri_row[i]
            elif use_mfr and mfr_row[i] != tok:
                baseline_pred = mfr_row[i]
            else:
                baseline_pred = tok
            baseline_row.append(baseline_pred)

            # Filter candidates for hard cases gating
            if xlmr_row[i] != 1:  # Not flagged by XLM-R
                continue
            if i in protected_indices:  # Emoji, URLs, or hashtag protected tokens
                continue
            if baseline_pred != tok:  # Baseline has already resolved/modified the token
                continue

            # Categorize the OOV frequency status for analytics
            normalized_lang = normalize_lang_code(lang)
            info = mfr_stats.get(normalized_lang, {}).get(tok)
            if info is None:
                cat = 'mfr_oov'
            else:
                candidates_dict = info.get('candidates') or {}
                non_raw = [(c, n) for c, n in candidates_dict.items() if c != tok]
                cat = 'all_cands_raw' if not non_raw else 'has_non_raw_cands'
            cats_counter[cat] += 1

            top5 = []
            if info is not None:
                top5 = sorted(info.get('candidates', {}).items(), key=lambda x: -x[1])[:5]

            prev_tok = raw[i - 1] if i > 0 else '<BOS>'
            next_tok = raw[i + 1] if i < len(raw) - 1 else '<EOS>'

            hard_records.append({
                'dev_row_idx': row_idx,
                'tok_idx': i,
                'lang': lang,
                'token': tok,
                'gt': '',
                'cat': cat,
                'raw_sentence': raw,
                'prev': prev_tok,
                'next': next_tok,
                'mfr_top5': top5,
            })
        baseline_per_row.append(baseline_row)

    # 6. Save Stage 2 input files (hard cases JSONL)
    hc_path: Path = paths_config.HARD_CASES_PATH
    hc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hc_path, "w", encoding="utf-8") as f:
        for record in hard_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 7. Save Stage 3 input files (entire baseline predictions)
    baseline_path: Path = paths_config.BASELINE_PATH
    baseline_out = [
        {
            'raw': raw_per_row[ri],
            'lang': lang_per_row[ri],
            'pred': baseline_per_row[ri]
        }
        for ri in range(len(raw_per_row))
    ]
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_out, f, ensure_ascii=False)

    print(f"\n[Hard cases extracted]")
    print(f"  Total hard cases mined: {len(hard_records):,}")
    print(f"\n  Category distribution:")
    for category_name, count in sorted(cats_counter.items(), key=lambda x: -x[1]):
        print(f"    {category_name:<20} {count}")
    print(f"\nSuccessfully generated outputs:")
    print(f"  -> {hc_path}")
    print(f"  -> {baseline_path} ({len(baseline_out)} rows)")
    print(f"  Total extraction execution completed in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mine hard cases from 12-lang validation set for LLM correction."
    )
    parser.add_argument(
        '--no-trigram', dest='use_trigram', action='store_false', default=True,
        help="Disable trigram baseline step."
    )
    parser.add_argument(
        '--no-mfr', dest='use_mfr', action='store_false', default=True,
        help="Disable MFR baseline step."
    )
    parser.add_argument(
        '--mfr-first', dest='mfr_first', action='store_true', default=False,
        help="Apply MFR before trigram (default: trigram first)."
    )
    sys.exit(mine_hard_cases(parser.parse_args()))
