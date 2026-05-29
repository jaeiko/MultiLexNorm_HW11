"""Trigram ablation grid runner.

Sweeps (variant, conf_min, protect) over both:
  - tri-only mode  : pure trigram baseline (no MFR / XLM-R / LLM)
  - full mode      : trigram -> MFR fallback -> baseline; XLM-R-flagged hard cases
                     overlaid with cached LLM predictions

For each config it writes a temporary predictions.json, runs the v2 evaluator,
parses the resulting overall CSV (eval_groups all / official12 / missing5),
and appends one CSV row to outputs/ablation_trigram_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import itertools
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

import paths_config
paths_config.setup_imports()

from trigram_predictor import predict_trigram
from smart_guard_mfr_v2 import predict_smart_guarded_mfr_v2, find_protected_indices


VARIANTS = ['pure', 'tri_biL', 'tri_biR', 'tri_bi_both']
CONF_GRID = [round(0.50 + 0.05 * i, 2) for i in range(11)]  # 0.50 .. 1.00
PROTECT_MODES = ['non_protect', 'protect']

OFFICIAL_LANGS = "id,ja,ko,th,vi"
MISSING_LANGS = "de,en,hr,iden,nl,sl,sr"


def load_val_resources():
    val_parquet = paths_config.DATASET_DIR / "validation-00000-of-00001.parquet"
    df = pd.read_parquet(val_parquet)
    samples: List[Dict[str, Any]] = []
    raw_per_row: List[List[str]] = []
    lang_per_row: List[str] = []
    for _, row in df.iterrows():
        raw_tokens = [str(x) for x in row['raw']]
        lang_str = str(row['lang'])
        raw_per_row.append(raw_tokens)
        lang_per_row.append(lang_str)
        samples.append({'lang': lang_str, 'raw': raw_tokens})
    return samples, raw_per_row, lang_per_row


def load_stats():
    with gzip.open(paths_config.MFR_STATS_PATH, "rb") as f:
        mfr_stats = pickle.load(f)
    with gzip.open(paths_config.TRIGRAM_STATS_PATH, "rb") as f:
        tri_stats = pickle.load(f)
    return mfr_stats, tri_stats


def compute_xlmr_preds(raw_per_row, lang_per_row):
    import torch
    from detection import AnomalyDetector
    xlmr_path = str(paths_config.XLMR_MODEL_PATH)
    detector = AnomalyDetector(xlmr_path)
    preds = detector.predict_labels(raw_per_row, lang_per_row, threshold=0.5)
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def load_llm_cache() -> Dict[Tuple[int, int], str]:
    cache: Dict[Tuple[int, int], str] = {}
    path = paths_config.ROOT_DIR / "outputs" / "llm_corrections_val.jsonl"
    if not path.exists():
        return cache
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        norm = obj.get('normalized')
        if not isinstance(norm, str):
            continue
        cache[(int(obj['dev_row_idx']), int(obj['tok_idx']))] = norm
    return cache


def build_tri_only_predictions(tri_preds, raw_per_row, lang_per_row):
    """tri pred (if changed) else raw."""
    records = []
    for ri, raw in enumerate(raw_per_row):
        tri_row = tri_preds[ri]
        pred = [tri_row[i] if tri_row[i] != tok else tok for i, tok in enumerate(raw)]
        records.append({
            'raw': raw,
            'norm': [''] * len(raw),
            'lang': lang_per_row[ri],
            'pred': pred,
        })
    return records


def build_full_predictions(
    tri_preds, mfr_preds, xlmr_preds, raw_per_row, lang_per_row, llm_cache,
):
    """tri -> mfr fallback baseline; XLM-R-flagged hard cases overlaid with LLM cache (strong protect)."""
    records = []
    for ri, raw in enumerate(raw_per_row):
        tri_row = tri_preds[ri]
        mfr_row = mfr_preds[ri]
        xlmr_row = xlmr_preds[ri]
        protected = set(find_protected_indices(raw))
        pred = []
        for i, tok in enumerate(raw):
            if tri_row[i] != tok:
                baseline_pred = tri_row[i]
            elif mfr_row[i] != tok:
                baseline_pred = mfr_row[i]
            else:
                baseline_pred = tok
            # Apply LLM overlay only on hard cases (XLMR=1, not strong-protected, baseline==raw)
            if (
                baseline_pred == tok
                and xlmr_row[i] == 1
                and i not in protected
            ):
                llm_norm = llm_cache.get((ri, i))
                if llm_norm is not None and llm_norm != tok:
                    pred.append(llm_norm)
                    continue
            pred.append(baseline_pred)
        records.append({
            'raw': raw,
            'norm': [''] * len(raw),
            'lang': lang_per_row[ri],
            'pred': pred,
        })
    return records


def run_evaluator(predictions_path: Path, model_name: str, out_dir: Path) -> Dict[str, Dict[str, float]]:
    """Calls v2 evaluator, parses eval_groups_overall.csv, returns {group: {err: ..., macro_err: ...}}."""
    cmd = [
        sys.executable,
        str(paths_config.ROOT_DIR / "multilexnorm_eval_package_v2" / "multilexnorm_evaluator_v2.py"),
        "evaluate",
        "--gold_parquet", str(paths_config.DATASET_DIR / "validation-00000-of-00001.parquet"),
        "--pred_path", str(predictions_path),
        "--model_name", model_name,
        "--dataset_name", "abl",
        "--eval_groups", "all", "official12", "missing5",
        "--official_langs", OFFICIAL_LANGS,
        "--missing_langs", MISSING_LANGS,
        "--out_dir", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  Evaluator failed: {res.stderr[-500:]}")
        return {}
    overall_csv = out_dir / f"abl_{model_name}_eval_groups_overall.csv"
    out: Dict[str, Dict[str, float]] = {}
    with open(overall_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row['eval_group']
            out[group] = {
                'err': float(row['err']),
                'macro_err': float(row['macro_err']),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Trigram ablation grid runner.")
    ap.add_argument('--mode', choices=['tri-only', 'full', 'both'], default='both')
    ap.add_argument('--output', default='outputs/ablation_trigram_results.csv')
    args = ap.parse_args()

    t0 = time.time()
    print("[ablation] Loading validation resources...")
    samples, raw_per_row, lang_per_row = load_val_resources()
    print(f"  rows={len(samples)} tokens={sum(len(r) for r in raw_per_row):,}")

    print("[ablation] Loading MFR + trigram stats...")
    mfr_stats, tri_stats = load_stats()

    mfr_preds = None
    xlmr_preds = None
    llm_cache: Dict[Tuple[int, int], str] = {}
    if args.mode in ('full', 'both'):
        print("[ablation] Computing MFR predictions (once)...")
        mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
        print("[ablation] Computing XLM-R predictions (once)...")
        xlmr_preds = compute_xlmr_preds(raw_per_row, lang_per_row)
        print("[ablation] Loading LLM cache...")
        llm_cache = load_llm_cache()
        print(f"  LLM cache entries: {len(llm_cache)}")

    out_csv_path = paths_config.ROOT_DIR / args.output
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="abl_trigram_"))
    eval_dir = tmp_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        'mode', 'variant', 'conf_min', 'protect',
        'err_all', 'macro_err_all',
        'err_new5', 'macro_err_new5',
        'err_orig7', 'macro_err_orig7',
        'err_weighted',
    ]
    fout = open(out_csv_path, 'w', encoding='utf-8', newline='')
    writer = csv.DictWriter(fout, fieldnames=fields)
    writer.writeheader()
    fout.flush()

    combos = list(itertools.product(VARIANTS, CONF_GRID, PROTECT_MODES))
    modes_to_run = ['tri-only', 'full'] if args.mode == 'both' else [args.mode]
    total = len(combos) * len(modes_to_run)
    done = 0
    t_loop = time.time()

    for variant in VARIANTS:
        for conf_min in CONF_GRID:
            for protect in PROTECT_MODES:
                cfg = {'variant': variant, 'conf_min': conf_min, 'protect': protect}
                tri_preds, _ = predict_trigram(samples, tri_stats, cfg)

                for mode in modes_to_run:
                    if mode == 'tri-only':
                        records = build_tri_only_predictions(tri_preds, raw_per_row, lang_per_row)
                    else:
                        records = build_full_predictions(
                            tri_preds, mfr_preds, xlmr_preds,
                            raw_per_row, lang_per_row, llm_cache,
                        )

                    pred_path = tmp_dir / "predictions.json"
                    with open(pred_path, 'w', encoding='utf-8') as f:
                        json.dump(records, f, ensure_ascii=False)

                    model_name = f"{mode}_{variant}_c{conf_min}_p{protect}".replace('.', '')
                    metrics = run_evaluator(pred_path, model_name, eval_dir)

                    if metrics:
                        err_new5 = metrics.get('official12', {}).get('err', float('nan'))
                        err_orig7 = metrics.get('missing5', {}).get('err', float('nan'))
                        err_weighted = (err_new5 + err_orig7) / 2.0
                        writer.writerow({
                            'mode': mode,
                            'variant': variant,
                            'conf_min': conf_min,
                            'protect': protect,
                            'err_all': metrics.get('all', {}).get('err'),
                            'macro_err_all': metrics.get('all', {}).get('macro_err'),
                            'err_new5': err_new5,
                            'macro_err_new5': metrics.get('official12', {}).get('macro_err'),
                            'err_orig7': err_orig7,
                            'macro_err_orig7': metrics.get('missing5', {}).get('macro_err'),
                            'err_weighted': err_weighted,
                        })
                        fout.flush()

                    done += 1
                    elapsed = time.time() - t_loop
                    eta = elapsed / done * (total - done)
                    print(
                        f"  [{done}/{total}] {mode:8s} variant={variant:<13s} "
                        f"conf={conf_min:.2f} protect={protect:<6s} "
                        f"err_all={metrics.get('all', {}).get('err', float('nan')):.4f} "
                        f"ETA={eta/60:.1f}min"
                    )

    fout.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n[ablation] Done. Total: {done} runs in {(time.time()-t0)/60:.1f}min")
    print(f"  Results -> {out_csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
