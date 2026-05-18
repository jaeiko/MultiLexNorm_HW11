#!/usr/bin/env python3
"""Evaluate 5 ablation combos on validation set (9056 rows, internal_v1).

Combos (tri/mfr/xlmr_s5 on-off; xlmr-only and all-off excluded):
  - tri_only
  - mfr_only
  - tri_mfr
  - mfr_xlmr
  - tri_mfr_xlmr (= baseline)

Reports per-combo TP/FP/FN/TN/ERR + pairwise token agreement matrix.
"""

import os
import sys
import gzip
import pickle
import ast
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "mfr_first_package"))

XLMR_PATH = os.environ.get("XLMR_MODEL_PATH", str(_HERE.parent / "external" / "xlmr_finetuned_colab"))
DATA_DIR = Path(os.environ.get("DATA_DIR", str(_HERE.parent / "multilexnorm2026-dataset")))

from trigram_predictor import predict_trigram  # noqa
from smart_guard_mfr_v2 import predict_smart_guarded_mfr_v2, find_protected_indices, normalize_lang_code  # noqa
from evaluation import compute_metrics  # noqa


def to_list(v):
    if hasattr(v, 'tolist'):
        return [str(x) for x in v.tolist()]
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return [str(x) for x in v]


COMBOS = [
    ('tri_only',     True,  False, False),
    ('mfr_only',     False, True,  False),
    ('tri_mfr',      True,  True,  False),
    ('mfr_xlmr',     False, True,  True),
    ('tri_mfr_xlmr', True,  True,  True),
]


def decide(tok, i, l, mfrp, trp, xlmr_row, protected, mfr_stats, blocked,
           use_tri, use_mfr, use_xlmr):
    if use_tri and l not in blocked and trp[i] != tok:
        return trp[i]
    if use_mfr and mfrp[i] != tok:
        return mfrp[i]
    if use_xlmr and xlmr_row[i] == 1 and i not in protected:
        lc = normalize_lang_code(l)
        info = mfr_stats.get(lc, {}).get(tok)
        if info is not None:
            cands = info.get('candidates') or {}
            items = sorted(((c, n) for c, n in cands.items() if c != tok), key=lambda x: -x[1])
            if items:
                return items[0][0]
    return tok


def main():
    print("[eval_val_combos] 5 combos on validation set (internal_v1)")
    t0 = time.time()
    base = _HERE
    repo = base.parent

    val_path = DATA_DIR / "internal_v1" / "data" / "validation-00000-of-00001.parquet"
    df = pd.read_parquet(val_path)
    print(f"  val rows={len(df)}  cols={list(df.columns)}")

    with gzip.open(base / "outputs" / "trigram_stats_internal_v1.pkl.gz", "rb") as f:
        tri_stats = pickle.load(f)
    with gzip.open(repo / "mfr_first_package" / "mfr_stats.pkl.gz", "rb") as f:
        mfr_stats = pickle.load(f)

    samples = []
    raw_per_row, norm_per_row, lang_per_row = [], [], []
    for _, row in df.iterrows():
        raw = to_list(row['raw'])
        norm = to_list(row['norm'])
        lang = str(row['lang'])
        raw_per_row.append(raw); norm_per_row.append(norm); lang_per_row.append(lang)
        samples.append({'lang': lang, 'raw': raw})

    print("[XLMR inference] ...")
    tokz = AutoTokenizer.from_pretrained(XLMR_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForTokenClassification.from_pretrained(XLMR_PATH).to(device).eval()

    BATCH = 16; MAX_LEN = 512
    xlmr_preds = []
    with torch.no_grad():
        for i in range(0, len(raw_per_row), BATCH):
            batch = raw_per_row[i:i+BATCH]
            enc = tokz(batch, is_split_into_words=True, return_tensors='pt',
                       padding=True, truncation=True, max_length=MAX_LEN)
            ed = {k: v.to(device) for k, v in enc.items()}
            preds = model(**ed).logits.argmax(-1).cpu().tolist()
            for b_idx in range(len(batch)):
                wids = tokz(batch[b_idx:b_idx+1], is_split_into_words=True,
                            return_tensors='pt', truncation=True, max_length=MAX_LEN).word_ids(0)
                rp = preds[b_idx]
                wp = [None] * len(batch[b_idx])
                for ti, w in enumerate(wids):
                    if w is None or w >= len(batch[b_idx]): continue
                    if wp[w] is None: wp[w] = rp[ti]
                xlmr_preds.append([0 if p is None else int(p) for p in wp])
    del model; torch.cuda.empty_cache()
    print(f"  done {len(xlmr_preds)} rows")

    print("[Predictor stats] ...")
    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    # Build per-combo flat token streams
    combo_preds = {c[0]: [] for c in COMBOS}
    flat_raw, flat_norm = [], []
    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        gt = norm_per_row[row_idx]
        mfrp = mfr_preds[row_idx]; trp = tri_preds[row_idx]
        protected = set(find_protected_indices(raw))
        xrow = xlmr_preds[row_idx]
        for i, tok in enumerate(raw):
            flat_raw.append(tok)
            flat_norm.append(gt[i] if i < len(gt) else tok)
            for label, use_tri, use_mfr, use_xlmr in COMBOS:
                p = decide(tok, i, l, mfrp, trp, xrow, protected, mfr_stats, blocked,
                           use_tri, use_mfr, use_xlmr)
                combo_preds[label].append(p)

    n_total = len(flat_raw)
    print(f"\n[Metrics over {n_total:,} tokens]")
    print(f"  {'combo':<15} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>6}  {'Prec':>7} {'Rec':>7} {'F1':>7}  {'ERR':>8}  {'Δvs_full':>9}")
    print("  " + "-"*96)
    results = {}
    full_err = None
    for label, _, _, _ in COMBOS:
        m = compute_metrics(combo_preds[label], flat_norm, flat_raw)
        results[label] = m
        if label == 'tri_mfr_xlmr':
            full_err = m['err']
    for label, _, _, _ in COMBOS:
        m = results[label]
        diff = (m['err'] - full_err) if full_err is not None else 0.0
        sign = '+' if diff >= 0 else ''
        print(f"  {label:<15} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['tn']:>6}  "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}  {m['err']:>8.4f}  {sign}{diff:>+.4f}")

    # Pairwise token agreement
    labels = [c[0] for c in COMBOS]
    print(f"\n[Pairwise token agreement % (over {n_total:,} tokens)]")
    header = "  " + " " * 16 + "  ".join(f"{l:>13}" for l in labels)
    print(header)
    for a in labels:
        row = [f"  {a:<14}"]
        for b in labels:
            if a == b:
                row.append(f"{'-':>13}")
            else:
                agree = sum(1 for x, y in zip(combo_preds[a], combo_preds[b]) if x == y)
                row.append(f"{agree/n_total*100:>12.2f}%")
        print("  ".join(row))

    # Pairwise vs ground truth (correctness overlap)
    print(f"\n[Both-correct overlap % (agreed on the right answer)]")
    correct = {l: [p == g for p, g in zip(combo_preds[l], flat_norm)] for l in labels}
    print(header)
    for a in labels:
        row = [f"  {a:<14}"]
        for b in labels:
            if a == b:
                row.append(f"{sum(correct[a])/n_total*100:>12.2f}%")
            else:
                both = sum(1 for x, y in zip(correct[a], correct[b]) if x and y)
                row.append(f"{both/n_total*100:>12.2f}%")
        print("  ".join(row))

    print(f"\nTotal: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
