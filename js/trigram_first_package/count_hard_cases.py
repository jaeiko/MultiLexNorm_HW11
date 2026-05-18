#!/usr/bin/env python3
"""Count production-realistic hard cases on any dataset (validation or test).

Usage:
  python count_hard_cases.py --dataset val_full
  python count_hard_cases.py --dataset test
  python count_hard_cases.py --dataset val_mini
"""

import os
import argparse
import sys
import gzip
import pickle
import ast
import time
from pathlib import Path
from collections import Counter

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


def to_list(v):
    if hasattr(v, 'tolist'):
        return [str(x) for x in v.tolist()]
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return [str(x) for x in v]


DATASETS = {
    'val_mini':  'mini_validation/validation_mini_17langs_1of8_seed42.parquet',
    'val_full':  'internal_v1/data/validation-00000-of-00001.parquet',
    'test':      'internal_v1/data/test-00000-of-00001.parquet',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASETS.keys()), required=True)
    args = ap.parse_args()
    base = _HERE
    repo = base.parent
    path = DATA_DIR / DATASETS[args.dataset]
    print(f"[count_hard_cases] dataset={args.dataset} path={path.name}")
    t0 = time.time()

    df = pd.read_parquet(path)
    print(f"  rows={len(df)}")

    with gzip.open(base / "outputs" / "trigram_stats_internal_v1.pkl.gz", "rb") as f:
        tri_stats = pickle.load(f)
    with gzip.open(repo / "mfr_first_package" / "mfr_stats.pkl.gz", "rb") as f:
        mfr_stats = pickle.load(f)

    # XLMR
    print("[XLMR inference] ...")
    tokz = AutoTokenizer.from_pretrained(XLMR_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForTokenClassification.from_pretrained(XLMR_PATH).to(device).eval()

    samples = []
    raw_per_row, lang_per_row, norm_per_row = [], [], []
    for _, row in df.iterrows():
        raw = to_list(row['raw']); norm = to_list(row['norm']) if 'norm' in df.columns else raw[:]
        lang = str(row['lang'])
        raw_per_row.append(raw); norm_per_row.append(norm); lang_per_row.append(lang)
        samples.append({'lang': lang, 'raw': raw})

    BATCH = 16; MAX_LEN = 512
    xlmr_preds = []
    with torch.no_grad():
        for i in range(0, len(raw_per_row), BATCH):
            batch = raw_per_row[i:i+BATCH]
            enc = tokz(batch, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
            ed = {k: v.to(device) for k, v in enc.items()}
            preds = model(**ed).logits.argmax(-1).cpu().tolist()
            for b_idx in range(len(batch)):
                wids = tokz(batch[b_idx:b_idx+1], is_split_into_words=True, return_tensors='pt', truncation=True, max_length=MAX_LEN).word_ids(0)
                rp = preds[b_idx]
                wp = [None] * len(batch[b_idx])
                for ti, w in enumerate(wids):
                    if w is None or w >= len(batch[b_idx]): continue
                    if wp[w] is None: wp[w] = rp[ti]
                xlmr_preds.append([0 if p is None else int(p) for p in wp])
    del model; torch.cuda.empty_cache()
    print(f"  done {len(xlmr_preds)} rows")

    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    n_total = 0
    n_hard = 0
    n_normal = 0
    n_nonstd = 0
    per_lang = Counter()

    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        norm = norm_per_row[row_idx]
        mfrp = mfr_preds[row_idx]; trp = tri_preds[row_idx]
        protected = set(find_protected_indices(raw))
        xlmr_row = xlmr_preds[row_idx]
        for i, tok in enumerate(raw):
            n_total += 1
            gt = norm[i] if i < len(norm) else tok
            # baseline
            use_t = l not in blocked and trp[i] != tok
            if use_t:
                bp = trp[i]
            elif mfrp[i] != tok:
                bp = mfrp[i]
            elif xlmr_row[i] == 1 and i not in protected:
                lc = normalize_lang_code(l)
                info = mfr_stats.get(lc, {}).get(tok)
                cand = None
                if info is not None:
                    cands = info.get('candidates') or {}
                    items = sorted(((c, n) for c, n in cands.items() if c != tok), key=lambda x: -x[1])
                    if items:
                        cand = items[0][0]
                bp = cand if cand is not None else tok
            else:
                bp = tok

            # production-realistic hard case
            if bp != tok: continue
            if xlmr_row[i] != 1: continue
            if i in protected: continue
            n_hard += 1
            per_lang[l] += 1
            if tok == gt:
                n_normal += 1
            else:
                n_nonstd += 1

    print(f"\n[Hard case count] dataset={args.dataset}")
    print(f"  total tokens     : {n_total:,}")
    print(f"  hard cases       : {n_hard:,}  ({n_hard/n_total*100:.2f}%)")
    if 'norm' in df.columns:
        print(f"    normal (raw==gt): {n_normal:,}  ({n_normal/n_hard*100:.1f}% of hard)")
        print(f"    nonstd (raw!=gt): {n_nonstd:,}  ({n_nonstd/n_hard*100:.1f}% of hard)")
    print(f"\n  per-lang:")
    for l, c in sorted(per_lang.items(), key=lambda x: -x[1]):
        print(f"    {l:<6} {c}")

    print(f"\n  Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
