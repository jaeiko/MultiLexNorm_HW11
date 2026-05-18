#!/usr/bin/env python3
"""Run baseline pipeline (Tri override + MFR + S5 top-2 fallback) on test set
and write predictions to test_pred.dev (parquet content, .dev extension)."""

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


def to_list(v):
    if hasattr(v, 'tolist'):
        return [str(x) for x in v.tolist()]
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return [str(x) for x in v]


def main():
    print("[Predict test, no LLM] baseline = Tri override + MFR + S5 top-2 fallback")
    t0 = time.time()
    base = _HERE
    repo = base.parent

    test_path = DATA_DIR / "internal_v1" / "data" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(test_path)
    print(f"  test rows={len(df)}")

    with gzip.open(base / "outputs" / "trigram_stats_internal_v1.pkl.gz", "rb") as f:
        tri_stats = pickle.load(f)
    with gzip.open(repo / "mfr_first_package" / "mfr_stats.pkl.gz", "rb") as f:
        mfr_stats = pickle.load(f)

    print("[XLMR inference] ...")
    tokz = AutoTokenizer.from_pretrained(XLMR_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForTokenClassification.from_pretrained(XLMR_PATH).to(device).eval()

    samples = []
    raw_per_row, lang_per_row = [], []
    for _, row in df.iterrows():
        raw = to_list(row['raw'])
        lang = str(row['lang'])
        raw_per_row.append(raw)
        lang_per_row.append(lang)
        samples.append({'lang': lang, 'raw': raw})

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

    print("[Baseline pipeline] Tri override + MFR + S5 top-2 ...")
    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    pred_per_row = []
    n_change = 0
    n_change_tri = 0
    n_change_mfr = 0
    n_change_s5 = 0
    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        mfrp = mfr_preds[row_idx]; trp = tri_preds[row_idx]
        protected = set(find_protected_indices(raw))
        row_pred = []
        for i, tok in enumerate(raw):
            if l not in blocked and trp[i] != tok:
                pred = trp[i]; n_change_tri += 1
            elif mfrp[i] != tok:
                pred = mfrp[i]; n_change_mfr += 1
            elif xlmr_preds[row_idx][i] == 1 and i not in protected:
                lc = normalize_lang_code(l)
                info = mfr_stats.get(lc, {}).get(tok)
                cand = None
                if info is not None:
                    cands = info.get('candidates') or {}
                    items = sorted(((c, n) for c, n in cands.items() if c != tok), key=lambda x: -x[1])
                    if items:
                        cand = items[0][0]
                if cand is not None:
                    pred = cand; n_change_s5 += 1
                else:
                    pred = tok
            else:
                pred = tok
            row_pred.append(pred)
            if pred != tok: n_change += 1
        pred_per_row.append(row_pred)

    n_total = sum(len(r) for r in raw_per_row)
    print(f"\n  changed tokens: {n_change}/{n_total} ({n_change/n_total*100:.2f}%)")
    print(f"    Trigram: {n_change_tri}")
    print(f"    MFR:     {n_change_mfr}")
    print(f"    S5:      {n_change_s5}")

    # Build output df: same shape as input (raw, norm, lang ...) but norm filled
    out_df = df.copy()
    out_df['norm'] = pred_per_row
    out_path = base / "outputs" / "test_pred_baseline_noLLM.dev"
    out_df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nOK saved: {out_path}  ({size_mb:.2f} MB)")
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
