#!/usr/bin/env python3
"""Build 4 ablation combos (tri / mfr / tri+mfr / mfr+xlmr) on test set,
then map each to dev-phase (5972 rows) and emit zipped predictions.json
matching the official sample format ({raw, norm:["",..], lang, pred:[...]}).

Combo 5 (tri+mfr+xlmr = baseline) is already built — reused if present.
"""

import os
import sys
import gzip
import pickle
import ast
import json
import time
import zipfile
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


# (label, use_tri, use_mfr, use_xlmr_s5)
COMBOS = [
    ('tri_only',    True,  False, False),
    ('mfr_only',    False, True,  False),
    ('tri_mfr',     True,  True,  False),
    ('mfr_xlmr',    False, True,  True),
    ('tri_mfr_xlmr', True, True,  True),   # = baseline, also emitted for parity
]


def decide(tok, i, l, raw, mfrp, trp, xlmr_row, protected, mfr_stats, blocked,
           use_tri, use_mfr, use_xlmr):
    if use_tri and l not in blocked and trp[i] != tok:
        return trp[i], 'tri'
    if use_mfr and mfrp[i] != tok:
        return mfrp[i], 'mfr'
    if use_xlmr and xlmr_row[i] == 1 and i not in protected:
        lc = normalize_lang_code(l)
        info = mfr_stats.get(lc, {}).get(tok)
        if info is not None:
            cands = info.get('candidates') or {}
            items = sorted(((c, n) for c, n in cands.items() if c != tok), key=lambda x: -x[1])
            if items:
                return items[0][0], 'xlmr'
    return tok, 'keep'


def main():
    print(f"[predict_test_combos] {len(COMBOS)} combos")
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
        raw_per_row.append(raw); lang_per_row.append(lang)
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

    print("[Predictor stats] ...")
    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    # Build predictions per combo (per test row)
    combo_preds = {c[0]: [] for c in COMBOS}
    combo_stats = {c[0]: {'tri': 0, 'mfr': 0, 'xlmr': 0, 'keep': 0} for c in COMBOS}

    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        mfrp = mfr_preds[row_idx]; trp = tri_preds[row_idx]
        protected = set(find_protected_indices(raw))
        xrow = xlmr_preds[row_idx]
        for label, use_tri, use_mfr, use_xlmr in COMBOS:
            row_pred = []
            for i, tok in enumerate(raw):
                p, src = decide(tok, i, l, raw, mfrp, trp, xrow, protected, mfr_stats, blocked,
                                use_tri, use_mfr, use_xlmr)
                row_pred.append(p)
                combo_stats[label][src] += 1
            combo_preds[label].append(row_pred)

    print("\n  Combo summary (token-level decision sources):")
    n_total = sum(len(r) for r in raw_per_row)
    for label, _, _, _ in COMBOS:
        s = combo_stats[label]
        changed = s['tri'] + s['mfr'] + s['xlmr']
        print(f"    {label:<15} tri={s['tri']:>5} mfr={s['mfr']:>5} xlmr={s['xlmr']:>5} keep={s['keep']:>6}  changed={changed} ({changed/n_total*100:.2f}%)")

    # ---- Build dev-phase 5972 submissions per combo ----
    print("\n[Building dev-phase submissions]")
    dev_sample_path = repo / "outputs" / "submission_dev" / "predictions.json"
    with open(dev_sample_path, 'r', encoding='utf-8') as f:
        dev = json.load(f)
    print(f"  dev sample rows: {len(dev)}")

    test_keys = [(tuple(r), lang_per_row[i]) for i, r in enumerate(raw_per_row)]
    key_to_idx = {}
    for i, k in enumerate(test_keys):
        key_to_idx.setdefault(k, []).append(i)

    # snapshot pool order per row map
    def build_dev_records(preds_per_test_row):
        kti = {k: list(v) for k, v in key_to_idx.items()}
        recs = []
        miss = 0
        for d in dev:
            k = (tuple(d['raw']), d['lang'])
            pool = kti.get(k)
            if not pool:
                miss += 1
                recs.append({'raw': d['raw'], 'norm': [''] * len(d['raw']),
                             'lang': d['lang'], 'pred': list(d['raw'])})
                continue
            idx = pool.pop(0)
            recs.append({'raw': d['raw'], 'norm': [''] * len(d['raw']),
                         'lang': d['lang'], 'pred': preds_per_test_row[idx]})
        return recs, miss

    out_root = base / "outputs" / "submissions_combos"
    out_root.mkdir(exist_ok=True)
    for label, _, _, _ in COMBOS:
        out_dir = out_root / label
        out_dir.mkdir(exist_ok=True)
        recs, miss = build_dev_records(combo_preds[label])
        out_json = out_dir / "predictions.json"
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False)
        zip_path = out_dir / "predictions.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_json, arcname='predictions.json')
        zs = zip_path.stat().st_size / (1024*1024)
        print(f"  [{label:<15}] miss={miss}  zip={zs:.2f}MB  → {zip_path.relative_to(repo)}")

    print(f"\nTotal: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
