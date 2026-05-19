#!/usr/bin/env python3
"""Count hard cases on dev 5972 rows under two pipelines (XLMR=1 detection used in both):

  v2 = Tri + MFR + S5 (current full baseline)
       hard cases: bp_v2 == tok AND xlmr == 1 AND not protected
  v3 = Tri + MFR        (no S5)
       hard cases: bp_v3 == tok AND xlmr == 1 AND not protected

Goal: measure the superset size (=v3) before running the actual LLM batch.
"""

import sys
import gzip
import pickle
import ast
import json
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

from trigram_predictor import predict_trigram  # noqa
from smart_guard_mfr_v2 import predict_smart_guarded_mfr_v2, find_protected_indices, normalize_lang_code  # noqa


def to_list(v):
    if hasattr(v, 'tolist'):
        return [str(x) for x in v.tolist()]
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return [str(x) for x in v]


def main():
    print("[count_dev_hard_v2v3] dev 5972 rows, XLMR=1 detection used in both")
    t0 = time.time()
    base = _HERE
    repo = base.parent

    # 1) test parquet 로드 → dev sample이 가리키는 5972 row만 사용
    test_path = repo / "multilexnorm2026-dataset" / "internal_v1" / "data" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(test_path)
    print(f"  test parquet rows={len(df)}")

    dev_sample_path = repo / "outputs" / "submission_dev" / "predictions.json"
    with open(dev_sample_path, 'r', encoding='utf-8') as f:
        dev = json.load(f)
    print(f"  dev sample rows={len(dev)}")

    # test row keys (raw_tuple, lang) → indices
    test_keys = [(tuple(str(x) for x in r['raw']), str(r['lang'])) for _, r in df.iterrows()]
    key_to_idx = {}
    for i, k in enumerate(test_keys):
        key_to_idx.setdefault(k, []).append(i)

    # dev row → test row index
    dev_test_indices = []
    missing = 0
    for d in dev:
        k = (tuple(d['raw']), d['lang'])
        pool = key_to_idx.get(k)
        if not pool:
            missing += 1
            continue
        dev_test_indices.append(pool.pop(0))
    print(f"  mapped dev → test indices: {len(dev_test_indices)}  missing={missing}")

    # dev 5972의 raw/lang만 추출
    raw_per_row, lang_per_row = [], []
    samples = []
    for ti in dev_test_indices:
        row = df.iloc[ti]
        raw = to_list(row['raw'])
        lang = str(row['lang'])
        raw_per_row.append(raw); lang_per_row.append(lang)
        samples.append({'lang': lang, 'raw': raw})
    print(f"  dev tokens total: {sum(len(r) for r in raw_per_row):,}")

    # 2) stats 로드
    with gzip.open(base / "outputs" / "trigram_stats_internal_v1.pkl.gz", "rb") as f:
        tri_stats = pickle.load(f)
    with gzip.open(repo / "mfr_first_package" / "mfr_stats.pkl.gz", "rb") as f:
        mfr_stats = pickle.load(f)

    # 3) XLMR
    print("[XLMR inference] ...")
    tokz = AutoTokenizer.from_pretrained(str(repo / "xlmr_finetuned_colab"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForTokenClassification.from_pretrained(str(repo / "xlmr_finetuned_colab")).to(device).eval()
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
    print(f"  XLMR done {len(xlmr_preds)} rows")

    # 4) Stage A: bp_v2 (Tri+MFR+S5), bp_v3 (Tri+MFR)
    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    n_total = 0
    n_v2_hard = 0
    n_v3_hard = 0
    n_v2_only = 0       # impossible by construction (v2 ⊆ v3)
    n_v3_only = 0       # v3 hard but not v2 hard (= S5가 변환한 토큰)
    per_lang_v2 = Counter()
    per_lang_v3 = Counter()

    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        mfrp = mfr_preds[row_idx]; trp = tri_preds[row_idx]
        protected = set(find_protected_indices(raw))
        xrow = xlmr_preds[row_idx]
        for i, tok in enumerate(raw):
            n_total += 1
            # v3 baseline: Tri (lang-block 적용) → MFR → 그 외 raw 유지
            if l not in blocked and trp[i] != tok:
                bp_v3 = trp[i]
            elif mfrp[i] != tok:
                bp_v3 = mfrp[i]
            else:
                bp_v3 = tok
            # v2 baseline: v3 위에 S5 추가
            if bp_v3 != tok:
                bp_v2 = bp_v3
            elif xrow[i] == 1 and i not in protected:
                lc = normalize_lang_code(l)
                info = mfr_stats.get(lc, {}).get(tok)
                cand = None
                if info is not None:
                    cands = info.get('candidates') or {}
                    items = sorted(((c, n) for c, n in cands.items() if c != tok), key=lambda x: -x[1])
                    if items:
                        cand = items[0][0]
                bp_v2 = cand if cand is not None else tok
            else:
                bp_v2 = tok

            # Hard case (둘 다 xlmr=1 + not protected gating)
            if xrow[i] != 1: continue
            if i in protected: continue
            v2_h = (bp_v2 == tok)
            v3_h = (bp_v3 == tok)
            if v3_h:
                n_v3_hard += 1
                per_lang_v3[l] += 1
            if v2_h:
                n_v2_hard += 1
                per_lang_v2[l] += 1
                if not v3_h:
                    n_v2_only += 1
            if v3_h and not v2_h:
                n_v3_only += 1

    print(f"\n[Dev 5972 row hard case summary]")
    print(f"  total tokens          : {n_total:,}")
    print(f"  v2 (Tri+MFR+S5) hard  : {n_v2_hard:,}  ({n_v2_hard/n_total*100:.2f}% of tokens)")
    print(f"  v3 (Tri+MFR)    hard  : {n_v3_hard:,}  ({n_v3_hard/n_total*100:.2f}% of tokens)")
    print(f"  difference (S5 변환)  : {n_v3_only:,}  (= v3에만 hard, v2에선 S5가 처리한 토큰)")
    print(f"  v2 \\ v3 (impossible)  : {n_v2_only}")
    print(f"\n  superset(=v3) size    : {n_v3_hard:,}  ← 한 번 gemma 호출할 토큰 수")

    print(f"\n[Per-lang v3 hard]")
    for l, c in sorted(per_lang_v3.items(), key=lambda x: -x[1]):
        print(f"    {l:<6} v3={c:>4d}  v2={per_lang_v2.get(l, 0):>4d}")

    # 예상 호출 시간 추산 (gemma e4b, mini 588 → 35.6분 기준 = 약 3.65초/call)
    sec_per_call = 35.6 * 60 / 588
    est_min = n_v3_hard * sec_per_call / 60
    print(f"\n[Time estimate (gemma e4b zs, workers=2, mini 35.6min/588 hard 기준)]")
    print(f"  ~{sec_per_call:.2f} sec/call → {est_min:.1f} min total ({est_min/60:.2f} h)")
    print(f"  workers=4로 늘리면 약 절반")
    print(f"\nTotal: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
