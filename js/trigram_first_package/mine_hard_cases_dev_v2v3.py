#!/usr/bin/env python3
"""Mine dev (5972 row) hard cases for v3-pipeline (superset), tag each record
with v2_hard flag so a single LLM batch can later serve both v2 and v3.

  v2 = Tri + MFR + S5     hard: bp_v2 == tok AND xlmr==1 AND not protected
  v3 = Tri + MFR          hard: bp_v3 == tok AND xlmr==1 AND not protected
  superset = v3 hard (always ⊇ v2 hard)

Output: outputs/hard_cases_dev_v2v3.jsonl  (~4067 records)
  Each record keeps the JSONL keys llm_correct_local.py expects, plus:
    'v2_hard'   : bool   — was this token in v2's hard set?
    'v3_hard'   : True   — by definition
    'dev_row_idx'  : int — dev sample row index (0..5971)
    'test_row_idx' : int — original test parquet row index
  (gt is unknown for test set → set to '' to keep schema compatibility)
"""

import os
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
    print("[mine dev v2v3] dev 5972 superset (v3), tagged with v2_hard")
    t0 = time.time()
    base = _HERE
    repo = base.parent

    test_path = repo / "multilexnorm2026-dataset" / "internal_v1" / "data" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(test_path)
    print(f"  test parquet rows={len(df)}")

    dev_sample_path = repo / "outputs" / "submission_dev" / "predictions.json"
    with open(dev_sample_path, 'r', encoding='utf-8') as f:
        dev = json.load(f)
    print(f"  dev sample rows={len(dev)}")

    # dev row → test parquet index (raw_tuple+lang 키 매칭, 중복 first-come 할당)
    test_keys = [(tuple(str(x) for x in r['raw']), str(r['lang'])) for _, r in df.iterrows()]
    key_to_idx = {}
    for i, k in enumerate(test_keys):
        key_to_idx.setdefault(k, []).append(i)
    dev_test_indices = []
    for d in dev:
        k = (tuple(d['raw']), d['lang'])
        pool = key_to_idx.get(k)
        dev_test_indices.append(pool.pop(0) if pool else None)
    miss = sum(1 for x in dev_test_indices if x is None)
    print(f"  mapped dev → test: {len(dev_test_indices) - miss} / {len(dev_test_indices)}  (missing={miss})")
    assert miss == 0

    raw_per_row, lang_per_row, samples = [], [], []
    for ti in dev_test_indices:
        row = df.iloc[ti]
        raw = to_list(row['raw']); lang = str(row['lang'])
        raw_per_row.append(raw); lang_per_row.append(lang)
        samples.append({'lang': lang, 'raw': raw})
    print(f"  dev tokens total: {sum(len(r) for r in raw_per_row):,}")

    with gzip.open(base / "outputs" / "trigram_stats_internal_v1.pkl.gz", "rb") as f:
        tri_stats = pickle.load(f)
    with gzip.open(repo / "mfr_first_package" / "mfr_stats.pkl.gz", "rb") as f:
        mfr_stats = pickle.load(f)

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

    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    hard_records = []
    cats = Counter()
    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        mfrp = mfr_preds[row_idx]; trp = tri_preds[row_idx]
        protected = set(find_protected_indices(raw))
        xrow = xlmr_preds[row_idx]
        for i, tok in enumerate(raw):
            # v3 baseline
            if l not in blocked and trp[i] != tok:
                bp_v3 = trp[i]
            elif mfrp[i] != tok:
                bp_v3 = mfrp[i]
            else:
                bp_v3 = tok
            # v2 baseline (v3 위에 S5)
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

            # superset gating (v3)
            if xrow[i] != 1: continue
            if i in protected: continue
            if bp_v3 != tok: continue  # v3가 이미 변환 → LLM 영역 아님

            v2_hard = (bp_v2 == tok)  # v2에서도 hard였는지 (False면 S5가 처리한 토큰)

            # cat (mfr_stats에서 fine-grained 분류 — 분석용)
            lc = normalize_lang_code(l)
            info = mfr_stats.get(lc, {}).get(tok)
            if info is None:
                cat = 'B_mfr_oov'
            else:
                cands = info.get('candidates') or {}
                non_raw = [(c, n) for c, n in cands.items() if c != tok]
                cat = 'C_all_cands_are_raw' if not non_raw else 'D_has_non_raw_cands'
            cats[cat] += 1

            top5 = []
            if info is not None:
                items = sorted(info.get('candidates', {}).items(), key=lambda x: -x[1])[:5]
                top5 = items
            prev_tok = raw[i-1] if i > 0 else '<BOS>'
            next_tok = raw[i+1] if i < len(raw) - 1 else '<EOS>'
            hard_records.append({
                # llm_correct_local.py 호환 키 (mini_row_idx 이름으로 dev_row_idx 저장)
                'mini_row_idx': row_idx,
                'tok_idx': i,
                'lang': l,
                'token': tok,
                'gt': '',  # test set has no gt
                'cat': cat,
                'raw_sentence': raw,
                # 부가 메타
                'dev_row_idx': row_idx,
                'test_row_idx': dev_test_indices[row_idx],
                'prev': prev_tok,
                'next': next_tok,
                'mfr_top5': top5,
                'v3_hard': True,
                'v2_hard': v2_hard,
            })

    out_path = base / "outputs" / "hard_cases_dev_v2v3.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in hard_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_v3 = len(hard_records)
    n_v2 = sum(1 for r in hard_records if r['v2_hard'])
    n_v3only = n_v3 - n_v2
    print(f"\n[Hard cases extracted]")
    print(f"  v3 (superset)        : {n_v3:,}")
    print(f"  v2 (subset of v3)    : {n_v2:,}")
    print(f"  v3-only (S5 handled) : {n_v3only:,}")
    print(f"\n  cat distribution:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c:<24} {n}")
    print(f"\nOK saved: {out_path}")
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
