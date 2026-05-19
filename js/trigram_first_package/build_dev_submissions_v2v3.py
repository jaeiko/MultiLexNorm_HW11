#!/usr/bin/env python3
"""Build two codabench dev submissions from a single gemma LLM run.

Inputs:
  outputs/hard_cases_dev_v2v3.jsonl              (4067 records, tagged v2_hard)
  outputs/llm_corrections_dev_v2v3_gemma_zs.jsonl   (one record per hard case)

Outputs:
  outputs/submissions_dev_v2v3/v2_full/predictions.{json,zip}
    baseline = Tri + MFR + S5
    overlay  = LLM 답 (단 v2_hard=True 인 토큰만)
  outputs/submissions_dev_v2v3/v3_no_s5/predictions.{json,zip}
    baseline = Tri + MFR (no S5)
    overlay  = LLM 답 (모든 hard case)
"""

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

from trigram_predictor import predict_trigram  # noqa
from smart_guard_mfr_v2 import predict_smart_guarded_mfr_v2, find_protected_indices, normalize_lang_code  # noqa


def to_list(v):
    if hasattr(v, 'tolist'):
        return [str(x) for x in v.tolist()]
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return [str(x) for x in v]


def main():
    print("[build_dev_submissions_v2v3]")
    t0 = time.time()
    base = _HERE
    repo = base.parent

    # 1) hard cases + LLM corrections 로드
    hc_path = base / "outputs" / "hard_cases_dev_v2v3.jsonl"
    hard = [json.loads(l) for l in hc_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f"  hard cases loaded: {len(hard)}")

    llm_path = base / "outputs" / "llm_corrections_dev_v2v3_gemma_zs.jsonl"
    if not llm_path.exists():
        print(f"  ERROR: {llm_path} 없음. 먼저 llm_correct_local.py 돌리세요.")
        return 1
    llm = [json.loads(l) for l in llm_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    # (row_idx, tok_idx) → normalized
    llm_map = {}
    for r in llm:
        norm = r.get('normalized')
        if isinstance(norm, str):
            llm_map[(r['mini_row_idx'], r['tok_idx'])] = norm
    print(f"  llm corrections (parsed): {len(llm_map)} / {len(llm)}")

    # 2) dev row → test row 매핑 (mining이 부착한 test_row_idx 활용)
    dev_sample_path = repo / "outputs" / "submission_dev" / "predictions.json"
    with open(dev_sample_path, 'r', encoding='utf-8') as f:
        dev = json.load(f)
    print(f"  dev sample rows: {len(dev)}")

    # 3) test parquet 다시 로드 → dev_row_idx별 raw/lang 재구성 + Stage A 재계산
    test_path = repo / "multilexnorm2026-dataset" / "internal_v1" / "data" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(test_path)
    test_keys = [(tuple(str(x) for x in r['raw']), str(r['lang'])) for _, r in df.iterrows()]
    key_to_idx = {}
    for i, k in enumerate(test_keys):
        key_to_idx.setdefault(k, []).append(i)
    dev_test_indices = []
    for d in dev:
        k = (tuple(d['raw']), d['lang'])
        pool = key_to_idx.get(k)
        dev_test_indices.append(pool.pop(0))
    raw_per_row, lang_per_row, samples = [], [], []
    for ti in dev_test_indices:
        row = df.iloc[ti]
        raw = to_list(row['raw']); lang = str(row['lang'])
        raw_per_row.append(raw); lang_per_row.append(lang)
        samples.append({'lang': lang, 'raw': raw})

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

    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    # 4) v2/v3 baseline 계산 + overlay
    v2_preds_per_row = []
    v3_preds_per_row = []
    n_overlay_v2 = 0
    n_overlay_v3 = 0
    n_skip_no_llm = 0
    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        mfrp = mfr_preds[row_idx]; trp = tri_preds[row_idx]
        protected = set(find_protected_indices(raw))
        xrow = xlmr_preds[row_idx]
        v2_row, v3_row = [], []
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

            # LLM overlay 시점 — superset 조건 (xlmr=1 + not protected + bp_v3==tok)
            is_v3_hard = (xrow[i] == 1 and i not in protected and bp_v3 == tok)
            is_v2_hard = is_v3_hard and (bp_v2 == tok)
            llm_norm = llm_map.get((row_idx, i)) if is_v3_hard else None

            # v2 overlay: v2_hard 에 한해서만 LLM 답 적용 (S5가 처리한 토큰은 v2에서 안 건드림)
            if is_v2_hard and llm_norm is not None and llm_norm != tok:
                v2_pred = llm_norm; n_overlay_v2 += 1
            else:
                v2_pred = bp_v2

            # v3 overlay: 모든 v3_hard 에 LLM 답 적용
            if is_v3_hard and llm_norm is not None and llm_norm != tok:
                v3_pred = llm_norm; n_overlay_v3 += 1
            elif is_v3_hard and llm_norm is None:
                n_skip_no_llm += 1
                v3_pred = bp_v3
            else:
                v3_pred = bp_v3

            v2_row.append(v2_pred); v3_row.append(v3_pred)
        v2_preds_per_row.append(v2_row); v3_preds_per_row.append(v3_row)

    print(f"  overlay v2 applied: {n_overlay_v2}")
    print(f"  overlay v3 applied: {n_overlay_v3}")
    print(f"  v3 hard cases without llm answer: {n_skip_no_llm}")

    # 5) dev sample 순서 그대로 records 생성 + zip
    out_root = base / "outputs" / "submissions_dev_v2v3"
    out_root.mkdir(exist_ok=True)
    for label, preds_per_row in [('v2_full', v2_preds_per_row), ('v3_no_s5', v3_preds_per_row)]:
        out_dir = out_root / label
        out_dir.mkdir(exist_ok=True)
        records = []
        for di, d in enumerate(dev):
            records.append({
                'raw': d['raw'],
                'norm': [''] * len(d['raw']),
                'lang': d['lang'],
                'pred': preds_per_row[di],
            })
        out_json = out_dir / "predictions.json"
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False)
        zip_path = out_dir / "predictions.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_json, arcname='predictions.json')
        zs = zip_path.stat().st_size / (1024*1024)
        # changed token count
        nch = sum(1 for di in range(len(dev)) for a, b in zip(dev[di]['raw'], preds_per_row[di]) if a != b)
        print(f"  [{label:<10}] changed={nch}  zip={zs:.2f}MB  → {zip_path.relative_to(repo)}")

    print(f"\nTotal: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
