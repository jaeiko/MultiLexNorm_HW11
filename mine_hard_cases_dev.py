#!/usr/bin/env python3
"""Mine dev (5972 row) hard cases for the LLM-correction stage.

  hard case: baseline_pred == tok AND xlmr == 1 AND not protected
    (Tri+MFR baseline left the token unchanged, XLM-R flags it as noise,
     and it is not a protected token → LLM should try to correct it)

Outputs:
  outputs/hard_cases_dev.jsonl  (~4065 records) — Stage 2(LLM) 입력
    레코드 키: dev_row_idx, tok_idx, lang, token, gt(=''), cat, raw_sentence, prev, next, mfr_top5
  outputs/baseline_dev.json     (5972 rows) — Stage 3(build) 입력
    [{raw, lang, pred}, ...]  pred = Tri+MFR baseline (LLM overlay 전)
"""

import os
import sys
import gzip
import pickle
import json
import time
from pathlib import Path
from collections import Counter

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

_HERE = Path(__file__).parent   # = MultiLexNorm_HW11 루트
sys.path.insert(0, str(_HERE))

XLMR_PATH = os.environ.get("XLMR_MODEL_PATH", str(_HERE.parent / "xlmr_finetuned_colab"))
# None → lang-specific threshold(detection.get_lang_threshold), 숫자 → 고정 threshold (0.5 = argmax)
XLMR_THRESHOLD = os.environ.get("XLMR_THRESHOLD")

from trigram_predictor import predict_trigram  # noqa
from smart_guard_mfr_v2 import predict_smart_guarded_mfr_v2, find_protected_indices, normalize_lang_code  # noqa
from detection import AnomalyDetector  # noqa


def main():
    print("[mine dev] dev 5972 hard case mining")
    t0 = time.time()

    # dev sample (predictions.json) = raw/lang의 단일 소스
    dev_sample_path = _HERE / "outputs" / "submission_dev" / "predictions.json"
    with open(dev_sample_path, 'r', encoding='utf-8') as f:
        dev = json.load(f)
    print(f"  dev sample rows={len(dev)}")

    raw_per_row, lang_per_row, samples = [], [], []
    for d in dev:
        raw = [str(x) for x in d['raw']]
        lang = str(d['lang'])
        raw_per_row.append(raw)
        lang_per_row.append(lang)
        samples.append({'lang': lang, 'raw': raw})
    print(f"  dev tokens total: {sum(len(r) for r in raw_per_row):,}")

    with gzip.open(_HERE / "outputs" / "trigram_stats_internal_v1.pkl.gz", "rb") as f:
        tri_stats = pickle.load(f)
    with gzip.open(_HERE / "mfr_stats.pkl.gz", "rb") as f:
        mfr_stats = pickle.load(f)

    threshold = None if XLMR_THRESHOLD is None else float(XLMR_THRESHOLD)
    threshold_mode = 'lang-specific' if threshold is None else f'fixed={threshold}'
    print(f"[XLMR detection] threshold={threshold_mode} ...")
    detector = AnomalyDetector(XLMR_PATH)
    xlmr_preds = detector.predict_labels(raw_per_row, lang_per_row, threshold=threshold)
    del detector
    torch.cuda.empty_cache()
    print(f"  XLMR done {len(xlmr_preds)} rows")

    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    hard_records = []
    baseline_per_row = []
    cats = Counter()
    for row_idx, raw in enumerate(raw_per_row):
        lang = lang_per_row[row_idx]
        mfr_row = mfr_preds[row_idx]
        tri_row = tri_preds[row_idx]
        xlmr_row = xlmr_preds[row_idx]
        protected = set(find_protected_indices(raw))
        baseline_row = []
        for i, tok in enumerate(raw):
            # baseline = Tri (block된 lang 제외) → MFR
            if lang not in blocked and tri_row[i] != tok:
                baseline_pred = tri_row[i]
            elif mfr_row[i] != tok:
                baseline_pred = mfr_row[i]
            else:
                baseline_pred = tok
            baseline_row.append(baseline_pred)

            # hard case gating: XLMR=1 AND not protected AND baseline가 안 건드림
            if xlmr_row[i] != 1:
                continue
            if i in protected:
                continue
            if baseline_pred != tok:
                continue  # baseline이 이미 변환 → LLM 영역 아님

            # cat (mfr_stats 기반 fine-grained 분류 — 분석용)
            lang_code = normalize_lang_code(lang)
            info = mfr_stats.get(lang_code, {}).get(tok)
            if info is None:
                cat = 'mfr_oov'
            else:
                cands = info.get('candidates') or {}
                non_raw = [(c, n) for c, n in cands.items() if c != tok]
                cat = 'all_cands_raw' if not non_raw else 'has_non_raw_cands'
            cats[cat] += 1

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
                'gt': '',  # test set has no gt
                'cat': cat,
                'raw_sentence': raw,
                # 분석용 부가 메타 (llm_correct/build는 미사용)
                'prev': prev_tok,
                'next': next_tok,
                'mfr_top5': top5,
            })
        baseline_per_row.append(baseline_row)

    # Stage 2 입력: hard cases
    hc_path = _HERE / "outputs" / "hard_cases_dev.jsonl"
    with open(hc_path, "w", encoding="utf-8") as f:
        for r in hard_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stage 3 입력: 전체 baseline (LLM overlay 전 Tri+MFR 예측)
    baseline_path = _HERE / "outputs" / "baseline_dev.json"
    baseline_out = [
        {'raw': raw_per_row[ri], 'lang': lang_per_row[ri], 'pred': baseline_per_row[ri]}
        for ri in range(len(raw_per_row))
    ]
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_out, f, ensure_ascii=False)

    print(f"\n[Hard cases extracted]")
    print(f"  hard cases : {len(hard_records):,}")
    print(f"\n  cat distribution:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c:<20} {n}")
    print(f"\nOK saved: {hc_path}")
    print(f"OK saved: {baseline_path}  ({len(baseline_out)} rows)")
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
