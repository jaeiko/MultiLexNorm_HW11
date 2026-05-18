#!/usr/bin/env python3
"""
Production-realistic LLM correction evaluator.

핵심 차이 (vs eval_llm_any.py):
  - Hard case 588개 중 정상(raw==gt)/비표준(raw!=gt) 분해
  - LLM이 정상 토큰을 잘못 변환 → ΔFP (진짜 production FP risk)
  - LLM이 비표준 토큰을 정답 → ΔTP
  - LLM "raw 유지" 응답률 (정상 토큰에 적용된 안전 행동)

Usage:
  python eval_llm_realistic.py --input llm_corrections_realistic_gpt5mini_default.jsonl --label gpt5mini_default
"""

import os
import argparse
import sys
import json
import gzip
import pickle
import ast
import time
from pathlib import Path
from collections import defaultdict, Counter

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='llm corrections jsonl filename (under outputs/)')
    ap.add_argument('--label', required=True)
    args = ap.parse_args()

    print(f"[Eval realistic] label={args.label}")
    t0 = time.time()
    base = _HERE
    repo = base.parent

    mini = pd.read_parquet(DATA_DIR / "mini_validation" / "validation_mini_17langs_1of8_seed42.parquet")
    with gzip.open(base / "outputs" / "trigram_stats_internal_v1.pkl.gz", "rb") as f:
        tri_stats = pickle.load(f)
    with gzip.open(repo / "mfr_first_package" / "mfr_stats.pkl.gz", "rb") as f:
        mfr_stats = pickle.load(f)

    # XLMR
    tokz = AutoTokenizer.from_pretrained(XLMR_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForTokenClassification.from_pretrained(XLMR_PATH).to(device).eval()

    samples, raw_per_row, lang_per_row, norm_per_row = [], [], [], []
    for _, row in mini.iterrows():
        raw = to_list(row['raw']); norm = to_list(row['norm']); lang = str(row['lang'])
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

    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    # Build baseline (S5+Tri block min_n=1 conf=0) — production pipeline
    flat_raw, flat_gt, flat_lang, flat_pred_best = [], [], [], []
    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        norm = norm_per_row[row_idx]
        mfrp = mfr_preds[row_idx]; trp = tri_preds[row_idx]
        protected = set(find_protected_indices(raw))
        for i, tok in enumerate(raw):
            gt = norm[i] if i < len(norm) else tok
            flat_raw.append(tok); flat_gt.append(gt); flat_lang.append(l)
            use_t = l not in blocked and trp[i] != tok
            if use_t:
                pred = trp[i]
            elif mfrp[i] != tok:
                pred = mfrp[i]
            elif xlmr_preds[row_idx][i] == 1 and i not in protected:
                lc = normalize_lang_code(l)
                info = mfr_stats.get(lc, {}).get(tok)
                cand = None
                if info is not None:
                    cands = info.get('candidates') or {}
                    items = sorted(((c, n) for c, n in cands.items() if c != tok), key=lambda x: -x[1])
                    if items:
                        cand = items[0][0]
                pred = cand if cand is not None else tok
            else:
                pred = tok
            flat_pred_best.append(pred)

    m_best = compute_metrics(flat_pred_best, flat_gt, flat_raw)
    print(f"\n[Baseline (S5+Tri block)]  TP={m_best['tp']} FP={m_best['fp']} FN={m_best['fn']} TN={m_best['tn']} ERR={m_best['err']:+.4f}")

    # Load LLM corrections
    llm_path = base / "outputs" / args.input
    llm_recs = [json.loads(l) for l in llm_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f"  LLM records: {len(llm_recs)}")

    # Decompose by normal/nonstandard
    by_normal = defaultdict(lambda: {'total': 0, 'kept_raw': 0, 'changed': 0, 'changed_correct': 0, 'changed_wrong': 0})
    overlay = {}
    for r in llm_recs:
        key = (r['mini_row_idx'], r['tok_idx'])
        norm = r.get('normalized')
        if norm is None:
            continue
        overlay[key] = norm
        is_normal = r.get('is_normal_gt', None)
        if is_normal is None:
            # Fall back: compute from token vs gt
            is_normal = (r['token'] == r['gt'])
        bucket = 'normal' if is_normal else 'nonstandard'
        by_normal[bucket]['total'] += 1
        if norm == r['token']:
            by_normal[bucket]['kept_raw'] += 1
        else:
            by_normal[bucket]['changed'] += 1
            if norm == r['gt']:
                by_normal[bucket]['changed_correct'] += 1
            else:
                by_normal[bucket]['changed_wrong'] += 1

    print("\n[LLM behavior on hard cases (normal vs nonstandard)]")
    print(f"  {'bucket':<14} {'total':>6} {'kept':>5} {'changed':>8} | {'cor':>5} {'wrg':>5}  keep%  change%")
    for b in ['normal', 'nonstandard']:
        s = by_normal[b]
        if s['total'] == 0: continue
        keep_pct = s['kept_raw'] / s['total'] * 100
        cha_pct = s['changed'] / s['total'] * 100
        print(f"  {b:<14} {s['total']:>6d} {s['kept_raw']:>5d} {s['changed']:>8d} | "
              f"{s['changed_correct']:>5d} {s['changed_wrong']:>5d}  {keep_pct:>5.1f}%  {cha_pct:>5.1f}%")

    # Apply LLM overlay
    flat_pred_llm = list(flat_pred_best)
    flat_idx = 0
    used = 0
    for row_idx, raw in enumerate(raw_per_row):
        for i, tok in enumerate(raw):
            key = (row_idx, i)
            if key in overlay:
                flat_pred_llm[flat_idx] = overlay[key]
                used += 1
            flat_idx += 1

    m_llm = compute_metrics(flat_pred_llm, flat_gt, flat_raw)
    print(f"\n[After LLM overlay] used={used}")
    print(f"  TP={m_llm['tp']} FP={m_llm['fp']} FN={m_llm['fn']} TN={m_llm['tn']} ERR={m_llm['err']:+.4f}")
    print(f"  ΔERR vs baseline: {m_llm['err']-m_best['err']:+.4f}")
    print(f"  ΔTP: {m_llm['tp']-m_best['tp']:+d}    ΔFP: {m_llm['fp']-m_best['fp']:+d}    ΔFN: {m_llm['fn']-m_best['fn']:+d}    ΔTN: {m_llm['tn']-m_best['tn']:+d}")

    # ΔFP / ΔTP source 검증 (LLM이 만든 거)
    # 정상 토큰을 LLM이 변환했고 wrong이면 ΔFP에 기여
    # 비표준 토큰을 LLM이 GT로 변환하면 ΔTP에 기여
    new_fp = by_normal['normal']['changed']  # LLM이 정상 토큰을 변환한 모든 케이스
    # 단 그 중 changed_correct 있을까? normal에선 raw==gt이라 changed_correct 의미상 없음 (raw로 갔어야 GT)
    # 만약 LLM이 normal 토큰에 다른 norm 출력 → norm != raw → FP
    # changed_correct in normal = norm==gt==tok = raw 유지인데 changed로 분류된 거? changed = norm!=tok, gt==tok이면 norm!=gt → wrong → changed_wrong
    # 즉 normal bucket에서 changed_correct는 0이어야 정상

    new_tp = by_normal['nonstandard']['changed_correct']
    new_fn_kept = by_normal['nonstandard']['kept_raw']  # 그대로 둬서 여전히 FN
    new_fn_wrong = by_normal['nonstandard']['changed_wrong']  # 다른 norm으로 잘못 변환

    print(f"\n[ΔFP/ΔTP breakdown]")
    print(f"  새로 생긴 FP (정상 토큰을 LLM이 변환): {new_fp}")
    print(f"  새로 생긴 TP (비표준 토큰을 LLM이 GT로): {new_tp}")
    print(f"  여전히 FN (비표준인데 raw 유지): {new_fn_kept}")
    print(f"  여전히 FN (비표준 + wrong norm): {new_fn_wrong}")

    # Save
    out_dir = base / "outputs" / "experiment_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'label': args.label,
        'input_file': args.input,
        'baseline': m_best,
        'llm_overlay': m_llm,
        'by_bucket': {b: dict(s) for b, s in by_normal.items()},
        'delta': {
            'tp': m_llm['tp'] - m_best['tp'],
            'fp': m_llm['fp'] - m_best['fp'],
            'fn': m_llm['fn'] - m_best['fn'],
            'tn': m_llm['tn'] - m_best['tn'],
            'err': m_llm['err'] - m_best['err'],
        },
    }
    out_file = out_dir / f"eval_llm_realistic_{args.label}_mini.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nOK saved: {out_file}  total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
