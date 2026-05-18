#!/usr/bin/env python3
"""
Production-realistic hard case mining on mini validation set (1131 rows).

기존 mine_hard_cases_mini.py와 차이:
  - gt(=norm) 정보를 추출 조건에 사용하지 않음
  - 조건: baseline_pred == raw AND XLMR=1 AND not protected
  - 즉 baseline (Tri + MFR + S5)이 변환 안 한 토큰 중 XLMR이 비표준이라 라벨한 영역
  - 이 영역엔 정상 토큰(raw==gt, XLMR FP_det)도 포함되어 LLM의 ΔFP를 production-realistic하게 측정 가능

분석용 카테고리는 gt 사용해 사후 분류만:
  - raw == gt → normal_xlmr_fp  (정상 토큰을 XLMR이 잘못 1로 라벨)
  - raw != gt → nonstandard_missed (비표준 영역. B/C/A_xlmr1 등 세분류는 기존과 동일)

Output: outputs/hard_cases_mini_realistic.jsonl
  레코드 형식은 기존 hard_cases_mini.jsonl과 동일 (gt 포함 — 평가 시 사용)
"""

import os
import sys
import json
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


def main():
    print("[Hard case mining REALISTIC] mini val 1131 (no gt-based filtering)")
    t0 = time.time()
    base = _HERE
    repo = base.parent

    mini_path = DATA_DIR / "mini_validation" / "validation_mini_17langs_1of8_seed42.parquet"
    mini_df = pd.read_parquet(mini_path)
    print(f"  mini rows={len(mini_df)}")

    with gzip.open(base / "outputs" / "trigram_stats_internal_v1.pkl.gz", "rb") as f:
        tri_stats = pickle.load(f)
    with gzip.open(repo / "mfr_first_package" / "mfr_stats.pkl.gz", "rb") as f:
        mfr_stats = pickle.load(f)

    # XLMR inference on mini
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    print("[XLMR inference on mini] ...")
    tokz = AutoTokenizer.from_pretrained(XLMR_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForTokenClassification.from_pretrained(XLMR_PATH).to(device).eval()

    samples = []
    raw_per_row, lang_per_row = [], []
    norm_per_row = []
    orig_id_per_row = []
    for _, row in mini_df.iterrows():
        raw = to_list(row['raw'])
        norm = to_list(row['norm'])
        lang = str(row['lang'])
        raw_per_row.append(raw)
        norm_per_row.append(norm)
        lang_per_row.append(lang)
        orig_id_per_row.append(int(row['__original_row_id']))
        samples.append({'lang': lang, 'raw': raw})

    BATCH = 16
    MAX_LEN = 512
    xlmr_preds_mini = []
    with torch.no_grad():
        for i in range(0, len(raw_per_row), BATCH):
            batch = raw_per_row[i:i+BATCH]
            enc = tokz(batch, is_split_into_words=True, return_tensors='pt',
                       padding=True, truncation=True, max_length=MAX_LEN)
            enc_dev = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc_dev).logits
            preds = logits.argmax(-1).cpu().tolist()
            for b_idx in range(len(batch)):
                word_ids = tokz(batch[b_idx:b_idx+1], is_split_into_words=True,
                                return_tensors='pt', truncation=True, max_length=MAX_LEN
                               ).word_ids(0)
                row_pred = preds[b_idx]
                wpreds = [None] * len(batch[b_idx])
                for tok_idx, w_id in enumerate(word_ids):
                    if w_id is None or w_id >= len(batch[b_idx]): continue
                    if wpreds[w_id] is None:
                        wpreds[w_id] = row_pred[tok_idx]
                wpreds = [0 if p is None else int(p) for p in wpreds]
                xlmr_preds_mini.append(wpreds)
    print(f"  XLMR done: {len(xlmr_preds_mini)} rows")
    del model
    torch.cuda.empty_cache()

    mfr_preds = predict_smart_guarded_mfr_v2(samples, mfr_stats)
    tri_preds, _ = predict_trigram(samples, tri_stats, {
        'variant': 'tri_bi_both', 'conf_min': 0.70, 'min_total': 1, 'use_protect': True
    })
    blocked = {'it', 'es', 'th', 'id'}

    hard_records = []
    cats = Counter()
    realistic_cats = Counter()
    n_total_tok = 0

    for row_idx, raw in enumerate(raw_per_row):
        l = lang_per_row[row_idx]
        norm = norm_per_row[row_idx]
        mfrp = mfr_preds[row_idx]
        trp = tri_preds[row_idx]
        xlmr_row = xlmr_preds_mini[row_idx]
        protected = set(find_protected_indices(raw))

        for i, tok in enumerate(raw):
            n_total_tok += 1
            gt = norm[i] if i < len(norm) else tok

            # Apply baseline pipeline (S5+Tri block min_n=1, conf=0)
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

            # Production-realistic hard case definition (no gt usage)
            if bp != tok:                # baseline 이미 변환 → LLM 영역 아님
                continue
            if xlmr_row[i] != 1:         # XLMR이 정상이라 판단 → LLM 영역 아님
                continue
            if i in protected:           # protect 정책
                continue

            # 이 토큰은 production-realistic hard case
            lc = normalize_lang_code(l)
            info = mfr_stats.get(lc, {}).get(tok)

            # 분석용 카테고리 (gt 사용 — 결과 분석에만)
            is_normal = (tok == gt)
            if is_normal:
                realistic_cat = 'normal_xlmr_fp'  # XLMR false alarm 영역 (정상 토큰)
            else:
                realistic_cat = 'nonstandard_missed'  # baseline FN 영역 (비표준)
            realistic_cats[realistic_cat] += 1

            # 세부 cat (기존 분류와 같음, 단 baseline_pred==raw 토큰만 포함)
            if info is None:
                cat = 'B_mfr_oov'
            else:
                cands = info.get('candidates') or {}
                non_raw = [(c, n) for c, n in cands.items() if c != tok]
                if not non_raw:
                    cat = 'C_all_cands_are_raw'
                elif any(c == gt for c, _ in non_raw):
                    cat = 'D_gt_in_cands_misranked'
                else:
                    cat = 'D_gt_not_in_cands'
            cats[cat] += 1

            # Top-3 MFR candidates
            top3 = []
            if info is not None:
                cands = info.get('candidates') or {}
                items = sorted(cands.items(), key=lambda x: -x[1])[:5]
                top3 = items
            prev_tok = raw[i-1] if i > 0 else '<BOS>'
            next_tok = raw[i+1] if i < len(raw) - 1 else '<EOS>'
            hard_records.append({
                'mini_row_idx': row_idx,
                'orig_id': orig_id_per_row[row_idx],
                'lang': l,
                'tok_idx': i,
                'raw_sentence': raw,
                'token': tok,
                'gt': gt,
                'prev': prev_tok,
                'next': next_tok,
                'mfr_top5': top3,
                'tri_pred': trp[i] if trp[i] != tok else None,
                'mfr_pred': mfrp[i] if mfrp[i] != tok else None,
                'cat': cat,
                'is_normal_gt': is_normal,  # 평가/분석용 (raw == gt 여부)
            })

    print(f"\n[Hard cases (realistic, no gt-filter)]")
    print(f"  total tokens: {n_total_tok}")
    print(f"  hard cases extracted: {len(hard_records)}")
    print(f"\n  realistic split (analysis only):")
    for k in sorted(realistic_cats.keys()):
        print(f"    {k:<22} {realistic_cats[k]}")
    print(f"\n  fine-grained category split:")
    for k in sorted(cats.keys()):
        print(f"    {k:<28} {cats[k]}")

    # Save
    out_path = base / "outputs" / "hard_cases_mini_realistic.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in hard_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nOK: {len(hard_records)} → {out_path}")

    # Per-lang dist
    lang_dist = Counter(r['lang'] for r in hard_records)
    print(f"\n[Per-lang]")
    for l, c in sorted(lang_dist.items(), key=lambda x: -x[1]):
        print(f"    {l:<6} {c:>4d}")

    print(f"\nTotal: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
