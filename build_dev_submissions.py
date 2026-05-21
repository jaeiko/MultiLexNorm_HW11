#!/usr/bin/env python3
"""Build the codabench dev submission from baseline + a single LLM run.

Inputs:
  outputs/baseline_dev.json        — Tri+MFR baseline (Stage 1 산출), [{raw,lang,pred}, ...]
  outputs/hard_cases_dev.jsonl     — hard case 키 (Stage 1 산출) — overlay 대상 집합
  outputs/llm_corrections_*.jsonl  — LLM 교정 결과 (Stage 2 산출), hard case당 1 레코드

Output:
  outputs/submissions_dev_final/predictions.{json,zip}
    baseline에 LLM 답을 overlay (현재 hard case이고 유효 답이며 raw 토큰과 다를 때만).

XLMR/MFR/Trigram 재계산 없음 — baseline은 Stage 1이 이미 영속화함.
"""

import os
import sys
import json
import time
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

_HERE = Path(__file__).parent   # = MultiLexNorm_HW11 루트


def main():
    print("[build_dev_submissions]")
    t0 = time.time()

    # 1) baseline 로드 (Stage 1 산출)
    baseline_path = _HERE / "outputs" / "baseline_dev.json"
    if not baseline_path.exists():
        print(f"  ERROR: {baseline_path} 없음. 먼저 mine_hard_cases_dev.py 돌리세요.")
        return 1
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    print(f"  baseline rows: {len(baseline)}")

    # 2) hard case 키 집합 로드 (Stage 1 산출) — overlay는 현재 hard case에 한해서만 적용
    hard_path = _HERE / "outputs" / "hard_cases_dev.jsonl"
    if not hard_path.exists():
        print(f"  ERROR: {hard_path} 없음. 먼저 mine_hard_cases_dev.py 돌리세요.")
        return 1
    hard = [json.loads(l) for l in hard_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    hard_set = {(r['dev_row_idx'], r['tok_idx']) for r in hard}
    print(f"  hard cases: {len(hard_set)}")

    # 3) LLM corrections 로드 (Stage 2 산출)
    llm_filename = os.environ.get("LLM_OUTPUT_JSONL", "llm_corrections_dev_fewshot3.jsonl")
    llm_path = _HERE / "outputs" / llm_filename
    if not llm_path.exists():
        print(f"  ERROR: {llm_path} 없음. 먼저 llm_correct_local.py 돌리세요.")
        return 1
    print(f"  llm jsonl: {llm_filename}")
    llm = [json.loads(l) for l in llm_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f"  llm corrections: {len(llm)} records")

    # 4) overlay — 현재 hard case의 유효 LLM 답을 baseline에 적용
    #    hard_set에 없는 llm 레코드 = stale(다른 mining 산출) → 무시. 비-hard 토큰 오염 방지.
    n_overlay = 0
    n_stale = 0
    for r in llm:
        key = (r['dev_row_idx'], r['tok_idx'])
        if key not in hard_set:
            n_stale += 1
            continue
        norm = r.get('normalized')
        if not isinstance(norm, str):
            continue
        row = baseline[key[0]]
        if norm != row['raw'][key[1]]:
            row['pred'][key[1]] = norm
            n_overlay += 1
    msg = f"  overlay applied: {n_overlay}"
    if n_stale:
        msg += f"  (stale llm records skipped: {n_stale})"
    print(msg)

    # 5) 제출 records 생성 + zip
    out_dir = _HERE / "outputs" / os.environ.get("SUBMISSION_SUBDIR", "submissions_dev_final")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {'raw': row['raw'], 'norm': [''] * len(row['raw']), 'lang': row['lang'], 'pred': row['pred']}
        for row in baseline
    ]
    out_json = out_dir / "predictions.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)
    zip_path = out_dir / "predictions.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname='predictions.json')

    n_changed = sum(1 for row in baseline for a, b in zip(row['raw'], row['pred']) if a != b)
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  changed tokens: {n_changed}  |  zip: {zip_mb:.2f}MB")
    print(f"  → {zip_path}")
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
