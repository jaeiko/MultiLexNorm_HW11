"""CodaBench Development Set Submission Builder.

This script overlays predictions made by the LLM stage onto the Tri+MFR baseline,
and builds the final CodaBench zip package containing the predictions.json.
Only tokens flagged as hard cases are updated using the overlay mechanism.
"""

from __future__ import annotations

import argparse
import sys
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set

# Centralized project imports using paths_config
import paths_config
paths_config.setup_imports()


def build_submission(use_llm: bool = True) -> int:
    """Builds the final predictions.json and zips it for CodaBench submission.

    Args:
        use_llm: When False, emits baseline predictions without any LLM overlay.

    Returns:
        int: Exit status code (0 for success, 1 for error).
    """
    print(f"[build_dev_submissions] mode={'baseline+llm' if use_llm else 'baseline-only'}")
    t0 = time.time()

    # 1. Load N-gram + MFR pre-LLM predictions (from Stage 1)
    baseline_path: Path = paths_config.NGRAM_MFR_PATH
    if not baseline_path.exists():
        print(f"  ERROR: N-gram+MFR prediction not found at {baseline_path}. Execute mine_hard_cases_dev.py first.")
        return 1
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline: List[Dict[str, Any]] = json.load(f)
    print(f"  Loaded baseline rows: {len(baseline)}")

    if use_llm:
        # 2. Load hard cases index mapping (from Stage 1)
        hard_path: Path = paths_config.HARD_CASES_PATH
        if not hard_path.exists():
            print(f"  ERROR: Hard cases file not found at {hard_path}. Execute mine_hard_cases_dev.py first.")
            return 1

        hard: List[Dict[str, Any]] = [
            json.loads(line) for line in hard_path.read_text(encoding='utf-8').splitlines() if line.strip()
        ]
        hard_set: Set[tuple[int, int]] = {(int(r['dev_row_idx']), int(r['tok_idx'])) for r in hard}
        print(f"  Indexed hard cases: {len(hard_set)}")

        # 3. Load LLM corrections (from Stage 2)
        llm_path: Path = paths_config.LLM_OUTPUT_PATH
        if not llm_path.exists():
            print(f"  ERROR: LLM corrections file not found at {llm_path}. Execute llm_correct_local.py first.")
            return 1
        print(f"  Loading LLM outputs file: {llm_path}")

        llm: List[Dict[str, Any]] = [
            json.loads(line) for line in llm_path.read_text(encoding='utf-8').splitlines() if line.strip()
        ]
        print(f"  Loaded LLM records: {len(llm)}")

        # 4. Apply overlays to the baseline on active hard cases
        n_overlay = 0
        n_stale = 0
        for r in llm:
            key = (int(r['dev_row_idx']), int(r['tok_idx']))
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

        msg = f"  Overlay corrections applied successfully: {n_overlay}"
        if n_stale:
            msg += f" (stale/outdated LLM predictions skipped: {n_stale})"
        print(msg)

    # 5. Format final submission data structures and export to compressed ZIP
    out_dir: Path = paths_config.SUBMISSION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {
            'raw': row['raw'],
            'norm': [''] * len(row['raw']),  # Blank placeholder as required by evaluator template
            'lang': row['lang'],
            'pred': row['pred']
        }
        for row in baseline
    ]

    out_json = out_dir / "predictions.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)

    zip_path = out_dir / "predictions.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname='predictions.json')

    # Compute descriptive summary metrics
    n_changed = sum(1 for row in baseline for a, b in zip(row['raw'], row['pred']) if a != b)
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  Total changed tokens over raw text: {n_changed}  |  Compressed size: {zip_mb:.2f}MB")
    print(f"  Final ZIP output generated -> {zip_path}")
    print(f"  Task execution completed in {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build submission ZIP from baseline (+ optional LLM overlay).")
    ap.add_argument('--no-llm', dest='use_llm', action='store_false', default=True,
                    help="Skip LLM overlay step — emit baseline predictions only.")
    args = ap.parse_args()
    sys.exit(build_submission(use_llm=args.use_llm))
