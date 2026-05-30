#!/usr/bin/env python3
"""
build_trigram_stats.py

Train parquet (dataset_12lang)에서 언어별 (prev, tok, next) trigram + bigram + unigram 통계를 한 번에 빌드.
출력: outputs/trigram_stats.pkl.gz

자료구조:
    stats[lang] = {
        'tri': {(prev, tok, next): Counter({norm: count, ...})},
        'biL': {(prev, tok):       Counter({norm: count, ...})},
        'biR': {(tok, next):       Counter({norm: count, ...})},
        'uni': {tok:               Counter({norm: count, ...})},
    }

문장 경계 sentinel: '<BOS>', '<EOS>'
"""

import sys
import gzip
import pickle
import time
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BOS = '<BOS>'
EOS = '<EOS>'


def to_list(value):
    if hasattr(value, 'tolist'):
        return [str(v) for v in value.tolist()]
    if isinstance(value, str):
        import ast
        value = ast.literal_eval(value)
    return [str(v) for v in value]


def build_stats(train_df):
    stats = defaultdict(lambda: {
        'tri': defaultdict(Counter),
        'biL': defaultdict(Counter),
        'biR': defaultdict(Counter),
    })

    n_rows = len(train_df)
    n_tokens = 0
    n_mismatch = 0
    t0 = time.time()

    for idx, row in enumerate(train_df.itertuples(index=False)):
        raw = to_list(row.raw)
        norm = to_list(row.norm)
        lang = str(row.lang)

        if len(raw) != len(norm):
            n_mismatch += 1
            continue

        padded = [BOS] + raw + [EOS]
        L = len(raw)
        d = stats[lang]
        for i in range(L):
            tok = raw[i]
            nrm = norm[i]
            prev = padded[i]      # padded[i] == raw[i-1] when i>0 else BOS
            nxt = padded[i + 2]   # padded[i+2] == raw[i+1] when i<L-1 else EOS
            d['tri'][(prev, tok, nxt)][nrm] += 1
            d['biL'][(prev, tok)][nrm] += 1
            d['biR'][(tok, nxt)][nrm] += 1
            n_tokens += 1

        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1:>6d}/{n_rows}] tokens={n_tokens:,} elapsed={elapsed:.1f}s")

    print(f"\n[Done] rows={n_rows} tokens={n_tokens:,} mismatch={n_mismatch} elapsed={time.time()-t0:.1f}s")

    # Convert defaultdict -> dict for cleaner pickle
    stats_out = {}
    for lang, d in stats.items():
        stats_out[lang] = {
            'tri': dict(d['tri']),
            'biL': dict(d['biL']),
            'biR': dict(d['biR']),
        }
    return stats_out


def report_size(stats):
    print("\n=== Per-language stats summary ===")
    print(f"{'lang':<6} {'tri':>10} {'biL':>10} {'biR':>10}")
    for lang in sorted(stats.keys()):
        d = stats[lang]
        print(f"{lang:<6} {len(d['tri']):>10,} {len(d['biL']):>10,} {len(d['biR']):>10,}")


def main():
    repo = Path(__file__).parent.resolve()  # MultiLexNorm_HW11
    train_path = repo / "multilexnorm2026-dataset" / "train-00000-of-00001.parquet"

    print(f"[Build trigram stats] reading {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"  rows={len(train_df)} langs={sorted(train_df['lang'].unique())}")

    stats = build_stats(train_df)
    report_size(stats)

    out_dir = repo / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trigram_stats.pkl.gz"

    print(f"\n[Saving] {out_path}")
    t0 = time.time()
    with gzip.open(out_path, "wb", compresslevel=6) as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  saved ({time.time()-t0:.1f}s) size={size_mb:.1f} MB")

    if size_mb > 2048:
        print(f"  WARNING: exceeded 2GB ({size_mb:.1f} MB). Consider pruning.")
    else:
        print(f"  OK: under 2GB threshold.")

    # Sanity: a couple of trigram lookups for 'ja' if exists
    if 'ja' in stats:
        ja = stats['ja']['tri']
        print(f"\n[Sanity-ja-tri] unique keys={len(ja)}, sample 5:")
        for k in list(ja.keys())[:5]:
            print(f"    {k!r} -> top {ja[k].most_common(3)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
