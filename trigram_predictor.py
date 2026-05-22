#!/usr/bin/env python3
"""
trigram_predictor.py

Trigram fallback-chain predictor.

config:
    variant   : 'pure' | 'tri_biL' | 'tri_biR' | 'tri_bi_both'
    conf_min  : float (default 0.8)
    min_total : int (default 2)
    use_protect: bool (default True)

Output:
    preds      : list[list[str]] — same shape as samples' raw
    hit_stats  : dict — coverage counters per source
"""

import sys
from pathlib import Path

# protect 로직은 smart_guard_mfr_v2에서 가져오려 했으나
# trigram은 강한 protect 시 성능이 떨어져 의도적으로 simple 버전을 쓴다.
_PKG_DIR = Path(__file__).parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
from smart_guard_mfr_v2 import find_protected_indices_simple  # noqa: E402

BOS = '<BOS>'
EOS = '<EOS>'


def _best_from_counter(counter, total: int, conf_min: float, min_total: int):
    """Returns the top normalization token if it passes thresholds.

    Args:
        counter: Counter object containing normalization counts.
        total: Total count of occurrences.
        conf_min: Minimum confidence threshold.
        min_total: Minimum total count.

    Returns:
        The normalized token string if thresholds are met; otherwise, None.
    """
    if total < min_total:
        return None
    top_norm, top_count = counter.most_common(1)[0]
    if (top_count / total) < conf_min:
        return None
    return top_norm


def _lookup_level(level_dict, key, conf_min: float, min_total: int):
    """Look up a single context level (tri/biL/biR) to find the best candidate.

    Args:
        level_dict: Count dictionary for the level.
        key: Context key tuple.
        conf_min: Minimum confidence threshold.
        min_total: Minimum total count.

    Returns:
        A tuple of (normalized_token_or_None, total_occurrences).
    """
    counter = level_dict.get(key)
    if counter is None:
        return None, 0
    total = sum(counter.values())
    norm = _best_from_counter(counter, total, conf_min, min_total)
    if norm is None:
        return None, total
    return norm, total


def predict_trigram(samples, stats, config) -> tuple[list[list[str]], dict[str, int]]:
    """Executes trigram and bigram fallback-chain predictions.

    Args:
        samples: A list of sample dicts containing raw sentence tokens and language metadata.
        stats: Precomputed trigram, left-bigram, and right-bigram counts.
        config: Configurations dictionary:
            - 'variant' (str): One of 'pure', 'tri_biL', 'tri_biR', or 'tri_bi_both'.
            - 'conf_min' (float): Minimum confidence threshold.
            - 'min_total' (int): Minimum count of occurrences required.
            - 'use_protect' (bool): Whether to protect certain tokens from modification.

    Returns:
        A tuple containing:
            - list[list[str]]: Padded/normalized token predictions matching the input shape.
            - dict[str, int]: Coverage and hit counters for analysis.
    """
    variant = config.get('variant', 'pure')
    conf_min = float(config.get('conf_min', 0.8))
    min_total = int(config.get('min_total', 2))
    use_protect = bool(config.get('use_protect', True))

    use_biL = variant in ('tri_biL', 'tri_bi_both')
    use_biR = variant in ('tri_biR', 'tri_bi_both')

    hit = {
        'tri_hit': 0, 'biL_hit': 0, 'biR_hit': 0,
        'no_match': 0, 'protected': 0, 'total': 0,
        'changed': 0,
    }

    preds_out = []
    for s in samples:
        lang = str(s['lang'])
        raw = [str(t) for t in s['raw']]
        L = len(raw)
        lang_stats = stats.get(lang)

        if not lang_stats:
            preds_out.append(list(raw))
            hit['no_match'] += L
            hit['total'] += L
            continue

        protected = set(find_protected_indices_simple(raw)) if use_protect else set()
        padded = [BOS] + raw + [EOS]
        tri_d = lang_stats.get('tri', {})
        biL_d = lang_stats.get('biL', {})
        biR_d = lang_stats.get('biR', {})

        preds = []
        for i in range(L):
            tok = raw[i]
            hit['total'] += 1
            if i in protected:
                preds.append(tok)
                hit['protected'] += 1
                continue

            prev = padded[i]
            nxt = padded[i + 2]

            # Level 1: trigram
            norm, _ = _lookup_level(tri_d, (prev, tok, nxt), conf_min, min_total)
            if norm is not None:
                hit['tri_hit'] += 1
                if norm != tok:
                    hit['changed'] += 1
                preds.append(norm)
                continue

            # Level 2: bigrams (depending on variant)
            biL_norm, biL_total = (None, 0)
            biR_norm, biR_total = (None, 0)
            if use_biL:
                biL_norm, biL_total = _lookup_level(biL_d, (prev, tok), conf_min, min_total)
            if use_biR:
                biR_norm, biR_total = _lookup_level(biR_d, (tok, nxt), conf_min, min_total)

            chosen = None
            chosen_src = None
            if biL_norm is not None and biR_norm is not None:
                # Both hit -> take the one with higher support count
                if biL_total >= biR_total:
                    chosen, chosen_src = biL_norm, 'biL_hit'
                else:
                    chosen, chosen_src = biR_norm, 'biR_hit'
            elif biL_norm is not None:
                chosen, chosen_src = biL_norm, 'biL_hit'
            elif biR_norm is not None:
                chosen, chosen_src = biR_norm, 'biR_hit'

            if chosen is not None:
                hit[chosen_src] += 1
                if chosen != tok:
                    hit['changed'] += 1
                preds.append(chosen)
            else:
                hit['no_match'] += 1
                preds.append(tok)

        preds_out.append(preds)

    return preds_out, hit
