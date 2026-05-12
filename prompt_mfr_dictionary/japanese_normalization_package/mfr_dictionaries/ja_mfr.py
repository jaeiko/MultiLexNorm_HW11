"""Utilities for the Japanese MFR dictionary.

The JSON dictionary intentionally separates direct high-confidence pairs from
review/ambiguous pairs. Apply only high_confidence_pairs automatically; send
review/ambiguous tokens to a context-aware detector or model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def load_ja_mfr_dictionary(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_high_confidence_pairs(dictionary: dict) -> dict[str, str]:
    return dict(dictionary.get("high_confidence_pairs", {}))


def is_ambiguous_token(token: str, dictionary: dict) -> bool:
    return token in dictionary.get("ambiguous_pairs", {}) or token in dictionary.get("review_pairs", {})


def apply_ja_mfr_to_tokens(tokens: Sequence[str], dictionary: dict, *, preserve_ambiguous: bool = True) -> list[str]:
    high_conf = get_high_confidence_pairs(dictionary)
    out: list[str] = []
    for tok in tokens:
        if preserve_ambiguous and is_ambiguous_token(tok, dictionary):
            out.append(tok)
        else:
            out.append(high_conf.get(tok, tok))
    return out


def changed_indices_after_mfr(tokens: Sequence[str], normalized: Sequence[str]) -> list[int]:
    return [i for i, (raw, norm) in enumerate(zip(tokens, normalized)) if raw != norm]


__all__ = [
    "load_ja_mfr_dictionary",
    "get_high_confidence_pairs",
    "is_ambiguous_token",
    "apply_ja_mfr_to_tokens",
    "changed_indices_after_mfr",
]
