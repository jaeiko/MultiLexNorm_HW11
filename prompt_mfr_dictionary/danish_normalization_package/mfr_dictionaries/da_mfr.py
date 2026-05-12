"""MFR utilities for Danish lexical normalization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

try:
    from prompts.common_prompt import is_protected_token
except ImportError:
    from ..prompts.common_prompt import is_protected_token

_DEFAULT_PATH = Path(__file__).with_name("da_mfr_dictionary.json")


def load_da_mfr_dictionary(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else _DEFAULT_PATH
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_replacement_pairs(dictionary: dict, mode: str = "conservative") -> dict[str, str]:
    if mode == "conservative":
        return dict(dictionary.get("high_confidence_pairs", {}))
    if mode == "balanced":
        return dict(dictionary.get("balanced_pairs_for_analysis", {}))
    raise ValueError("mode must be 'conservative' or 'balanced'")


def apply_da_mfr_to_tokens(tokens: Sequence[str], dictionary: dict | None = None, *, mode: str = "conservative") -> list[str]:
    d = dictionary or load_da_mfr_dictionary()
    pairs = get_replacement_pairs(d, mode=mode)
    out: list[str] = []
    for tok in tokens:
        s = str(tok)
        if is_protected_token(s):
            out.append(s)
        else:
            out.append(pairs.get(s, s))
    return out


def is_da_ambiguous_token(token: str, dictionary: dict | None = None) -> bool:
    d = dictionary or load_da_mfr_dictionary()
    s = str(token)
    return s in d.get("ambiguous_pairs", {}) or s.lower() in set(d.get("context_sensitive_tokens", []))


def get_da_candidates_for_token(token: str, dictionary: dict | None = None) -> dict:
    d = dictionary or load_da_mfr_dictionary()
    s = str(token)
    return d.get("ambiguous_pairs", {}).get(s) or d.get("review_pairs", {}).get(s) or {}
