"""Italian MFR dictionary utilities."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

try:
    from prompts.common_prompt import is_protected_token
except Exception:  # pragma: no cover
    def is_protected_token(tok: str) -> bool:
        s = str(tok).strip()
        if not s:
            return True
        if s.startswith("#") or s.startswith("@"):
            return True
        if re.fullmatch(r"[0-9]+(?:[.:/\-][0-9]+)*", s):
            return True
        if re.search(r"[A-Za-zÀ-ÿ]", s) and re.search(r"[0-9]", s):
            return True
        return False

DEFAULT_DICTIONARY_PATH = Path(__file__).with_name("it_mfr_dictionary.json")


def load_it_mfr_dictionary(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_DICTIONARY_PATH
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_replacement_pairs(dictionary: dict, mode: str = "conservative") -> dict[str, str]:
    if mode == "conservative":
        return dict(dictionary.get("high_confidence_pairs", {}))
    if mode == "balanced":
        return dict(dictionary.get("balanced_pairs_for_analysis", {}))
    raise ValueError("mode must be 'conservative' or 'balanced'")


def apply_it_mfr_to_tokens(tokens: Sequence[str], dictionary: dict, *, mode: str = "conservative") -> list[str]:
    pairs = get_replacement_pairs(dictionary, mode=mode)
    out: list[str] = []
    for tok in tokens:
        s = str(tok)
        if is_protected_token(s):
            out.append(s)
        else:
            out.append(pairs.get(s, s))
    return out


def is_context_sensitive_token(tok: str, dictionary: dict) -> bool:
    return str(tok) in set(dictionary.get("context_sensitive_tokens", [])) or str(tok).lower() in set(dictionary.get("context_sensitive_tokens", []))


def maybe_accent_or_apostrophe_candidate(tok: str) -> bool:
    """Heuristic helper for prompt candidate selection, not direct replacement."""
    s = str(tok).strip().lower()
    if not s:
        return False
    if s.endswith("'") or s.endswith("’"):
        return True
    if s in {"perche", "perchè", "perche'", "puo", "puo'", "pero", "pero'", "gia", "cosi", "piu"}:
        return True
    return False
