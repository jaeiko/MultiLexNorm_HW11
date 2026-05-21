"""Japanese MFR utilities for the v2 boundary-aware normalization package.

Production policy:
- Use `high_confidence_pairs` only in conservative mode.
- Do not directly replace context-sensitive, pronunciation-sensitive, protected,
  or named-entity-like tokens.
- Use balanced mode only for ablation, never as a default production setting.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

DEFAULT_DICT_PATH = Path(__file__).with_name("ja_mfr_dictionary.json")

PROTECTED_RE = re.compile(
    r"^(#|@|https?://|www\.|[\d０-９]+$|[\d０-９]+[:：][\d０-９]+|[_＿]+$)"
)


def load_ja_mfr_dictionary(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_DICT_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_protected_token(token: str) -> bool:
    if not token:
        return True
    if PROTECTED_RE.search(token):
        return True
    if re.fullmatch(r"[\W_]+", token) and token not in {"〜", "～", "…"}:
        return True
    return False


def get_pairs(dictionary: dict, mode: str = "conservative") -> dict[str, str]:
    if mode == "conservative":
        return dict(dictionary.get("high_confidence_pairs", {}))
    if mode == "balanced":
        pairs = dict(dictionary.get("high_confidence_pairs", {}))
        pairs.update(dictionary.get("balanced_pairs_for_analysis", {}))
        return pairs
    raise ValueError("mode must be 'conservative' or 'balanced'")


def is_context_sensitive_token(token: str, dictionary: dict) -> bool:
    return token in set(dictionary.get("context_sensitive_tokens", []))


def is_ambiguous_token(token: str, dictionary: dict) -> bool:
    return (
        token in dictionary.get("ambiguous_pairs", {})
        or token in dictionary.get("review_pairs", {})
        or is_context_sensitive_token(token, dictionary)
    )


def apply_ja_mfr_to_tokens(
    tokens: Sequence[str],
    dictionary: dict,
    *,
    mode: str = "conservative",
    preserve_ambiguous: bool = True,
    preserve_protected: bool = True,
) -> list[str]:
    pairs = get_pairs(dictionary, mode=mode)
    out: list[str] = []
    for tok in tokens:
        if preserve_protected and is_protected_token(tok):
            out.append(tok)
            continue
        if preserve_ambiguous and is_ambiguous_token(tok, dictionary):
            # If a pair is explicitly high-confidence, allow it in conservative mode;
            # otherwise keep it for target prompt/model fallback.
            if mode == "conservative" and tok in dictionary.get("high_confidence_pairs", {}):
                out.append(pairs.get(tok, tok))
            else:
                out.append(tok)
            continue
        out.append(pairs.get(tok, tok))
    return out


def changed_indices_after_mfr(tokens: Sequence[str], normalized: Sequence[str]) -> list[int]:
    return [i for i, (raw, norm) in enumerate(zip(tokens, normalized)) if raw != norm]


__all__ = [
    "load_ja_mfr_dictionary",
    "is_protected_token",
    "get_pairs",
    "is_context_sensitive_token",
    "is_ambiguous_token",
    "apply_ja_mfr_to_tokens",
    "changed_indices_after_mfr",
]
