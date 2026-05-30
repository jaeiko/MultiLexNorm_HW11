"""Utilities for Thai MFR dictionaries.

Apply only high_confidence_pairs automatically. Repetition, tone/vowel
variation, homophonic transformations, and transliteration/OOV cases should be
sent to context-aware prompt/model fallback unless they appear in the direct MFR.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")
THAI_REPEAT_RE = re.compile(r"([\u0E00-\u0E7F])\1{1,}")
LAUGHTER_RE = re.compile(r"^(?:5{2,}|๕{2,}|ฮา(?:ฮา)+)$")


def load_th_mfr_dictionary(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_high_confidence_pairs(dictionary: dict) -> dict[str, str]:
    return dict(dictionary.get("high_confidence_pairs", {}))


def is_ambiguous_token(token: str, dictionary: dict) -> bool:
    return token in dictionary.get("ambiguous_pairs", {}) or token in dictionary.get("review_pairs", {})


def is_thai_token(token: str) -> bool:
    return bool(THAI_CHAR_RE.search(token or ""))


def is_laughter_token(token: str) -> bool:
    return bool(LAUGHTER_RE.fullmatch(token or ""))


def reduce_repeated_thai_characters(token: str, *, max_repeats: int = 1) -> str:
    """Reduce repeated Thai characters.

    This is a helper, not a safe final normalizer. Thai repetition can encode
    sentiment/emphasis, so direct use should be validated by dictionary/context.
    """
    def repl(match: re.Match[str]) -> str:
        return match.group(1) * max_repeats
    return THAI_REPEAT_RE.sub(repl, token)


def normalize_thai_repetition_marker_spacing(token: str) -> str | None:
    """Return a spaced form for Thai repetition marker ๆ when obvious.

    Examples: จริงๆ -> จริง ๆ, มากๆ -> มาก ๆ.
    This helper is conservative and returns None unless the token contains exactly
    one adjacent repetition marker at the end of a Thai string.
    """
    if re.fullmatch(r"([\u0E00-\u0E7F]+)ๆ", token or ""):
        return token[:-1] + " ๆ"
    return None


def apply_th_mfr_to_tokens(
    tokens: Sequence[str],
    dictionary: dict,
    *,
    preserve_ambiguous: bool = True,
    use_repetition_bridge: bool = False,
) -> list[str]:
    """Apply safe Thai high-confidence MFR replacements.

    By default, this does not invent new repeated-character normalizations. It
    only applies exact high-confidence pairs observed in training data. Set
    use_repetition_bridge=True only in an ablation if you want to test reducing
    an unseen repeated form to a seen high-confidence key.
    """
    high = get_high_confidence_pairs(dictionary)
    out: list[str] = []
    for tok in tokens:
        if is_laughter_token(tok):
            out.append(tok)
            continue
        if preserve_ambiguous and is_ambiguous_token(tok, dictionary) and tok not in high:
            out.append(tok)
            continue
        if tok in high:
            out.append(high[tok])
            continue
        if use_repetition_bridge and is_thai_token(tok):
            reduced = reduce_repeated_thai_characters(tok)
            if reduced in high:
                out.append(high[reduced])
                continue
        out.append(tok)
    return out


def changed_indices_after_mfr(tokens: Sequence[str], normalized: Sequence[str]) -> list[int]:
    return [i for i, (raw, norm) in enumerate(zip(tokens, normalized)) if raw != norm]


__all__ = [
    "load_th_mfr_dictionary",
    "get_high_confidence_pairs",
    "is_ambiguous_token",
    "is_thai_token",
    "is_laughter_token",
    "reduce_repeated_thai_characters",
    "normalize_thai_repetition_marker_spacing",
    "apply_th_mfr_to_tokens",
    "changed_indices_after_mfr",
]
