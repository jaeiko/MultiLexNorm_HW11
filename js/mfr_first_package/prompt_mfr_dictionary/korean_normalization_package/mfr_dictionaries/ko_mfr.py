"""Utilities for Korean MFR dictionaries.

Apply only high_confidence_pairs automatically. Korean social-media text often
contains intentional noisy forms, community jargon, profanity avoidance, and
expressive particles; many such forms should be sent to context-aware prompt or
model fallback rather than direct replacement.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

KOREAN_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
LAUGHTER_EMOTION_RE = re.compile(r"^(?:[ㅋㅎ]{1,}|[ㅠㅜ]+|[ㅇㄷㄱㅅㅊㅉㅍㅌㅁㅊ]{1,3})$")
REPEATED_KO_RE = re.compile(r"([가-힣ㄱ-ㅎㅏ-ㅣ])\1{2,}")


def load_ko_mfr_dictionary(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_high_confidence_pairs(dictionary: dict) -> dict[str, str]:
    return dict(dictionary.get("high_confidence_pairs", {}))


def is_ambiguous_token(token: str, dictionary: dict) -> bool:
    return token in dictionary.get("ambiguous_pairs", {}) or token in dictionary.get("review_pairs", {})


def is_korean_token(token: str) -> bool:
    return bool(KOREAN_RE.search(token or ""))


def is_laughter_or_emotion_token(token: str) -> bool:
    return bool(LAUGHTER_EMOTION_RE.fullmatch(token or ""))


def reduce_repeated_korean_characters(token: str, *, max_repeats: int = 2) -> str:
    """Reduce long repeated Korean characters.

    This is a helper, not a safe final normalizer. Repetition can encode emotion
    or laughter and should only be used in ablations or candidate generation.
    """
    def repl(match: re.Match[str]) -> str:
        return match.group(1) * max_repeats
    return REPEATED_KO_RE.sub(repl, token)


def apply_ko_mfr_to_tokens(
    tokens: Sequence[str],
    dictionary: dict,
    *,
    preserve_ambiguous: bool = True,
) -> list[str]:
    """Apply safe Korean high-confidence MFR replacements only."""
    high = get_high_confidence_pairs(dictionary)
    out: list[str] = []
    for tok in tokens:
        if is_laughter_or_emotion_token(tok) and tok not in high:
            out.append(tok)
            continue
        if preserve_ambiguous and is_ambiguous_token(tok, dictionary) and tok not in high:
            out.append(tok)
            continue
        out.append(high.get(tok, tok))
    return out


def changed_indices_after_mfr(tokens: Sequence[str], normalized: Sequence[str]) -> list[int]:
    return [i for i, (raw, norm) in enumerate(zip(tokens, normalized)) if raw != norm]


__all__ = [
    "load_ko_mfr_dictionary",
    "get_high_confidence_pairs",
    "is_ambiguous_token",
    "is_korean_token",
    "is_laughter_or_emotion_token",
    "reduce_repeated_korean_characters",
    "apply_ko_mfr_to_tokens",
    "changed_indices_after_mfr",
]
