"""Utilities for Korean MFR dictionaries.

Version: ko_v2_paper_informed

Only high_confidence_pairs should be used automatically in production. Korean
slang/profanity/meme forms are often context-sensitive; use prompt/model fallback
for review_pairs, ambiguous_pairs, and context_sensitive_tokens.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

try:
    from prompts.common_prompt import is_protected_token, safe_normalization_result
except Exception:  # pragma: no cover
    is_protected_token = None
    safe_normalization_result = None

KOREAN_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
LAUGHTER_EMOTION_RE = re.compile(r"^(?:[ㅋㅎ]{1,}|[ㅠㅜ]+|(?:ㅇㅇ|ㄴㄴ|ㄷㄷ|ㄱㄱ|ㅊㅋ|ㄱㅅ|ㅇㅋ|ㅇㄷ|ㅁㅊ|ㅉㅉ|ㅍㅌㅊ))$")
REPEATED_KO_RE = re.compile(r"([가-힣ㄱ-ㅎㅏ-ㅣ])\1{2,}")

DEFAULT_DICT_PATH = Path(__file__).with_name("ko_mfr_dictionary.json")


def load_ko_mfr_dictionary(path: str | Path | None = None) -> dict:
    p = DEFAULT_DICT_PATH if path is None else Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_pairs(dictionary: dict, *, mode: str = "conservative") -> dict[str, str]:
    if mode in {"conservative", "guarded", "production"}:
        return dict(dictionary.get("high_confidence_pairs", {}))
    if mode in {"balanced", "analysis"}:
        pairs = dict(dictionary.get("balanced_pairs_for_analysis", {}))
        pairs.update(dictionary.get("high_confidence_pairs", {}))
        return pairs
    if mode in {"none", "off"}:
        return {}
    raise ValueError(f"Unknown Korean MFR mode: {mode}")


def get_high_confidence_pairs(dictionary: dict) -> dict[str, str]:
    return dict(dictionary.get("high_confidence_pairs", {}))


def is_ambiguous_token(token: str, dictionary: dict) -> bool:
    return (
        token in dictionary.get("ambiguous_pairs", {})
        or token in dictionary.get("review_pairs", {})
        or token in set(dictionary.get("context_sensitive_tokens", []))
    )


def is_korean_token(token: str) -> bool:
    return bool(KOREAN_RE.search(token or ""))


def is_laughter_or_emotion_token(token: str) -> bool:
    return bool(LAUGHTER_EMOTION_RE.fullmatch(token or ""))


def reduce_repeated_korean_characters(token: str, *, max_repeats: int = 2) -> str:
    """Helper for ablation/candidate generation, not a safe final normalizer."""
    def repl(match: re.Match[str]) -> str:
        return match.group(1) * max_repeats
    return REPEATED_KO_RE.sub(repl, token)


def _is_protected(tok: str, dictionary: dict) -> bool:
    extra = set(dictionary.get("preserve_by_default", []))
    if is_protected_token is not None:
        return bool(is_protected_token(tok, extra_protected=extra))
    return tok in extra or tok.startswith("#") or tok.startswith("@")


def apply_ko_mfr_to_tokens(
    tokens: Sequence[str],
    dictionary: dict | None = None,
    *,
    mode: str = "conservative",
    preserve_ambiguous: bool = True,
    apply_offensive_sensitive: bool = True,
) -> list[str]:
    """Apply Korean MFR replacements with paper-informed guardrails.

    Parameters
    ----------
    mode:
        conservative/production: high_confidence_pairs only.
        balanced/analysis: includes balanced_pairs_for_analysis; not recommended
        for production without ablation.
    preserve_ambiguous:
        If True, review/ambiguous/context-sensitive tokens are preserved unless
        they are explicitly included in the selected pair set.
    apply_offensive_sensitive:
        If False, offensive/identity-sensitive pairs are preserved for prompt
        resolution instead of direct MFR.
    """
    d = load_ko_mfr_dictionary() if dictionary is None else dictionary
    pairs = get_pairs(d, mode=mode)
    sensitive = set(d.get("offensive_sensitive_pairs", {}).keys())
    out: list[str] = []
    for tok in tokens:
        if _is_protected(tok, d):
            out.append(tok)
            continue
        if is_laughter_or_emotion_token(tok) and tok not in pairs:
            out.append(tok)
            continue
        if preserve_ambiguous and is_ambiguous_token(tok, d) and tok not in pairs:
            out.append(tok)
            continue
        if not apply_offensive_sensitive and tok in sensitive:
            out.append(tok)
            continue
        norm = pairs.get(tok, tok)
        if safe_normalization_result is not None:
            norm = safe_normalization_result(raw_target=tok, normalized=norm, max_expansion_tokens=6)
        out.append(norm)
    return out


def changed_indices_after_mfr(tokens: Sequence[str], normalized: Sequence[str]) -> list[int]:
    return [i for i, (raw, norm) in enumerate(zip(tokens, normalized)) if raw != norm]


__all__ = [
    "load_ko_mfr_dictionary",
    "get_pairs",
    "get_high_confidence_pairs",
    "is_ambiguous_token",
    "is_korean_token",
    "is_laughter_or_emotion_token",
    "reduce_repeated_korean_characters",
    "apply_ko_mfr_to_tokens",
    "changed_indices_after_mfr",
]
