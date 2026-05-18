"""Utilities for Indonesian/id-en MFR dictionaries.

The JSON dictionary separates direct high-confidence pairs from review/ambiguous
pairs. Apply only high_confidence_pairs automatically; send review/ambiguous
forms to a context-aware prompt/model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

SUPPORTED_LANGS = {"id", "iden"}


def _validate_lang(lang: str) -> str:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported lang={lang!r}; expected one of {sorted(SUPPORTED_LANGS)}")
    return lang


def load_id_mfr_dictionary(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_lang_section(dictionary: dict, lang: str = "id") -> dict:
    lang = _validate_lang(lang)
    return dictionary.get("languages", {}).get(lang, {})


def get_high_confidence_pairs(dictionary: dict, lang: str = "id") -> dict[str, str]:
    return dict(get_lang_section(dictionary, lang).get("high_confidence_pairs", {}))


def is_ambiguous_token(token: str, dictionary: dict, lang: str = "id") -> bool:
    section = get_lang_section(dictionary, lang)
    return token in section.get("ambiguous_pairs", {}) or token in section.get("review_pairs", {})


def normalize_mention(token: str) -> str | None:
    if token.startswith("@") and len(token) > 1:
        return "[mention]"
    return None


def reduce_repeated_characters(token: str, *, max_repeats: int = 2) -> str:
    """Reduce alphabetic character runs to at most max_repeats.

    This is a pre-normalization helper, not necessarily the final normalized form.
    """
    def repl(match: re.Match[str]) -> str:
        ch = match.group(1)
        return ch * max_repeats
    return re.sub(r"([A-Za-z])\1{%d,}" % max_repeats, repl, token)


def reduplication_marker_candidate(token: str) -> str | None:
    """Conservative reduplication-marker expansion.

    Examples:
    - anak2 -> anak-anak
    - anak2nya -> anak-anaknya

    This is intentionally conservative; prefixed forms like bersama2 can be wrong
    if naively expanded, so callers should validate with dictionary/context.
    """
    m = re.fullmatch(r"([A-Za-z]{3,})(?:2|['\"]{1,2})([A-Za-z]*)", token)
    if not m:
        return None
    base, suffix = m.group(1), m.group(2)
    if base.lower().startswith(("ber", "ter", "mem", "men", "meng", "pem", "pen")):
        return None
    return f"{base}-{base}{suffix}"


def apply_id_mfr_to_tokens(
    tokens: Sequence[str],
    dictionary: dict,
    *,
    lang: str = "id",
    preserve_ambiguous: bool = True,
    normalize_mentions_policy: bool = True,
    use_repetition_bridge: bool = True,
) -> list[str]:
    """Apply safe high-confidence MFR replacements.

    The optional repetition bridge tries a repeated-character-reduced version only
    when that reduced form itself appears in high_confidence_pairs. This keeps the
    rule data-driven rather than free-form.
    """
    lang = _validate_lang(lang)
    high_conf = get_high_confidence_pairs(dictionary, lang)
    out: list[str] = []
    for tok in tokens:
        if normalize_mentions_policy and lang == "id":
            mention = normalize_mention(tok)
            if mention is not None:
                out.append(mention)
                continue
        if preserve_ambiguous and is_ambiguous_token(tok, dictionary, lang):
            out.append(tok)
            continue
        if tok in high_conf:
            out.append(high_conf[tok])
            continue
        if use_repetition_bridge:
            reduced = reduce_repeated_characters(tok)
            if reduced in high_conf:
                out.append(high_conf[reduced])
                continue
        out.append(tok)
    return out


def changed_indices_after_mfr(tokens: Sequence[str], normalized: Sequence[str]) -> list[int]:
    return [i for i, (raw, norm) in enumerate(zip(tokens, normalized)) if raw != norm]


__all__ = [
    "load_id_mfr_dictionary",
    "get_lang_section",
    "get_high_confidence_pairs",
    "is_ambiguous_token",
    "normalize_mention",
    "reduce_repeated_characters",
    "reduplication_marker_candidate",
    "apply_id_mfr_to_tokens",
    "changed_indices_after_mfr",
]
