"""MFR utilities for Turkish and Turkish-German normalization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

try:
    from prompts.common_prompt import is_protected_token, enforce_protected_output
except Exception:  # pragma: no cover
    from ..prompts.common_prompt import is_protected_token, enforce_protected_output

PACKAGE_DIR = Path(__file__).resolve().parent
CONTEXT_SENSITIVE_BY_LANG = {
    "tr": {"mi", "mı", "mu", "mü", "de", "da", "ki", "ne", "bu", "o", "a", "e", "i", "ı", "İ", "t", "k", "ben", "sen", "allah", "Allah"},
    "trde": {"mi", "mı", "de", "da", "ne", "i", "e", "o", "a", "oda", "party", "u", "foto", "schule", "theorie", "bahn", "DAS", "das", "ben", "ich", "hab", "ma", "mal", "bende", "sende"},
}


def load_turkish_mfr_dictionary(lang: str, path: str | Path | None = None) -> dict:
    if lang not in {"tr", "trde"}:
        raise ValueError("lang must be 'tr' or 'trde'")
    if path is None:
        path = PACKAGE_DIR / f"{lang}_mfr_dictionary.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def high_confidence_map(dictionary: dict) -> dict[str, str]:
    return {raw: entry["norm"] for raw, entry in dictionary.get("high_confidence_pairs", {}).items()}


def is_context_sensitive(raw: str, lang: str, dictionary: dict | None = None) -> bool:
    if raw in CONTEXT_SENSITIVE_BY_LANG.get(lang, set()):
        return True
    if dictionary and raw in set(dictionary.get("context_sensitive_tokens", [])):
        return True
    return False


def apply_turkish_mfr_to_tokens(tokens: Sequence[str], dictionary: dict, *, lang: str, skip_context_sensitive: bool = True) -> list[str]:
    mapping = high_confidence_map(dictionary)
    output: list[str] = []
    for tok in tokens:
        raw = str(tok)
        if is_protected_token(raw):
            output.append(raw)
            continue
        if skip_context_sensitive and is_context_sensitive(raw, lang, dictionary):
            output.append(raw)
            continue
        pred = mapping.get(raw, raw)
        output.append(enforce_protected_output(raw, pred))
    return output


def get_ambiguous_tokens(dictionary: dict) -> set[str]:
    return set(dictionary.get("ambiguous_pairs", {}).keys())


def get_review_tokens(dictionary: dict) -> set[str]:
    return set(dictionary.get("review_pairs", {}).keys())

# Aliases for simple per-language imports.
def load_tr_mfr_dictionary(path: str | Path | None = None) -> dict:
    return load_turkish_mfr_dictionary("tr", path)


def load_trde_mfr_dictionary(path: str | Path | None = None) -> dict:
    return load_turkish_mfr_dictionary("trde", path)


def apply_tr_mfr_to_tokens(tokens: Sequence[str], dictionary: dict, *, skip_context_sensitive: bool = True) -> list[str]:
    return apply_turkish_mfr_to_tokens(tokens, dictionary, lang="tr", skip_context_sensitive=skip_context_sensitive)


def apply_trde_mfr_to_tokens(tokens: Sequence[str], dictionary: dict, *, skip_context_sensitive: bool = True) -> list[str]:
    return apply_turkish_mfr_to_tokens(tokens, dictionary, lang="trde", skip_context_sensitive=skip_context_sensitive)
