"""Italian v2 MFR dictionary utilities.

The default production behavior is conservative:
- apply only high_confidence_pairs
- skip protected tokens
- skip context-sensitive tokens
- skip entity/acronym-like tokens unless an exact high-confidence safe pair is provided

For experiments, mode="balanced" can be used, but it is intended for ablation only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence, Mapping

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

ENTITY_LIKE_RE = re.compile(r"^(?:[A-Z]{2,}|[A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Za-z]+[0-9]+)$")
ITALIAN_ACRONYM_WHITELIST = {
    "PD", "CGIL", "CISL", "UIL", "RAI", "TG1", "BCE", "UE", "ONU", "USA", "UK", "EU",
}


def load_it_mfr_dictionary(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_DICTIONARY_PATH
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _set_lower(items: Sequence[str] | set[str]) -> set[str]:
    return {str(x).lower() for x in items}


def is_entity_like_token(tok: str) -> bool:
    s = str(tok).strip()
    if not s:
        return False
    if s in ITALIAN_ACRONYM_WHITELIST:
        return True
    if s.startswith("#") or s.startswith("@"):
        return True
    if ENTITY_LIKE_RE.match(s) and len(s) >= 2:
        return True
    # Mixed-case product/brand-like forms: iPhone, eBay, YouTube.
    if re.search(r"[a-zà-ÿ][A-Z]", s):
        return True
    return False


def is_context_sensitive_token(tok: str, dictionary: Mapping) -> bool:
    s = str(tok)
    ctx = set(dictionary.get("context_sensitive_tokens", []))
    return s in ctx or s.lower() in _set_lower(ctx)


def get_replacement_pairs(dictionary: Mapping, mode: str = "conservative") -> dict[str, str]:
    if mode == "conservative":
        return dict(dictionary.get("high_confidence_pairs", {}))
    if mode == "accent_abbrev":
        pairs = dict(dictionary.get("high_confidence_pairs", {}))
        pairs.update(dictionary.get("accent_apostrophe_pairs", {}))
        pairs.update(dictionary.get("abbreviation_pairs", {}))
        return pairs
    if mode == "balanced":
        return dict(dictionary.get("balanced_pairs_for_analysis", {}))
    raise ValueError("mode must be 'conservative', 'accent_abbrev', or 'balanced'")


def apply_it_mfr_to_tokens(
    tokens: Sequence[str],
    dictionary: Mapping,
    *,
    mode: str = "conservative",
    skip_context_sensitive: bool = True,
    skip_entity_like: bool = True,
) -> list[str]:
    pairs = get_replacement_pairs(dictionary, mode=mode)
    out: list[str] = []

    for tok in tokens:
        s = str(tok)
        if is_protected_token(s):
            out.append(s)
            continue
        if skip_context_sensitive and is_context_sensitive_token(s, dictionary):
            out.append(s)
            continue
        if skip_entity_like and is_entity_like_token(s) and s not in pairs:
            out.append(s)
            continue
        out.append(pairs.get(s, s))

    return out


def candidate_reason(tok: str, dictionary: Mapping | None = None) -> str | None:
    """Return a coarse reason why the token should be sent to prompt/model fallback."""
    s = str(tok).strip()
    if not s or is_protected_token(s):
        return None
    low = s.lower()

    if dictionary:
        if s in dictionary.get("accent_apostrophe_pairs", {}) or low in _set_lower(dictionary.get("accent_apostrophe_pairs", {}).keys()):
            return "accent_or_apostrophe"
        if s in dictionary.get("abbreviation_pairs", {}) or low in _set_lower(dictionary.get("abbreviation_pairs", {}).keys()):
            return "abbreviation"
        if is_context_sensitive_token(s, dictionary):
            return "context_sensitive"
        if s in dictionary.get("case_entity_sensitive_pairs", {}) or low in _set_lower(dictionary.get("case_entity_sensitive_pairs", {}).keys()):
            return "case_or_entity_sensitive"

    if re.search(r"['’]$", s):
        return "apostrophe_final"
    if re.search(r"([A-Za-zÀ-ÿ])\1{2,}", s):
        return "repeated_character"
    if len(s) >= 3 and re.fullmatch(r"[A-ZÀÈÉÌÍÒÓÙÚ]+", s) and s not in ITALIAN_ACRONYM_WHITELIST:
        return "all_caps_possible_emphasis"
    if low in {"perche", "perchè", "perche'", "puo", "puo'", "pero", "pero'", "gia", "cosi", "piu", "nn", "ke", "cmq", "cn", "dx", "sx"}:
        return "italian_seed"
    return None


def candidate_indices(tokens: Sequence[str], dictionary: Mapping | None = None) -> list[int]:
    return [i for i, tok in enumerate(tokens) if candidate_reason(str(tok), dictionary) is not None]
