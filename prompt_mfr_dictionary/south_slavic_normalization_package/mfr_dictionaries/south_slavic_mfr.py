"""MFR utilities for South Slavic lexical normalization (sr/hr/sl)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

try:
    from prompts.common_prompt import is_protected_token
except ImportError:  # pragma: no cover
    from common_prompt import is_protected_token  # type: ignore

SUPPORTED_LANGS = {"sr", "hr", "sl"}


def load_mfr_dictionary(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def high_confidence_map(dictionary: dict, *, skip_context_sensitive: bool = False) -> dict[str, str]:
    pairs = dictionary.get("high_confidence_pairs", [])
    out: dict[str, str] = {}
    for item in pairs:
        if skip_context_sensitive and item.get("context_sensitive"):
            continue
        out[item["raw"]] = item["norm"]
    return out


def known_context_sensitive_tokens(dictionary: dict) -> set[str]:
    return set(dictionary.get("statistics", {}).get("known_context_sensitive_tokens", []))


def apply_mfr_to_tokens(
    tokens: Sequence[str],
    dictionary: dict,
    *,
    skip_context_sensitive: bool = False,
    protect_tokens: bool = True,
) -> list[str]:
    """Apply high-confidence MFR replacements.

    skip_context_sensitive=False gives the strongest MFR-only baseline.
    skip_context_sensitive=True is safer when a prompt/model will handle ambiguous tokens.
    Protected tokens are always preserved when protect_tokens=True.
    """
    mapping = high_confidence_map(dictionary, skip_context_sensitive=skip_context_sensitive)
    output: list[str] = []
    for tok in tokens:
        if protect_tokens and is_protected_token(str(tok)):
            output.append(str(tok))
        else:
            output.append(mapping.get(str(tok), str(tok)))
    return output


def is_known_ambiguous(token: str, dictionary: dict) -> bool:
    t = str(token)
    if t.lower() in known_context_sensitive_tokens(dictionary):
        return True
    for item in dictionary.get("ambiguous_pairs", []):
        if item.get("raw") == t:
            return True
    return False


__all__ = [
    "SUPPORTED_LANGS", "load_mfr_dictionary", "high_confidence_map", "known_context_sensitive_tokens",
    "apply_mfr_to_tokens", "is_known_ambiguous",
]
