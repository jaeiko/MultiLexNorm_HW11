"""Spanish MFR dictionary utilities."""
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
        if re.search(r"[A-Za-z]", s) and re.search(r"[0-9]", s):
            return True
        return s in {"_", ".", ",", ":", ";", "!", "?", "…", "¿", "¡"}

DEFAULT_PATH = Path(__file__).with_name("es_mfr_dictionary.json")


def load_es_mfr_dictionary(path: str | Path = DEFAULT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pairs(dictionary: dict, mode: str = "conservative") -> dict[str, str]:
    if mode == "balanced":
        return dict(dictionary.get("balanced_pairs_for_analysis", {}))
    return dict(dictionary.get("high_confidence_pairs", {}))


def is_context_sensitive(raw: str, dictionary: dict) -> bool:
    return str(raw).lower() in set(dictionary.get("context_sensitive_tokens", []))


def apply_es_mfr_to_tokens(tokens: Sequence[str], dictionary: dict, *, mode: str = "conservative", skip_context_sensitive: bool = True) -> list[str]:
    pairs = get_pairs(dictionary, mode=mode)
    context_sensitive = set(dictionary.get("context_sensitive_tokens", []))
    out: list[str] = []
    for tok in tokens:
        raw = str(tok)
        if is_protected_token(raw):
            out.append(raw)
            continue
        if skip_context_sensitive and raw.lower() in context_sensitive and raw not in pairs:
            out.append(raw)
            continue
        out.append(pairs.get(raw, raw))
    return out


def repeated_letter_candidate(tok: str) -> bool:
    return bool(re.search(r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])\1{2,}", str(tok)))


def missing_accent_candidate(tok: str) -> bool:
    low = str(tok).lower()
    return low in {
        "tambien", "despues", "aqui", "alli", "corazon", "cafe", "mio", "mia", "tio", "pais", "estan", "habra", "tendre", "muchisimo"
    }
