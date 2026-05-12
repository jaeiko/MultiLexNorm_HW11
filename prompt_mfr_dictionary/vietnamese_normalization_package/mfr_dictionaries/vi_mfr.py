"""
Vietnamese MFR lookup helpers.

The JSON dictionary separates:
- high_confidence_pairs: train-derived MFR pairs safe for direct fallback lookup.
- review_pairs: moderately reliable pairs; use with LLM/context or stricter validation.
- ambiguous_pairs: known ambiguous tokens; do not blindly replace.
- paper_seed_pairs: external ViLexNorm-inspired pairs; disabled by default for direct MFR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DICT_PATH = Path(__file__).with_name("vi_mfr_dictionary.json")


def load_vi_mfr_dictionary(path: str | Path = DEFAULT_DICT_PATH) -> dict:
    """Load the Vietnamese MFR dictionary JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lookup_vi_mfr(
    token: str,
    dictionary: dict | None = None,
    *,
    min_count: int = 3,
    min_confidence: float = 0.90,
    allow_review_pairs: bool = False,
    allow_paper_seed_pairs: bool = False,
) -> tuple[str | None, dict | None]:
    """Return a high-confidence Vietnamese normalization candidate.

    Returns (norm, record). If no safe candidate is found, returns (None, None).
    """
    if dictionary is None:
        dictionary = load_vi_mfr_dictionary()

    record = dictionary.get("high_confidence_pairs", {}).get(token)
    if record and record.get("count", 0) >= min_count and record.get("confidence", 0.0) >= min_confidence:
        return record["norm"], record

    if allow_review_pairs:
        record = dictionary.get("review_pairs", {}).get(token)
        if record and record.get("count", 0) >= min_count and record.get("confidence", 0.0) >= min_confidence:
            return record["norm"], record

    if allow_paper_seed_pairs:
        norm = dictionary.get("paper_seed_pairs", {}).get(token)
        if norm is not None:
            return norm, {"source": "paper_seed_pairs", "confidence": None, "count": None}

    return None, None


def is_vi_ambiguous_mfr_token(token: str, dictionary: dict | None = None) -> bool:
    """Return True if a token has multiple plausible train-derived normalizations."""
    if dictionary is None:
        dictionary = load_vi_mfr_dictionary()
    return token in dictionary.get("ambiguous_pairs", {}) or token in dictionary.get("review_pairs", {})


def apply_vi_mfr_to_tokens(
    tokens: Sequence[str],
    dictionary: dict | None = None,
    *,
    min_count: int = 3,
    min_confidence: float = 0.90,
) -> list[str]:
    """Apply high-confidence Vietnamese MFR replacements to a token sequence."""
    if dictionary is None:
        dictionary = load_vi_mfr_dictionary()
    output = []
    for tok in tokens:
        norm, _ = lookup_vi_mfr(tok, dictionary, min_count=min_count, min_confidence=min_confidence)
        output.append(norm if norm is not None else tok)
    return output


if __name__ == "__main__":
    d = load_vi_mfr_dictionary()
    print(d["metadata"])
    demo = ["t", "ko", "đc", "ngta", "20m", "zui"]
    print(demo, "->", apply_vi_mfr_to_tokens(demo, d))
    for tok in demo:
        print(tok, lookup_vi_mfr(tok, d), "ambiguous?", is_vi_ambiguous_mfr_token(tok, d))
