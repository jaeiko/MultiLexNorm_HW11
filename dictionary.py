# dictionary.py

"""
Language-aware MFR dictionary module for MultiLexNorm.

Supported dictionary formats
----------------------------

1. Old flat mock format:
{
    "u": "you",
    "bcause": "because",
    "ㄹㅇ": "진짜"
}

2. New nested MFR format:
{
    "metadata": {...},
    "entries": {
        "en": {
            "u": {
                "norm": "you",
                "count": 328,
                "total": 350,
                "confidence": 0.937,
                "entropy": 0.19,
                "margin": 0.891
            }
        },
        "ko": {
            "왤케": {
                "norm": "왜이렇게",
                "count": 12,
                "total": 13,
                "confidence": 0.923
            }
        }
    }
}

Important
---------
- New MFR lookup should be language-aware:
    correct(token, lang)

- Empty string "" can be a valid normalization target in MultiLexNorm-style
  aligned data, so callers must check:
    if result is not None:
  not:
    if result:
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MFRDictionary:
    """
    MFR dictionary wrapper.

    This class supports both:
    - old flat dictionaries: token -> normalized token
    - new nested dictionaries: lang -> token -> metadata dict
    """

    def __init__(self, json_file_path: str | Path) -> None:
        self.path = Path(json_file_path)
        self.payload: dict[str, Any] = self._load_json(self.path)

        self.metadata: dict[str, Any] = {}
        self.entries: dict[str, dict[str, dict[str, Any]]] = {}
        self.flat_mapping: dict[str, str] = {}
        self.flat_mode: bool = False

        self._initialize_payload(self.payload)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"MFR dictionary file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError(
                "MFR dictionary JSON must be a dictionary object. "
                f"Got {type(payload)!r}."
            )

        return payload

    def _initialize_payload(self, payload: dict[str, Any]) -> None:
        """
        Detect dictionary format and initialize internal fields.
        """
        if "entries" in payload:
            self._initialize_nested_payload(payload)
        else:
            self._initialize_flat_payload(payload)

    def _initialize_flat_payload(self, payload: dict[str, Any]) -> None:
        """
        Initialize old mock dictionary format:
            {"u": "you", "bcause": "because"}
        """
        invalid_items = [
            (key, value)
            for key, value in payload.items()
            if not isinstance(key, str) or not isinstance(value, str)
        ]

        if invalid_items:
            preview = invalid_items[:3]
            raise ValueError(
                "Flat MFR dictionary must be {str: str}. "
                f"Invalid examples: {preview}"
            )

        self.flat_mode = True
        self.flat_mapping = dict(payload)
        self.metadata = {
            "format": "flat",
            "num_entries": len(self.flat_mapping),
        }

    def _initialize_nested_payload(self, payload: dict[str, Any]) -> None:
        """
        Initialize new nested MFR dictionary format:
            {"metadata": {...}, "entries": {"en": {"u": {"norm": "you"}}}}
        """
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("Nested MFR dictionary must contain dict field 'entries'.")

        normalized_entries: dict[str, dict[str, dict[str, Any]]] = {}

        for lang, lang_entries in entries.items():
            if not isinstance(lang, str):
                raise ValueError(f"Language key must be str, got {type(lang)!r}")

            if not isinstance(lang_entries, dict):
                raise ValueError(
                    f"entries[{lang!r}] must be a dictionary, "
                    f"got {type(lang_entries)!r}"
                )

            normalized_entries[lang] = {}

            for raw_token, info in lang_entries.items():
                if not isinstance(raw_token, str):
                    raise ValueError(
                        f"Raw token key under lang={lang!r} must be str, "
                        f"got {type(raw_token)!r}"
                    )

                if not isinstance(info, dict):
                    raise ValueError(
                        f"Entry for lang={lang!r}, token={raw_token!r} "
                        f"must be dict, got {type(info)!r}"
                    )

                if "norm" not in info:
                    raise ValueError(
                        f"Entry for lang={lang!r}, token={raw_token!r} "
                        "does not contain required field 'norm'."
                    )

                if not isinstance(info["norm"], str):
                    raise ValueError(
                        f"'norm' for lang={lang!r}, token={raw_token!r} "
                        f"must be str, got {type(info['norm'])!r}"
                    )

                normalized_entries[lang][raw_token] = dict(info)

        self.flat_mode = False
        self.entries = normalized_entries
        self.metadata = dict(payload.get("metadata", {}))

    @staticmethod
    def _normalize_lang(lang: str | None) -> str | None:
        """
        Normalize language code for lookup.

        Keep this conservative. Do not over-map language codes unless the dataset
        actually uses the alias.
        """
        if lang is None:
            return None

        lang = str(lang).strip()

        # Common defensive aliases.
        aliases = {
            "id-en": "iden",
            "id_en": "iden",
            "tr-de": "trde",
            "tr_de": "trde",
        }

        return aliases.get(lang.lower(), lang.lower())

    def correct(self, word: str, lang: str | None = None) -> str | None:
        """
        Return normalized token if found; otherwise return None.

        Parameters
        ----------
        word:
            Raw/noisy token.
        lang:
            Language code. Required for nested MFR dictionaries.

        Returns
        -------
        str | None
            Normalized token if found. None if no dictionary entry exists.

        Important
        ---------
        Empty string "" can be a valid normalization target.
        Therefore callers must use:

            result = dictionary.correct(token, lang)
            if result is not None:
                ...

        not:

            if result:
                ...
        """
        if not isinstance(word, str):
            word = str(word)

        if self.flat_mode:
            return self.flat_mapping.get(word)

        normalized_lang = self._normalize_lang(lang)
        if normalized_lang is None:
            return None

        item = self.entries.get(normalized_lang, {}).get(word)
        if item is None:
            return None

        return item["norm"]

    def lookup_info(self, word: str, lang: str | None = None) -> dict[str, Any] | None:
        """
        Return full metadata for a dictionary entry.

        Useful for threshold-based logic, logging, debugging, or LLM prompt hints.
        """
        if not isinstance(word, str):
            word = str(word)

        if self.flat_mode:
            norm = self.flat_mapping.get(word)
            if norm is None:
                return None
            return {
                "norm": norm,
                "source": "flat_mock_dictionary",
            }

        normalized_lang = self._normalize_lang(lang)
        if normalized_lang is None:
            return None

        item = self.entries.get(normalized_lang, {}).get(word)
        if item is None:
            return None

        return dict(item)

    def has_entry(self, word: str, lang: str | None = None) -> bool:
        """
        Return True if the dictionary has an entry for the given token.
        """
        return self.lookup_info(word, lang) is not None

    def get_confidence(self, word: str, lang: str | None = None) -> float | None:
        """
        Return confidence score if available.
        """
        info = self.lookup_info(word, lang)
        if info is None:
            return None

        confidence = info.get("confidence")
        if confidence is None:
            return None

        try:
            return float(confidence)
        except (TypeError, ValueError):
            return None

    def correct_if_confident(
        self,
        word: str,
        lang: str | None = None,
        *,
        min_confidence: float = 0.0,
        min_count: int = 1,
        min_margin: float = 0.0,
    ) -> str | None:
        """
        Return normalized token only if entry satisfies confidence thresholds.

        This is useful when we want high-precision MFR replacement before LLM.
        """
        info = self.lookup_info(word, lang)
        if info is None:
            return None

        norm = info.get("norm")
        if not isinstance(norm, str):
            return None

        confidence = float(info.get("confidence", 1.0))
        count = int(info.get("count", info.get("top_count", 1)))
        margin = float(info.get("margin", 1.0))

        if confidence < min_confidence:
            return None
        if count < min_count:
            return None
        if margin < min_margin:
            return None

        return norm

    def available_languages(self) -> list[str]:
        """
        Return available language codes.
        """
        if self.flat_mode:
            return []
        return sorted(self.entries.keys())

    def __len__(self) -> int:
        """
        Return number of dictionary entries.
        """
        if self.flat_mode:
            return len(self.flat_mapping)

        return sum(len(lang_entries) for lang_entries in self.entries.values())

    def __repr__(self) -> str:
        mode = "flat" if self.flat_mode else "nested"
        return f"MFRDictionary(mode={mode!r}, entries={len(self)})"
