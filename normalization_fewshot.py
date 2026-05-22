"""Edit-distance based dynamic few-shot retrieval for normalization prompts.

Indexes pairs of (raw, norm) tokens by language from a training set (parquet),
and at runtime retrieves the top K positive/negative examples closest to a
target token based on edit distance.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Sequence, Dict, List, Tuple, Union

import pandas as pd


def _to_tokens(value: Any) -> list[str]:
    """Converts raw/norm sequence column values into list of token strings.

    Args:
        value: Token sequence in raw list, array, or string format.

    Returns:
        list[str]: Standard list of token strings.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(t) for t in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(t) for t in parsed]
        except (ValueError, SyntaxError):
            pass
        return value.split()
    return []


try:
    from rapidfuzz.distance import Levenshtein as _rf_lev

    def _levenshtein(a: str, b: str) -> int:
        return _rf_lev.distance(a, b)
except ImportError:
    def _levenshtein(a: str, b: str) -> int:
        """Pure-python fallback Levenshtein distance calculation.

        Args:
            a: First string sequence.
            b: Second string sequence.

        Returns:
            int: Edit distance between a and b.
        """
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i]
            for j, cb in enumerate(b, start=1):
                curr.append(min(
                    curr[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (ca != cb),
                ))
            prev = curr
        return prev[-1]


def _normalized_edit(a: str, b: str) -> float:
    """Calculates normalized edit distance (edit distance / max length).

    Args:
        a: First string sentence.
        b: Second string sentence.

    Returns:
        float: Normalized edit distance.
    """
    denom = max(len(a), len(b), 1)
    return _levenshtein(a, b) / denom


class NormalizationFewshot:
    """Retrieves context-sensitive positive/negative dynamic few-shot examples.

    Attributes:
        default_positive_k (int): Number of positive templates to return.
        default_negative_k (int): Number of negative templates to return.
        lang_overrides (dict): Language-specific K configurations.
    """

    def __init__(
        self,
        train_dataset_path: str | Path,
        *,
        default_positive_k: int = 5,
        default_negative_k: int = 5,
        lang_overrides: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Initializes the NormalizationFewshot indexing.

        Args:
            train_dataset_path: Path to the parquet training set.
            default_positive_k: Default count of positive items to retrieve.
            default_negative_k: Default count of negative items to retrieve.
            lang_overrides: Custom dictionary overriding default positive/negative K by language.
        """
        self.default_positive_k = int(default_positive_k)
        self.default_negative_k = int(default_negative_k)
        self.lang_overrides = dict(lang_overrides or {})
        self._by_lang: dict[str, list[dict[str, Any]]] = {}
        self._load(Path(train_dataset_path))

    def _load(self, path: Path) -> None:
        """Loads and indexes the train parquet dataset items.

        Args:
            path: Absolute path to train parquet.
        """
        df = pd.read_parquet(path)
        if not {"lang", "raw", "norm"}.issubset(df.columns):
            raise ValueError(f"train dataset missing required columns: {path}")

        for lang, group in df.groupby("lang"):
            pool: list[dict[str, Any]] = []
            for raw_val, norm_val in zip(group["raw"], group["norm"]):
                raw = _to_tokens(raw_val)
                norm = _to_tokens(norm_val)
                if len(raw) != len(norm) or not raw:
                    continue
                changed = [i for i, (r, n) in enumerate(zip(raw, norm)) if r != n]
                unchanged = [i for i, (r, n) in enumerate(zip(raw, norm)) if r == n]
                pool.append({
                    "raw": raw,
                    "norm": norm,
                    "raw_text": " ".join(raw),
                    "changed_indices": changed,
                    "unchanged_indices": unchanged,
                })
            self._by_lang[str(lang)] = pool

    def k_for_lang(self, lang: str | None) -> tuple[int, int]:
        """Gets positive and negative K values configured for a language.

        Args:
            lang: Language code identifier.

        Returns:
            tuple[int, int]: (positive_k, negative_k) values.
        """
        if lang is None:
            return self.default_positive_k, self.default_negative_k
        override = self.lang_overrides.get(str(lang), {})
        positive_k = int(override.get("positive_k", self.default_positive_k))
        negative_k = int(override.get("negative_k", self.default_negative_k))
        return positive_k, negative_k

    def _rank_pool(
        self,
        pool: list[dict[str, Any]],
        target_token: str,
        index_key: str,
        full_sentence_text: str,
        k: int,
    ) -> list[dict[str, Any]]:
        """Ranks and extracts the best K candidate sentences matching target criteria.

        Args:
            pool: List of indexed training instances.
            target_token: The string token under comparison.
            index_key: Key indicator ('changed_indices' or 'unchanged_indices').
            full_sentence_text: Concatenated string of the current context sentence.
            k: Maximum candidates to extract.

        Returns:
            list[dict[str, Any]]: List of top K retrieved few-shot item dictionaries.
        """
        if k <= 0 or not pool:
            return []

        scored: list[tuple[int, float, dict[str, Any], int]] = []
        for entry in pool:
            indices = entry[index_key]
            if not indices:
                continue
            best_pos = -1
            best_dist = -1
            for idx in indices:
                cand_token = entry["raw"][idx]
                dist = _levenshtein(target_token, cand_token)
                if best_pos < 0 or dist < best_dist:
                    best_pos = idx
                    best_dist = dist
                    if dist == 0:
                        break
            if best_pos < 0:
                continue
            sent_score = (
                _normalized_edit(full_sentence_text, entry["raw_text"])
                if full_sentence_text
                else 0.0
            )
            scored.append((best_dist, sent_score, entry, best_pos))

        scored.sort(key=lambda x: (x[0], x[1]))

        out: list[dict[str, Any]] = []
        for _, _, entry, pos in scored[:k]:
            out.append({
                "raw": entry["raw"],
                "target_token": entry["raw"][pos],
                "norm": entry["norm"],
            })
        return out

    def retrieve(
        self,
        *,
        target_token: str,
        lang: str,
        positive_k: int | None = None,
        negative_k: int | None = None,
        full_sentence_tokens: Sequence[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieves dynamic matching positive and negative few-shot examples.

        Args:
            target_token: Token requesting few-shots context.
            lang: Target language code string.
            positive_k: Override for number of positive items.
            negative_k: Override for number of negative items.
            full_sentence_tokens: Word token context of the target sentence.

        Returns:
            dict: Mapping {"positive": list[pairs], "negative": list[pairs]}
        """
        default_pos, default_neg = self.k_for_lang(lang)
        pos_k = default_pos if positive_k is None else int(positive_k)
        neg_k = default_neg if negative_k is None else int(negative_k)

        pool = self._by_lang.get(str(lang), [])
        target_token_s = str(target_token)
        full_sentence_text = (
            " ".join(str(t) for t in full_sentence_tokens) if full_sentence_tokens else ""
        )

        positives = self._rank_pool(pool, target_token_s, "changed_indices", full_sentence_text, pos_k)
        negatives = self._rank_pool(pool, target_token_s, "unchanged_indices", full_sentence_text, neg_k)

        return {"positive": positives, "negative": negatives}
