"""Edit-distance based dynamic fewshot retrieval for normalization prompts.

학습셋(parquet)에서 (raw, norm) 페어를 언어별로 인덱싱하고,
런타임에 target token과 가장 비슷한 사례 K개를 positive/negative로 나누어
검색해 normalization prompt에 끼워 넣는다.

- positive: 그 train row에서 raw != norm인 changed token 중 target과
  edit distance가 가장 작은 위치 → "이 토큰은 정규화됐다"는 시그널.
- negative: 그 train row에서 raw == norm인 unchanged token 중 target과
  edit distance가 가장 작은 위치 → "이 토큰은 그대로 둬도 된다"는 시그널.

랭킹 기준:
  1. token-level edit distance (target_token vs train의 후보 token)
  2. sentence-level normalized edit distance (tie-break: 문맥 유사도)
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def _to_tokens(value: Any) -> list[str]:
    """raw/norm 컬럼이 list, numpy array, str repr 어느 형태든 토큰 리스트로 변환."""
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
    denom = max(len(a), len(b), 1)
    return _levenshtein(a, b) / denom


class NormalizationFewshot:
    """언어별 train 인덱스 + target-token 기반 positive/negative retrieval.

    페어 형식: {"raw": [...], "target_token": "...", "norm": [...]}

    Args:
        train_dataset_path: train parquet 경로 (lang/raw/norm 컬럼 필요).
        default_positive_k: 기본 positive 페어 개수.
        default_negative_k: 기본 negative 페어 개수.
        lang_overrides: 언어별 K 오버라이드.
            예: {"th": {"positive_k": 2, "negative_k": 2}}
    """

    def __init__(
        self,
        train_dataset_path: str | Path,
        *,
        default_positive_k: int = 5,
        default_negative_k: int = 5,
        lang_overrides: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.default_positive_k = int(default_positive_k)
        self.default_negative_k = int(default_negative_k)
        self.lang_overrides = dict(lang_overrides or {})
        self._by_lang: dict[str, list[dict[str, Any]]] = {}
        self._load(Path(train_dataset_path))

    def _load(self, path: Path) -> None:
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
        """언어별 (positive_k, negative_k)를 반환한다."""
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
        """pool에서 각 sentence의 index_key 위치 중 target과 가장 가까운 토큰을
        골라 점수를 매기고 상위 k개를 페어 형태로 반환한다."""
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
        """target_token에 가까운 positive/negative 페어를 검색해 dict로 반환.

        Returns:
            {"positive": [pair, ...], "negative": [pair, ...]}
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
