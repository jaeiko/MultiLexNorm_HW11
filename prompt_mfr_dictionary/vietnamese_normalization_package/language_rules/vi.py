"""Vietnamese-specific rules built on top of the common target-only prompt.

The language-specific file supplies candidate detection, guidance, and few-shot
examples.  The common prompt in prompts/common_prompt.py enforces alignment,
protected tokens, target-only output, and the no-translation/no-paraphrase rule.
"""
from __future__ import annotations

import re
from typing import Sequence

try:
    from prompts.common_prompt import (
        build_common_detection_prompt,
        build_common_normalization_prompt,
        find_protected_indices,
        is_protected_token,
    )
except ImportError:  # pragma: no cover
    from common_prompt import (  # type: ignore
        build_common_detection_prompt,
        build_common_normalization_prompt,
        find_protected_indices,
        is_protected_token,
    )

LANG = "vi"

VI_DIACRITIC_RE = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
LATIN_RE = re.compile(r"[A-Za-zÀ-ỹ]")
REPEATED_LATIN_RE = re.compile(r"([A-Za-zÀ-ỹ])\1{2,}")

COMMON_VI_NORMALIZATION_CUES: dict[str, str] = {
    "k": "không", "ko": "không", "khong": "không", "khum": "không", "hong": "không", "hông": "không", "hok": "không", "hk": "không", "kg": "không",
    "đc": "được", "dc": "được", "duoc": "được", "dk": "được/đăng ký", "đk": "được/đăng ký",
    "r": "rồi", "roi": "rồi", "rùi": "rồi", "gòi": "rồi",
    "t": "tôi/tao/tớ/tui", "m": "mình/mày", "mk": "mình", "mik": "mình", "e": "em", "a": "anh", "c": "chị",
    "ng": "người", "ngta": "người ta", "ny": "người yêu", "mn": "mọi người", "mng": "mọi người",
    "nma": "nhưng mà", "nhma": "nhưng mà", "gđ": "gia đình", "đt": "điện thoại", "sn": "sinh nhật",
    "zui": "vui", "dô": "vô", "zô": "vô", "j": "gì", "gi": "gì", "jz": "gì vậy",
    "bh": "bây giờ/bao giờ", "bt": "biết/bình thường", "bn": "bao nhiêu/bạn", "tr": "trời/trường",
}

AMBIGUOUS_VI_TOKENS = {"t", "m", "c", "a", "e", "r", "bh", "bt", "bn", "tr", "đk", "dk", "hk", "hum", "j"}

VIETNAMESE_RULE_BLOCK = """
Vietnamese-specific guidance:
- Vietnamese social media text contains many short abbreviations and diacritic-free variants.
- Frequent stable candidates include ko/k/khong/khum->không, đc/dc/duoc->được, ngta->người ta, ny->người yêu, mn->mọi người, nma/nhma->nhưng mà, zui->vui.
- Vietnamese is syllable-spaced; a single raw token may normalize to a multi-syllable expression, e.g. ngta->người ta or ny->người yêu.
- Short tokens are often ambiguous: t, m, c, a, e, r, bh, bt, bn, tr, đk/dk. Use context and avoid direct replacement when unsure.
- Do not translate names, hashtags, mentions, numbers, event tags, product names, or alphanumeric entities.
- Do not expand casual particles or expressive tokens unless dataset-style examples strongly support it.
""".strip()

FEW_SHOT_EXAMPLES = [
    {"tokens": ["t", "ko", "biết"], "target_index": 1, "target": "ko", "label": 1, "normalized": "không", "notes": "ko is a high-signal abbreviation for không."},
    {"tokens": ["ngta", "nói", "zui"], "target_index": 0, "target": "ngta", "label": 1, "normalized": "người ta", "notes": "One raw token may expand to a multi-syllable Vietnamese expression."},
    {"tokens": ["t", "ko", "biết"], "target_index": 0, "target": "t", "label": 1, "normalized": "tôi", "notes": "t can require normalization, but the exact pronoun is context-dependent."},
    {"tokens": ["#Vietnam", "2024", "ok"], "target_index": 0, "target": "#Vietnam", "label": 0, "normalized": "#Vietnam", "notes": "Hashtags and numbers are protected social-media artifacts."},
]


def has_vietnamese_diacritic(token: str) -> bool:
    return bool(VI_DIACRITIC_RE.search(token or ""))


def is_vi_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    if not token or is_protected_token(token):
        return False
    lower = token.lower()
    if lower in COMMON_VI_NORMALIZATION_CUES:
        return True
    if REPEATED_LATIN_RE.search(token):
        return True
    if LATIN_RE.search(token) and len(lower) <= 4 and not has_vietnamese_diacritic(token):
        # Short diacritic-free forms are common Vietnamese abbreviation candidates.
        return True
    return False


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_vi_likely_candidate(tok, tokens=sentence_tokens, index=i)]


def build_vi_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=VIETNAMESE_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )


def build_vi_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=VIETNAMESE_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )

# Backward-compatible aliases
build_vi_target_prompt = build_vi_target_detection_prompt
build_vi_normalization_prompt = build_vi_target_normalization_prompt

__all__ = [
    "LANG", "VIETNAMESE_RULE_BLOCK", "COMMON_VI_NORMALIZATION_CUES", "AMBIGUOUS_VI_TOKENS", "FEW_SHOT_EXAMPLES",
    "has_vietnamese_diacritic", "is_vi_likely_candidate", "candidate_indices",
    "build_vi_target_detection_prompt", "build_vi_target_normalization_prompt", "build_vi_target_prompt", "build_vi_normalization_prompt",
]
