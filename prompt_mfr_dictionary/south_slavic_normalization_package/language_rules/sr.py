"""SR-specific wrappers for South Slavic lexical normalization."""
from __future__ import annotations
from typing import Sequence

try:
    from language_rules.south_slavic import (
        candidate_indices as south_slavic_candidate_indices,
        build_target_detection_prompt,
        build_target_normalization_prompt,
        get_rule_block,
        is_common_candidate,
    )
except ImportError:  # pragma: no cover
    from south_slavic import (  # type: ignore
        candidate_indices as south_slavic_candidate_indices,
        build_target_detection_prompt,
        build_target_normalization_prompt,
        get_rule_block,
        is_common_candidate,
    )

LANG = "sr"
LANGUAGE_RULE_BLOCK = get_rule_block(LANG)
FEW_SHOT_EXAMPLES = [
    {
        "raw_sentence": "komunalci kace pocne kaznjavanje ?",
        "tokens": [
            "komunalci",
            "kace",
            "pocne",
            "kaznjavanje",
            "?"
        ],
        "target_index": 2,
        "target": "pocne",
        "label": 1,
        "normalized": "počne",
        "notes": "Serbian diacritic restoration."
    },
    {
        "raw_sentence": "sta cu kad nece",
        "tokens": [
            "sta",
            "cu",
            "kad",
            "nece"
        ],
        "target_index": 0,
        "target": "sta",
        "label": 1,
        "normalized": "šta",
        "notes": "sta is commonly normalized to šta in Serbian."
    },
    {
        "raw_sentence": "jbg bas me nervira",
        "tokens": [
            "jbg",
            "bas",
            "me",
            "nervira"
        ],
        "target_index": 0,
        "target": "jbg",
        "label": 1,
        "normalized": "jebiga",
        "notes": "Frequent Serbian abbreviation."
    },
    {
        "raw_sentence": "@user sta ima #beograd",
        "tokens": [
            "@user",
            "sta",
            "ima",
            "#beograd"
        ],
        "target_index": 3,
        "target": "#beograd",
        "label": 0,
        "normalized": "#beograd",
        "notes": "Hashtag is protected."
    },
    {
        "raw_sentence": "ne znam ko je dosao",
        "tokens": [
            "ne",
            "znam",
            "ko",
            "je",
            "dosao"
        ],
        "target_index": 2,
        "target": "ko",
        "label": 0,
        "normalized": "ko",
        "notes": "ko is context-sensitive; preserve when unsure."
    }
]


def is_sr_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    return is_common_candidate(token, LANG)


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return south_slavic_candidate_indices(sentence_tokens, LANG)


def build_sr_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_target_detection_prompt(LANG, sentence_tokens, target_index, raw_sentence=raw_sentence, fewshot_examples=FEW_SHOT_EXAMPLES)


def build_sr_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_target_normalization_prompt(LANG, sentence_tokens, target_index, raw_sentence=raw_sentence, fewshot_examples=FEW_SHOT_EXAMPLES)

build_sr_target_prompt = build_sr_target_detection_prompt
build_sr_normalization_prompt = build_sr_target_normalization_prompt

__all__ = [
    "LANG", "LANGUAGE_RULE_BLOCK", "FEW_SHOT_EXAMPLES", "is_sr_likely_candidate", "candidate_indices",
    "build_sr_target_detection_prompt", "build_sr_target_normalization_prompt", "build_sr_target_prompt", "build_sr_normalization_prompt",
]
