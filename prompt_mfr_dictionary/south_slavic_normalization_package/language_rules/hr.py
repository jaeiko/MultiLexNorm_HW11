"""HR-specific wrappers for South Slavic lexical normalization."""
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

LANG = "hr"
LANGUAGE_RULE_BLOCK = get_rule_block(LANG)
FEW_SHOT_EXAMPLES = [
    {
        "raw_sentence": "veceras san osta sam",
        "tokens": [
            "veceras",
            "san",
            "osta",
            "sam"
        ],
        "target_index": 1,
        "target": "san",
        "label": 1,
        "normalized": "sam",
        "notes": "Croatian dialectal/nonstandard form normalized in dataset examples."
    },
    {
        "raw_sentence": "ak neš ne mogu smislit",
        "tokens": [
            "ak",
            "neš",
            "ne",
            "mogu",
            "smislit"
        ],
        "target_index": 0,
        "target": "ak",
        "label": 1,
        "normalized": "ako",
        "notes": "Croatian shortened conjunction."
    },
    {
        "raw_sentence": "kaj ima tak posebnog",
        "tokens": [
            "kaj",
            "ima",
            "tak",
            "posebnog"
        ],
        "target_index": 2,
        "target": "tak",
        "label": 1,
        "normalized": "tako",
        "notes": "Final vowel dropping."
    },
    {
        "raw_sentence": "pofotkat ruke #karla_photography",
        "tokens": [
            "pofotkat",
            "ruke",
            "#karla_photography"
        ],
        "target_index": 2,
        "target": "#karla_photography",
        "label": 0,
        "normalized": "#karla_photography",
        "notes": "Hashtag is protected."
    },
    {
        "raw_sentence": "komp mi je crkao",
        "tokens": [
            "komp",
            "mi",
            "je",
            "crkao"
        ],
        "target_index": 0,
        "target": "komp",
        "label": 0,
        "normalized": "komp",
        "notes": "Do not semantically translate colloquial komp to kompjuter without exact evidence."
    }
]


def is_hr_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    return is_common_candidate(token, LANG)


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return south_slavic_candidate_indices(sentence_tokens, LANG)


def build_hr_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_target_detection_prompt(LANG, sentence_tokens, target_index, raw_sentence=raw_sentence, fewshot_examples=FEW_SHOT_EXAMPLES)


def build_hr_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_target_normalization_prompt(LANG, sentence_tokens, target_index, raw_sentence=raw_sentence, fewshot_examples=FEW_SHOT_EXAMPLES)

build_hr_target_prompt = build_hr_target_detection_prompt
build_hr_normalization_prompt = build_hr_target_normalization_prompt

__all__ = [
    "LANG", "LANGUAGE_RULE_BLOCK", "FEW_SHOT_EXAMPLES", "is_hr_likely_candidate", "candidate_indices",
    "build_hr_target_detection_prompt", "build_hr_target_normalization_prompt", "build_hr_target_prompt", "build_hr_normalization_prompt",
]
