"""SL-specific wrappers for South Slavic lexical normalization."""
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

LANG = "sl"
LANGUAGE_RULE_BLOCK = get_rule_block(LANG)
FEW_SHOT_EXAMPLES = [
    {
        "raw_sentence": "jst bi tud najdu kovanec",
        "tokens": [
            "jst",
            "bi",
            "tud",
            "najdu",
            "kovanec"
        ],
        "target_index": 0,
        "target": "jst",
        "label": 1,
        "normalized": "jaz",
        "notes": "Slovenian colloquial pronoun spelling."
    },
    {
        "raw_sentence": "sm vidu da je blo dobr",
        "tokens": [
            "sm",
            "vidu",
            "da",
            "je",
            "blo",
            "dobr"
        ],
        "target_index": 0,
        "target": "sm",
        "label": 1,
        "normalized": "sem",
        "notes": "Frequent Slovenian auxiliary normalization; still use context."
    },
    {
        "raw_sentence": "sej je tko zdej",
        "tokens": [
            "sej",
            "je",
            "tko",
            "zdej"
        ],
        "target_index": 2,
        "target": "tko",
        "label": 1,
        "normalized": "tako",
        "notes": "Slovenian colloquial spelling."
    },
    {
        "raw_sentence": "Iago Aspas hahaha :) #nogomet",
        "tokens": [
            "Iago",
            "Aspas",
            "hahaha",
            ":)",
            "#nogomet"
        ],
        "target_index": 4,
        "target": "#nogomet",
        "label": 0,
        "normalized": "#nogomet",
        "notes": "Hashtag is protected."
    },
    {
        "raw_sentence": "ne vem k je blo",
        "tokens": [
            "ne",
            "vem",
            "k",
            "je",
            "blo"
        ],
        "target_index": 2,
        "target": "k",
        "label": 0,
        "normalized": "k",
        "notes": "k is highly ambiguous; preserve if context is insufficient."
    }
]


def is_sl_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    return is_common_candidate(token, LANG)


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return south_slavic_candidate_indices(sentence_tokens, LANG)


def build_sl_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_target_detection_prompt(LANG, sentence_tokens, target_index, raw_sentence=raw_sentence, fewshot_examples=FEW_SHOT_EXAMPLES)


def build_sl_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_target_normalization_prompt(LANG, sentence_tokens, target_index, raw_sentence=raw_sentence, fewshot_examples=FEW_SHOT_EXAMPLES)

build_sl_target_prompt = build_sl_target_detection_prompt
build_sl_normalization_prompt = build_sl_target_normalization_prompt

__all__ = [
    "LANG", "LANGUAGE_RULE_BLOCK", "FEW_SHOT_EXAMPLES", "is_sl_likely_candidate", "candidate_indices",
    "build_sl_target_detection_prompt", "build_sl_target_normalization_prompt", "build_sl_target_prompt", "build_sl_normalization_prompt",
]
