"""German language-specific rules and prompt builders.

This module is data-driven and designed for MultiLexNorm-style lexical normalization.
It focuses on minimal surface normalization: spelling, capitalization when supported,
umlaut restoration, contractions, and split/merge patterns. It must not translate or
rewrite German text.
"""
from __future__ import annotations

import re
from typing import Sequence

from prompts.common_prompt import (
    build_common_detection_prompt,
    build_common_normalization_prompt,
    find_protected_indices,
    is_protected_token,
)

LANG = "de"

GERMAN_RULE_BLOCK = """
German-specific guidance:
- Focus on dataset-style lexical normalization, not translation, paraphrase, grammar correction, or style correction.
- Frequent project-data variants include:
  nich/net/ned → nicht, nix → nichts, grad/grade → gerade, heut → heute,
  hab → habe, gibts → gibt es, gehts → geht es, 's → es, nen → einen, nem → einem.
- Umlaut/diacritic candidates include:
  fuer/fur → für, ueber/uber → über, schoen → schön, waere/wär → wäre, wuerd/würd → würde.
  Do not globally convert ae/oe/ue/ss; use dataset evidence and context.
- German capitalization is important, especially nouns and sentence-initial words, but do not titlecase everything.
  Capitalize only when the dataset annotation style or sentence position strongly supports it.
- Split/merge candidates include:
  gibts → gibt es, gehts → geht es, habs → habe es, zuhause → zu Hause,
  schonmal → schon mal, naja → Na ja, Achso → Ach so, wettendass → Wetten, dass.
- Short words are often ambiguous: ne, n, nen, nem, das, was, ich, hab, ma, ja.
  If context is insufficient, preserve the raw token or send it to the target-token model rather than applying direct replacement.
- Preserve hashtags, mentions, URLs, emojis, numbers, product names, usernames, all-caps acronyms, and alphanumeric entities.
""".strip()

FEWSHOT_EXAMPLES = [
    {
        "raw_sentence": "ich hab grad nix gesehen",
        "tokens": ["ich", "hab", "grad", "nix", "gesehen"],
        "target_index": 2,
        "target": "grad",
        "label": 1,
        "normalized": "gerade",
        "notes": "grad is a stable colloquial variant of gerade in the project data.",
    },
    {
        "raw_sentence": "ich hab grad nix gesehen",
        "tokens": ["ich", "hab", "grad", "nix", "gesehen"],
        "target_index": 3,
        "target": "nix",
        "label": 1,
        "normalized": "nichts",
        "notes": "nix is commonly normalized to nichts.",
    },
    {
        "raw_sentence": "fuer mich gibts keine frage",
        "tokens": ["fuer", "mich", "gibts", "keine", "frage"],
        "target_index": 0,
        "target": "fuer",
        "label": 1,
        "normalized": "für",
        "notes": "fuer is an ASCII spelling of für.",
    },
    {
        "raw_sentence": "fuer mich gibts keine frage",
        "tokens": ["fuer", "mich", "gibts", "keine", "frage"],
        "target_index": 2,
        "target": "gibts",
        "label": 1,
        "normalized": "gibt es",
        "notes": "gibts can be split into gibt es under the dataset annotation style.",
    },
    {
        "raw_sentence": "Ich liebe #Tatort2024 @user",
        "tokens": ["Ich", "liebe", "#Tatort2024", "@user"],
        "target_index": 2,
        "target": "#Tatort2024",
        "label": 0,
        "normalized": "#Tatort2024",
        "notes": "Hashtags and mentions are protected social-media entities.",
    },
    {
        "raw_sentence": "iPhone15 ist neu",
        "tokens": ["iPhone15", "ist", "neu"],
        "target_index": 0,
        "target": "iPhone15",
        "label": 0,
        "normalized": "iPhone15",
        "notes": "Alphanumeric product/entity tokens are protected.",
    },
]

COMMON_GERMAN_VARIANTS = {
    "nich", "net", "ned", "nix", "grad", "grade", "heut", "hab", "habs", "gibts", "gehts", "is",
    "ne", "nen", "nem", "n", "'n", "'ne", "'s", "ma", "vllt", "un", "find", "kenn", "geh", "werd", "würd", "wär", "glaub", "freu", "les", "bekomm", "scho", "schonmal", "naja", "achso", "wettendass", "zuhause", "drüber", "rum", "eh", "spass", "scheiss", "fuer", "fur", "ueber", "uber", "schoen", "koenn", "kannste", "haste", "willste", "biste", "isses", "wirds", "solls", "heut", "gesehn", "jaa"
}

SHORT_CONTEXT_SENSITIVE = {
    "ich", "das", "was", "ja", "ne", "n", "nen", "nem", "ma", "hab", "is", "werd", "würd", "wär", "find", "glaub", "kenn", "freu", "die", "der", "und", "aber", "wie", "jetzt", "es"
}


def has_repeated_letters(tok: str) -> bool:
    return bool(re.search(r"([A-Za-zÀ-ž])\1{2,}", str(tok)))


def looks_like_umlaut_candidate(tok: str) -> bool:
    s = str(tok).lower()
    if any(x in s for x in ("fuer", "fur", "ueber", "uber", "schoen", "waere", "wuerd", "koenn", "muess", "grues")):
        return True
    return bool(re.search(r"(?:ae|oe|ue)", s)) and len(s) > 4


def looks_like_contraction(tok: str) -> bool:
    s = str(tok).lower()
    return s in {"ne", "nen", "nem", "n", "'n", "'ne", "'s", "gibts", "gehts", "habs", "wars", "isses", "wirds", "solls", "haste", "biste", "willste", "kannste"}


def is_likely_candidate(tok: str) -> bool:
    if is_protected_token(tok):
        return False
    s = str(tok)
    low = s.lower()
    if low in COMMON_GERMAN_VARIANTS:
        return True
    if looks_like_umlaut_candidate(s):
        return True
    if looks_like_contraction(s):
        return True
    if has_repeated_letters(s):
        return True
    # mixed script mojibake-ish or foreign character replacement candidates observed in data
    if re.search(r"[дД]", s):
        return True
    # sentence-initial/lowercase common words should be decided with context, not direct MFR
    if low in SHORT_CONTEXT_SENSITIVE:
        return True
    return False


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_likely_candidate(str(tok))]


def build_de_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens) if protected_indices is None else protected_indices,
        language_rule_block=GERMAN_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )


def build_de_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens) if protected_indices is None else protected_indices,
        language_rule_block=GERMAN_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )

build_detection_prompt = build_de_target_detection_prompt
build_normalization_prompt = build_de_target_normalization_prompt

__all__ = [
    "LANG", "GERMAN_RULE_BLOCK", "FEWSHOT_EXAMPLES", "COMMON_GERMAN_VARIANTS",
    "is_likely_candidate", "candidate_indices", "build_de_target_detection_prompt",
    "build_de_target_normalization_prompt", "build_detection_prompt", "build_normalization_prompt",
]
