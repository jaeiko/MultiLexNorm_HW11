"""Dutch language-specific rules and prompt builders.

This module is intentionally data-driven: it uses Dutch patterns observed in the
project data rather than a large external linguistic taxonomy.
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

LANG = "nl"

DUTCH_RULE_BLOCK = """
Dutch-specific guidance:
- Focus on dataset-style lexical normalization, not translation or paraphrase.
- Common high-signal Dutch/Flemish social-media variants include:
  ni/nie/nt → niet, da → dat, goe → goed, ma/mr → maar, wa → wat,
  ff/effe → even, mss → misschien, gwn → gewoon, wss → waarschijnlijk,
  mn/m'n → mijn, 't/t → het, 'n/n/ne/nen → een, vd → van de.
- Common clitic or split forms include:
  aant → aan het, kheb → ik heb, kga → ik ga, kben → ik ben,
  kzal → ik zal, kweet → ik weet, ist/tis → is het/het is,
  das → dat is, dak → dat ik, daje → dat je.
- Dutch/Flemish colloquial pronouns and particles can be ambiguous:
  gij/ge may map to jij/je depending on context; n/t/k/ne/ma/wa/da are short and context-sensitive.
- Restore minimal orthographic forms only when supported by examples, such as prive → privé or Oke/Okee → Oké.
- Repeated letters may be normalized when they are used only for emphasis, but preserve expressive laughter or emoticons unless examples support a specific normalization.
- Do not normalize hashtags, mentions, URLs, emojis, numbers, product names, usernames, or alphanumeric entities.
- If a token could be a proper name, username, hashtag fragment, or ordinary word, preserve unless context clearly supports a minimal surface correction.
""".strip()

FEWSHOT_EXAMPLES = [
    {
        "raw_sentence": "ik moet ff naar huis",
        "tokens": ["ik", "moet", "ff", "naar", "huis"],
        "target_index": 2,
        "target": "ff",
        "label": 1,
        "normalized": "even",
        "notes": "ff is a stable Dutch social-media abbreviation for even in the project data.",
    },
    {
        "raw_sentence": "kheb da ni gezien",
        "tokens": ["kheb", "da", "ni", "gezien"],
        "target_index": 0,
        "target": "kheb",
        "label": 1,
        "normalized": "ik heb",
        "notes": "kheb is a cliticized form that can normalize to ik heb; casing depends on sentence position.",
    },
    {
        "raw_sentence": "kheb da ni gezien",
        "tokens": ["kheb", "da", "ni", "gezien"],
        "target_index": 2,
        "target": "ni",
        "label": 1,
        "normalized": "niet",
        "notes": "ni is a high-confidence project-data variant of niet.",
    },
    {
        "raw_sentence": "m'n fiets staat daar",
        "tokens": ["m'n", "fiets", "staat", "daar"],
        "target_index": 0,
        "target": "m'n",
        "label": 1,
        "normalized": "mijn",
        "notes": "m'n is normally expanded to mijn in this dataset.",
    },
    {
        "raw_sentence": "Check #feestje2019 @vriend",
        "tokens": ["Check", "#feestje2019", "@vriend"],
        "target_index": 1,
        "target": "#feestje2019",
        "label": 0,
        "normalized": "#feestje2019",
        "notes": "Hashtags and mentions are protected social-media entities.",
    },
    {
        "raw_sentence": "Dat is iPhone15 nieuws",
        "tokens": ["Dat", "is", "iPhone15", "nieuws"],
        "target_index": 2,
        "target": "iPhone15",
        "label": 0,
        "normalized": "iPhone15",
        "notes": "Alphanumeric product/entity tokens are protected.",
    },
]

# High-signal forms from project data. These are candidates, not necessarily automatic replacements.
COMMON_DUTCH_VARIANTS = {
    "ni", "nie", "nt", "da", "goe", "ma", "mr", "wa", "ff", "effe", "mss", "gwn", "wss",
    "mn", "m'n", "'t", "t", "'n", "n", "ne", "nen", "vd", "vr", "vn", "nr", "aub", "idd", "ipv", "ivm",
    "aant", "kheb", "kga", "kben", "kzal", "kzou", "kmoe", "kwas", "khad", "kwil", "kweet",
    "ist", "tis", "das", "mja", "dak", "daje", "int", "vant", "omda", "tzal", "tzijn",
    "gij", "ge", "sebiet", "strx", "fdaag", "gister", "prive", "oke", "okee", "suc6", "8er",
    "jah", "jaa", "neej", "owk", "ofsow", "men", "dr", "em", "z'n", "ikke", "kijke", "zitte", "hebk",
}

SHORT_CONTEXT_SENSITIVE = {
    "n", "t", "k", "ne", "da", "ma", "wa", "ge", "gij", "men", "gn", "dr", "na", "bn", "me", "my", "em",
}


def has_repeated_letters(tok: str) -> bool:
    return bool(re.search(r"([A-Za-zÀ-ž])\1{2,}", str(tok)))


def looks_like_k_clitic(tok: str) -> bool:
    s = str(tok).lower()
    return bool(re.fullmatch(r"k(?:heb|ga|ben|zal|zou|moe|moet|was|had|wil|weet|kom|kan|moest|had).*", s))


def has_digit_wordplay(tok: str) -> bool:
    s = str(tok).lower()
    return bool(re.search(r"[0-9]", s)) and not is_protected_token(s)


def is_likely_candidate(tok: str) -> bool:
    if is_protected_token(tok):
        return False
    s = str(tok)
    low = s.lower()
    if low in COMMON_DUTCH_VARIANTS:
        return True
    if looks_like_k_clitic(s):
        return True
    if has_repeated_letters(s):
        return True
    if has_digit_wordplay(s):
        return True
    if re.fullmatch(r"[a-z]{1,4}", low) and low in SHORT_CONTEXT_SENSITIVE:
        return True
    # Common dropped-final-n/e colloquial endings observed in Dutch/Flemish data.
    if re.search(r"(?:e|n)$", low) and low in {"moete", "hebbe", "wete", "kome", "zitte", "mense", "geslape", "drinkn"}:
        return True
    return False


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_likely_candidate(str(tok))]


def build_nl_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens) if protected_indices is None else protected_indices,
        language_rule_block=DUTCH_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )


def build_nl_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens) if protected_indices is None else protected_indices,
        language_rule_block=DUTCH_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )

# Backward-compatible aliases.
build_detection_prompt = build_nl_target_detection_prompt
build_normalization_prompt = build_nl_target_normalization_prompt

__all__ = [
    "LANG", "DUTCH_RULE_BLOCK", "FEWSHOT_EXAMPLES", "COMMON_DUTCH_VARIANTS",
    "is_likely_candidate", "candidate_indices", "build_nl_target_detection_prompt",
    "build_nl_target_normalization_prompt", "build_detection_prompt", "build_normalization_prompt",
]
