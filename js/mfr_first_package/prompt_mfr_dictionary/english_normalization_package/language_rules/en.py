"""English language-specific rules and prompt builders.

This module is data-driven and designed for MultiLexNorm-style lexical normalization.
It focuses on minimal surface normalization: contractions, abbreviations, slang spellings,
phonetic spellings, g-dropping, repeated letters, and selected split/expansion patterns.
It must not translate, paraphrase, rewrite, or normalize protected social-media entities.
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

LANG = "en"

ENGLISH_RULE_BLOCK = """
English-specific guidance:
- Focus on dataset-style lexical normalization, not translation, paraphrase, grammar correction, or style correction.
- Common project-data variants include:
  u → you, r → are, n/nd → and, pls/plz → please, ppl → people,
  dont/didnt/cant/isnt/wasnt/doesnt → standard contractions with apostrophes,
  im/ive/ill/theyre/youre/thats/theres/whos → standard contractions with apostrophes.
- Informal expansions include:
  gonna → going to, wanna → want to, tryna → trying to, finna → going to,
  ima/imma → i'm going to, cuz/bc/cause → because, tho → though, bout/abt → about,
  lil → little, bruh/bro → brother, ty → thank you, thx → thanks.
- Phonetic/eye-dialect spellings may be normalized when supported by data/context:
  dat/dis/da/tha/wit/kno/wat/ya/yall/goin/gettin/talkin/feelin/lookin/comin.
- British/American spelling normalization appears in the data, e.g. favourite → favorite, neighbour → neighbor, but only apply exact observed pairs or prompt/model evidence.
- Numeric shorthand such as 2 → to and 4 → for can be normalization candidates, but pure numbers are protected by default and must not be changed unless the target clearly functions as English shorthand.
- RT/rt and DM/dm are social-media terms. They may be normalized when the dataset context supports retweet/direct message, but many occurrences are preserved; treat them as context-sensitive.
- Offensive colloquial spellings may appear in the training-derived dictionary. If present, treat them only as dataset annotation artifacts and use the minimum dataset-style surface normalization.
- Preserve hashtags, mentions, URLs, emojis, numbers, product names, usernames, all-caps acronyms, and alphanumeric entities.
""".strip()

FEWSHOT_EXAMPLES = [
    {
        "raw_sentence": "u dont know what im saying",
        "tokens": ["u", "dont", "know", "what", "im", "saying"],
        "target_index": 0,
        "target": "u",
        "label": 1,
        "normalized": "you",
        "notes": "u is a stable English shorthand for you in the project data.",
    },
    {
        "raw_sentence": "u dont know what im saying",
        "tokens": ["u", "dont", "know", "what", "im", "saying"],
        "target_index": 1,
        "target": "dont",
        "label": 1,
        "normalized": "don't",
        "notes": "dont is normalized by adding the missing apostrophe.",
    },
    {
        "raw_sentence": "im gonna see yall soon",
        "tokens": ["im", "gonna", "see", "yall", "soon"],
        "target_index": 1,
        "target": "gonna",
        "label": 1,
        "normalized": "going to",
        "notes": "gonna is expanded to going to when used as a future marker.",
    },
    {
        "raw_sentence": "check #OpenFollow @user 2 tickets",
        "tokens": ["check", "#OpenFollow", "@user", "2", "tickets"],
        "target_index": 1,
        "target": "#OpenFollow",
        "label": 0,
        "normalized": "#OpenFollow",
        "notes": "Hashtags are protected social-media entities and should be preserved.",
    },
    {
        "raw_sentence": "I need 2 tickets 4 tomorrow",
        "tokens": ["I", "need", "2", "tickets", "4", "tomorrow"],
        "target_index": 2,
        "target": "2",
        "label": 0,
        "normalized": "2",
        "notes": "A pure number should be preserved when it functions as a quantity. Numeric shorthand requires context.",
    },
]

CONTRACTION_CANDIDATES = {
    "im", "ive", "ill", "id", "dont", "didnt", "cant", "wont", "isnt", "wasnt", "werent",
    "doesnt", "shouldnt", "couldnt", "wouldnt", "havent", "hasnt", "theyre", "youre", "thats",
    "theres", "whos", "hes", "shes", "its", "lets", "arent", "ain't", "aint",
}

SLANG_ABBREVIATION_CANDIDATES = {
    "u", "ur", "r", "n", "nd", "w", "wit", "pls", "plz", "ppl", "bc", "cuz", "cause",
    "tho", "bout", "abt", "lil", "bruh", "bro", "gonna", "wanna", "tryna", "finna", "ima",
    "imma", "gon", "rn", "af", "tf", "dm", "rt", "ty", "thx", "fav", "pic", "vid", "congrats",
    "dat", "da", "tha", "dis", "wat", "kno", "ya", "yall", "yo", "em", "b", "d", "c", "2", "4",
}

EYE_DIALECT_RE = re.compile(r"^[A-Za-z]+in$", re.IGNORECASE)  # goin, gettin, talkin; filtered by length
REPEATED_RE = re.compile(r"([A-Za-z])\1{2,}")
MISSING_APOSTROPHE_RE = re.compile(r"^(?:[A-Za-z]+nt|[A-Za-z]+re|[A-Za-z]+ve|[A-Za-z]+ll|[A-Za-z]+s)$", re.IGNORECASE)


def is_en_likely_candidate(tok: str) -> bool:
    s = str(tok).strip()
    if not s:
        return False
    # Pure numbers are normally protected; allow 2/4 as prompt candidates only when the caller has not already protected them.
    if is_protected_token(s) and s not in {"2", "4"}:
        return False
    low = s.lower()
    if low in CONTRACTION_CANDIDATES or low in SLANG_ABBREVIATION_CANDIDATES:
        return True
    if REPEATED_RE.search(s):
        return True
    if len(s) >= 5 and EYE_DIALECT_RE.match(s) and low not in {"begin", "within", "again"}:
        return True
    if MISSING_APOSTROPHE_RE.match(s) and low in CONTRACTION_CANDIDATES:
        return True
    # Common missing-apostrophe family not fully covered above.
    if low.endswith("nt") and len(low) > 4:
        return True
    return False


def candidate_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_en_likely_candidate(str(tok))]


def build_en_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=ENGLISH_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )


def build_en_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=ENGLISH_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )

# Backward-compatible aliases
build_detection_prompt = build_en_target_detection_prompt
build_normalization_prompt = build_en_target_normalization_prompt
