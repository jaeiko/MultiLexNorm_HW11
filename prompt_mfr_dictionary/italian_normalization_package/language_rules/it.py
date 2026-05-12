"""Italian language-specific rules and prompt builders.

This module is data-driven and designed for MultiLexNorm-style lexical normalization.
It focuses on minimal surface normalization: accent/apostrophe restoration, Italian
social-media abbreviations, repeated letters, and cautious casing. It must not translate,
paraphrase, rewrite, or normalize protected social-media entities.
"""
from __future__ import annotations

import re
from typing import Sequence

from prompts.common_prompt import (
    build_common_detection_prompt,
    build_common_normalization_prompt,
    is_protected_token,
)

LANG = "it"

ITALIAN_RULE_BLOCK = """
Italian-specific guidance:
- Focus on dataset-style lexical normalization, not translation, paraphrase, grammar correction, or style correction.
- Common project-data variants include:
  nn → non, ke → che, x → per, cmq/Cmq → comunque/Comunque, nov → novembre,
  dx → destra, sx → sinistra, info → informazioni.
- Accent and apostrophe restoration is important:
  e' / E' → è / È, é → è, perche' / perchè → perché, puo' → può, pero' → però,
  gia → già, cosi → così, piu → più.
- Repeated letters may be reduced when they are clearly expressive spelling, e.g. Beppeee → Beppe.
- Casing changes occur frequently, but many are sentence-initial capitalization, all-caps emphasis,
  acronyms, political names, person names, place names, or product names. Do not globally lowercase or titlecase.
- Preserve named entities and acronyms unless the exact dataset-style surface correction is strongly supported by context.
  Examples such as Monti, Roma, Italia, Mario, PD, CGIL, CISL, UIL, iPhone, Sky, One Direction, Niall are context-sensitive.
- Short tokens such as e, si, la, il, x, de/di, and all-caps function words require context.
- Preserve hashtags, mentions, URLs, emojis, numbers, product names, usernames, party names, organization names,
  all-caps acronyms, and alphanumeric entities.
""".strip()

FEWSHOT_EXAMPLES = [
    {
        "raw_sentence": "nn so perche' e' successo",
        "tokens": ["nn", "so", "perche'", "e'", "successo"],
        "target_index": 0,
        "target": "nn",
        "label": 1,
        "normalized": "non",
        "notes": "nn is a stable Italian social-media abbreviation for non.",
    },
    {
        "raw_sentence": "nn so perche' e' successo",
        "tokens": ["nn", "so", "perche'", "e'", "successo"],
        "target_index": 2,
        "target": "perche'",
        "label": 1,
        "normalized": "perché",
        "notes": "perche' is normalized by restoring the standard accented final vowel.",
    },
    {
        "raw_sentence": "ke fai x cena ?",
        "tokens": ["ke", "fai", "x", "cena", "?"],
        "target_index": 0,
        "target": "ke",
        "label": 1,
        "normalized": "che",
        "notes": "ke is a common spelling variant for che.",
    },
    {
        "raw_sentence": "seguo #Roma2024 @utente",
        "tokens": ["seguo", "#Roma2024", "@utente"],
        "target_index": 1,
        "target": "#Roma2024",
        "label": 0,
        "normalized": "#Roma2024",
        "notes": "Hashtags and alphanumeric social-media entities are protected.",
    },
    {
        "raw_sentence": "Monti parla del Governo",
        "tokens": ["Monti", "parla", "del", "Governo"],
        "target_index": 0,
        "target": "Monti",
        "label": 0,
        "normalized": "Monti",
        "notes": "Names and political/person entities should not be changed unless the exact token is clearly a dataset-style casing correction.",
    },
]

ABBREVIATION_CANDIDATES = {
    "nn", "ke", "k", "x", "cmq", "cmnq", "nov", "info", "dx", "sx", "tvb", "qnd", "qlc", "qlcs", "qlk", "xk", "xke", "xchè", "xche", "qst", "qsto", "qsta", "qsti", "qste",
}
ACCENT_CANDIDATES = {
    "e'", "E'", "é", "perchè", "perche'", "perche", "puo'", "puo", "pero'", "pero", "gia", "cosi", "piu", "po'", "citta", "così", "perché", "può", "però",
}
CASE_CONTEXT_CANDIDATES = {
    "governo", "Governo", "MONTI", "monti", "ROMA", "roma", "ITALIA", "italia", "MARIO", "mario", "pd", "Pd", "CGIL", "Cgil", "Cisl", "Uil", "SKY", "iphone", "App", "One", "one", "direction", "niall", "Berlusconi", "BERLUSCONI",
}
COMMON_ALLCAPS_RE = re.compile(r"^[A-ZÀÈÉÌÍÒÓÙÚ]{2,}$")
REPEATED_RE = re.compile(r"([A-Za-zÀ-ÿ])\1{2,}")
APOSTROPHE_RE = re.compile(r"^[A-Za-zÀ-ÿ]+['’]$")


def is_it_likely_candidate(tok: str) -> bool:
    s = str(tok).strip()
    if not s:
        return False
    if is_protected_token(s):
        return False
    low = s.lower()
    if s in ABBREVIATION_CANDIDATES or low in ABBREVIATION_CANDIDATES:
        return True
    if s in ACCENT_CANDIDATES or low in ACCENT_CANDIDATES:
        return True
    if s in CASE_CONTEXT_CANDIDATES or low in {x.lower() for x in CASE_CONTEXT_CANDIDATES}:
        return True
    if REPEATED_RE.search(s):
        return True
    if APOSTROPHE_RE.match(s):
        return True
    # All-caps ordinary words can be normalization candidates, but acronyms/entities are protected by prompt and post-guard.
    if len(s) >= 3 and COMMON_ALLCAPS_RE.match(s):
        return True
    return False


def candidate_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_it_likely_candidate(str(tok))]


def build_it_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=ITALIAN_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )


def build_it_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=ITALIAN_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )

# Backward-compatible aliases
build_detection_prompt = build_it_target_detection_prompt
build_normalization_prompt = build_it_target_normalization_prompt
