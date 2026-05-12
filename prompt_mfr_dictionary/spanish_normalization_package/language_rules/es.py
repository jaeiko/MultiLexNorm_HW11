"""Spanish language-specific rules and prompt builders.

This module is data-driven and designed for MultiLexNorm-style lexical normalization.
It focuses on minimal surface normalization: accent restoration, social-media abbreviations,
phonetic spelling, laughter normalization, repeated letters, and selected split/expansion patterns.
It must not translate, paraphrase, rewrite, or normalize protected social-media entities.
"""
from __future__ import annotations

import re
from typing import Sequence

from prompts.common_prompt import (
    build_common_detection_prompt,
    build_common_normalization_prompt,
    is_protected_token,
)

LANG = "es"

SPANISH_RULE_BLOCK = """
Spanish-specific guidance:
- Focus on dataset-style lexical normalization, not translation, paraphrase, grammar correction, or style correction.
- Common project-data variants include accent restoration: tambien → también, despues → después, aqui → aquí, alli → allí, corazon → corazón, cafe → café, mio/mia/tio/pais → mío/mía/tío/país.
- Common abbreviation and colloquial expansions include: pa → para, cn → con, tb/tmb → también, esq/esque/esqe → es_que, finde → fin_de_semana.
- Que-family variants are context-sensitive: q/k/ke/qe may normalize to que or qué depending on context. Do not choose automatically without context.
- Si-family variants are context-sensitive: si can mean if or sí; elongated forms such as sii/siii often normalize to sí when used affirmatively.
- Repeated letters are often reduced when they mark emphasis: noo → no, buenoo → bueno, graciias → gracias, laa → la, muchoo → mucho.
- Laughter variants such as jajaj, jajajaj, jajajajaj are often normalized to ja in this dataset, but do not normalize hashtags/usernames containing such strings.
- Informal spellings such as kiero/qiero → quiero, estoi → estoy, voi → voy can be candidates when supported by context/data.
- Preserve hashtags, mentions, URLs, emojis, numbers, product names, usernames, all-caps acronyms, and alphanumeric entities.
- If uncertain between multiple normalized forms, preserve the original target or return the most conservative dataset-style surface form only when context strongly supports it.
""".strip()

FEWSHOT_EXAMPLES = [
    {
        "raw_sentence": "yo tambien kiero ir",
        "tokens": ["yo", "tambien", "kiero", "ir"],
        "target_index": 1,
        "target": "tambien",
        "label": 1,
        "normalized": "también",
        "notes": "tambien is a stable missing-accent variant of también.",
    },
    {
        "raw_sentence": "yo tambien kiero ir",
        "tokens": ["yo", "tambien", "kiero", "ir"],
        "target_index": 2,
        "target": "kiero",
        "label": 1,
        "normalized": "quiero",
        "notes": "kiero is a project-data phonetic spelling of quiero.",
    },
    {
        "raw_sentence": "pa mi eso no es tan claro",
        "tokens": ["pa", "mi", "eso", "no", "es", "tan", "claro"],
        "target_index": 0,
        "target": "pa",
        "label": 1,
        "normalized": "para",
        "notes": "pa is commonly expanded to para when context supports it.",
    },
    {
        "raw_sentence": "noo jajajaj #Fiesta2024 @amigo",
        "tokens": ["noo", "jajajaj", "#Fiesta2024", "@amigo"],
        "target_index": 2,
        "target": "#Fiesta2024",
        "label": 0,
        "normalized": "#Fiesta2024",
        "notes": "Hashtags are protected social-media entities and should be preserved.",
    },
    {
        "raw_sentence": "si vienes dime que hora es",
        "tokens": ["si", "vienes", "dime", "que", "hora", "es"],
        "target_index": 0,
        "target": "si",
        "label": 0,
        "normalized": "si",
        "notes": "si may be the conjunction 'if'; do not add an accent without context indicating affirmative sí.",
    },
]

ACCENT_CANDIDATES = {
    "tambien", "despues", "aqui", "alli", "corazon", "cafe", "mio", "mia", "tio", "pais", "estan", "habra", "tendre", "muchisimo", "rapidas", "romeria", "movil", "bateria", "adios", "egocentrico", "maricon", "lio", "rie"
}

ABBREVIATION_CANDIDATES = {
    "q", "k", "ke", "qe", "xq", "pq", "porq", "pa", "cn", "tb", "tmb", "esq", "esque", "esqe", "finde", "x", "dspue", "tds", "insti", "peli"
}

PHONETIC_CANDIDATES = {
    "kiero", "qiero", "voi", "estoi", "porai", "queva", "qur", "aseh", "pasao", "pnsao", "llege", "pokitin"
}

AMBIGUOUS_SHORT = {"si", "q", "k", "ke", "qe", "pa", "to", "ma", "na", "d", "x", "mi", "tu", "tuu"}

REPEATED_RE = re.compile(r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])\1{2,}")
LAUGHTER_RE = re.compile(r"^(?:j+a+)+j*a*$", re.IGNORECASE)


def is_es_likely_candidate(tok: str) -> bool:
    s = str(tok).strip()
    if not s:
        return False
    if is_protected_token(s):
        return False
    low = s.lower()
    if low in ACCENT_CANDIDATES or low in ABBREVIATION_CANDIDATES or low in PHONETIC_CANDIDATES:
        return True
    if low in AMBIGUOUS_SHORT:
        return True
    if REPEATED_RE.search(s):
        return True
    if LAUGHTER_RE.match(s) and len(s) >= 4:
        return True
    # Missing accents in common ASCII-only Spanish words: conservative prompt candidate only.
    if any(ch in low for ch in ["á", "é", "í", "ó", "ú", "ñ", "ü"]):
        return False
    if low.endswith(("cion", "mente", "isimo")):
        return True
    return False


def candidate_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_es_likely_candidate(str(tok))]


def build_es_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=SPANISH_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )


def build_es_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=SPANISH_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )

# Backward-compatible aliases
build_detection_prompt = build_es_target_detection_prompt
build_normalization_prompt = build_es_target_normalization_prompt
