"""Italian v2 language-specific rules and prompt builders.

This module is paper-guided and data-driven. It is designed for a pipeline where
trigram and MFR are applied first, XLM-R detects likely targets, and the LLM is
used only for untouched candidates. The main v2 addition is Italian-specific
candidate injection and more precise negative guidance for capitalization,
hashtags, phrasal abbreviations, clitics, and preposition contractions.
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
Italian-specific guidance, v2:
- This is Italian lexical normalization, not translation, paraphrase, grammar correction, or readability rewriting.
- Normalize only the target token/span. Do not rewrite the full sentence.
- Focus on Italian social-media surface normalization:
  * abbreviations: nn → non, ke → che, cn → con, cmq → comunque, dx → destra, sx → sinistra
  * accent/apostrophe restoration: e' → è, perche' / perchè → perché, puo' → può, pero' → però, gia → già, cosi → così, piu → più
  * repeated-character reduction when clearly expressive: Beppeee → Beppe, ciaoooo → ciao
  * split/merge only when the target span itself requires it: Vabbene → va bene, un ultima → un'ultima
- Capitalization is part of Italian normalization, but it is risky:
  * Correct clear ordinary all-caps emphasis when context supports it, e.g. NON → non.
  * Preserve acronyms, party names, organization names, brands, person/place names, and mixed-case products.
  * If capitalization is doubtful, preserve the original target.
- Keep Italian clitics and personal pronoun forms when they are already standard:
  mi → mi, ci → ci, arrendermi → arrendermi. Do not expand mi/ci to "a me"/"a noi".
- Keep contracted prepositions when they are standard:
  del, della, alla, sull', dell' should not be split into di il, di la, a la, etc.
- Phrasal abbreviations and interjections/non-words are not automatically expanded:
  omg → omg, lol → lol, ahahah → ahahah, Bhe → Bhe unless project data provides exact evidence.
- Preserve hashtags, usernames, URLs, emails, emojis, numbers, dates, times, product names, usernames,
  party/organization acronyms, song titles, group names, and alphanumeric entities.
- For digit shorthand such as 6 → sei, use context and project evidence. In the shared pipeline numbers are protected by default.
- If multiple normalized forms are plausible or the token may be an entity, choose preserve.
""".strip()

FEWSHOT_EXAMPLES = [
    {
        "raw_sentence": "joker cmq nn e' nnt di ke !",
        "tokens": ["joker", "cmq", "nn", "e'", "nnt", "di", "ke", "!"],
        "target_index": 1,
        "target": "cmq",
        "label": 1,
        "normalized": "comunque",
        "notes": "cmq is a common Italian social-media abbreviation for comunque.",
    },
    {
        "raw_sentence": "joker cmq nn e' nnt di ke !",
        "tokens": ["joker", "cmq", "nn", "e'", "nnt", "di", "ke", "!"],
        "target_index": 2,
        "target": "nn",
        "label": 1,
        "normalized": "non",
        "notes": "nn is a stable abbreviation for non.",
    },
    {
        "raw_sentence": "joker cmq nn e' nnt di ke !",
        "tokens": ["joker", "cmq", "nn", "e'", "nnt", "di", "ke", "!"],
        "target_index": 3,
        "target": "e'",
        "label": 1,
        "normalized": "è",
        "notes": "apostrophe-based accent notation should be normalized to è.",
    },
    {
        "raw_sentence": "nn so perche' e' successo",
        "tokens": ["nn", "so", "perche'", "e'", "successo"],
        "target_index": 2,
        "target": "perche'",
        "label": 1,
        "normalized": "perché",
        "notes": "restore the standard accented final vowel.",
    },
    {
        "raw_sentence": "ke fai x cena ?",
        "tokens": ["ke", "fai", "x", "cena", "?"],
        "target_index": 0,
        "target": "ke",
        "label": 1,
        "normalized": "che",
        "notes": "ke is a spelling variant for che.",
    },
    {
        "raw_sentence": "ke fai x cena ?",
        "tokens": ["ke", "fai", "x", "cena", "?"],
        "target_index": 2,
        "target": "x",
        "label": 1,
        "normalized": "per",
        "notes": "x can mean per, but only normalize when used as a word-level Italian shorthand, not as a symbol or hashtag part.",
    },
    {
        "raw_sentence": "un ultima occhiata",
        "tokens": ["un", "ultima", "occhiata"],
        "target_index": 0,
        "target": "un ultima",
        "label": 1,
        "normalized": "un'ultima",
        "notes": "A determiner plus vowel-initial feminine noun may require contraction; treat as a target span, not a full-sentence rewrite.",
    },
    {
        "raw_sentence": "mi arrendermi adesso",
        "tokens": ["mi", "arrendermi", "adesso"],
        "target_index": 0,
        "target": "mi",
        "label": 0,
        "normalized": "mi",
        "notes": "Italian clitics are standard and must not be expanded to a me.",
    },
    {
        "raw_sentence": "la CGIL parla con il PD",
        "tokens": ["la", "CGIL", "parla", "con", "il", "PD"],
        "target_index": 1,
        "target": "CGIL",
        "label": 0,
        "normalized": "CGIL",
        "notes": "Organization and party acronyms are protected.",
    },
    {
        "raw_sentence": "seguo #OroRosso e @utente",
        "tokens": ["seguo", "#OroRosso", "e", "@utente"],
        "target_index": 1,
        "target": "#OroRosso",
        "label": 0,
        "normalized": "#OroRosso",
        "notes": "Hashtags and usernames are preserved in this project pipeline.",
    },
    {
        "raw_sentence": "omg ahahah che giornata",
        "tokens": ["omg", "ahahah", "che", "giornata"],
        "target_index": 0,
        "target": "omg",
        "label": 0,
        "normalized": "omg",
        "notes": "Phrasal abbreviations and non-words are not automatically expanded.",
    },
]

ABBREVIATION_CANDIDATES = {
    "nn", "NN", "ke", "Ke", "KE", "cn", "CN", "cmq", "Cmq", "CMQ", "cmnq",
    "nnt", "nov", "info", "dx", "sx", "qnd", "qlc", "qlcn", "qlcs",
    "qst", "qsto", "qsta", "qsti", "qste",
    "x", "X", "xk", "xke", "xké", "xchè", "xche", "Xke", "Xké",
}
ACCENT_CANDIDATES = {
    "e'", "e’", "E'", "E’", "é",
    "perchè", "perche'", "perche’", "perche",
    "puo'", "puo’", "puo",
    "pero'", "pero’", "pero",
    "gia", "cosi", "piu", "citta",
}
CASE_CONTEXT_CANDIDATES = {
    "governo", "Governo", "GOVERNO",
    "MONTI", "monti", "ROMA", "roma", "ITALIA", "italia", "MARIO", "mario",
    "pd", "Pd", "PD", "CGIL", "Cgil", "CISL", "Cisl", "UIL", "Uil",
    "RAI", "Rai", "BCE", "Bce", "TG1", "Tg1", "iphone", "iPhone",
    "Berlusconi", "BERLUSCONI",
}
PRESERVE_BY_DEFAULT = {
    "lol", "LOL", "omg", "OMG", "ahahah", "ahah", "Bhe", "bhe",
    "tvb", "tvtb", "xoxo",
}
ITALIAN_ACRONYMS = {"PD", "CGIL", "CISL", "UIL", "RAI", "TG1", "BCE", "UE", "ONU", "USA", "EU"}

COMMON_ALLCAPS_RE = re.compile(r"^[A-ZÀÈÉÌÍÒÓÙÚ]{2,}$")
REPEATED_RE = re.compile(r"([A-Za-zÀ-ÿ])\1{2,}")
APOSTROPHE_RE = re.compile(r"^[A-Za-zÀ-ÿ]+['’]$")
HAS_VOWEL_INITIAL_RE = re.compile(r"^[aeiouAEIOUàèéìòóùÀÈÉÌÒÓÙ]")


def is_entity_like(tok: str) -> bool:
    s = str(tok).strip()
    if not s:
        return False
    if s in ITALIAN_ACRONYMS:
        return True
    if s.startswith("#") or s.startswith("@"):
        return True
    if re.search(r"[a-zà-ÿ][A-Z]", s):  # iPhone, eBay
        return True
    if re.fullmatch(r"[A-Z]{2,}[0-9]*", s) and s not in {"NON", "CHE", "PER", "DEL", "DEI", "CON"}:
        return True
    return False


def is_it_likely_candidate(tok: str) -> bool:
    s = str(tok).strip()
    if not s:
        return False
    if is_protected_token(s):
        return False
    if s in PRESERVE_BY_DEFAULT:
        return False
    low = s.lower()

    if s in ABBREVIATION_CANDIDATES or low in {x.lower() for x in ABBREVIATION_CANDIDATES}:
        return True
    if s in ACCENT_CANDIDATES or low in {x.lower() for x in ACCENT_CANDIDATES}:
        return True
    if s in CASE_CONTEXT_CANDIDATES or low in {x.lower() for x in CASE_CONTEXT_CANDIDATES}:
        return True
    if REPEATED_RE.search(s):
        return True
    if APOSTROPHE_RE.match(s):
        return True
    if len(s) >= 3 and COMMON_ALLCAPS_RE.match(s) and not is_entity_like(s):
        return True
    return False


def candidate_indices(tokens: Sequence[str]) -> list[int]:
    """Return token-level candidates for XLM-R/LLM fallback injection."""
    return [i for i, tok in enumerate(tokens) if is_it_likely_candidate(str(tok))]


def candidate_spans(tokens: Sequence[str]) -> list[tuple[int, int, str]]:
    """Return optional span-level candidates as (start, end_exclusive, reason).

    Useful for incorrect splitting/merging cases such as:
    - Vabbene → va bene
    - un ultima → un'ultima
    - un occhiata → un'occhiata
    """
    spans: list[tuple[int, int, str]] = []
    n = len(tokens)

    for i, tok in enumerate(tokens):
        s = str(tok)
        low = s.lower()
        if is_protected_token(s):
            continue
        if low in {"vabbene", "vabene"}:
            spans.append((i, i + 1, "merge_to_va_bene"))
        if is_it_likely_candidate(s):
            spans.append((i, i + 1, "single_token_candidate"))

    # determiner contraction candidates
    for i in range(n - 1):
        a = str(tokens[i])
        b = str(tokens[i + 1])
        if is_protected_token(a) or is_protected_token(b):
            continue
        if a.lower() in {"un", "l", "dell", "all"} and HAS_VOWEL_INITIAL_RE.match(b):
            spans.append((i, i + 2, "possible_apostrophe_contraction"))

    # Deduplicate while keeping order
    seen: set[tuple[int, int, str]] = set()
    uniq: list[tuple[int, int, str]] = []
    for sp in spans:
        key = sp
        if key not in seen:
            seen.add(key)
            uniq.append(sp)
    return uniq


def build_it_target_detection_prompt(
    sentence_tokens: Sequence[str],
    target_index: int,
    *,
    raw_sentence: str | None = None,
    protected_indices: Sequence[int] | None = None,
) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=ITALIAN_RULE_BLOCK,
        fewshot_examples=FEWSHOT_EXAMPLES,
    )


def build_it_target_normalization_prompt(
    sentence_tokens: Sequence[str],
    target_index: int,
    *,
    raw_sentence: str | None = None,
    protected_indices: Sequence[int] | None = None,
) -> str:
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
