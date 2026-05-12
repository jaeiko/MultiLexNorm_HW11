"""Danish-specific lexical-normalization rules.

This module is intentionally data-driven. It provides candidate selection and
prompt wrappers for MultiLexNorm-style Danish normalization while relying on
prompts.common_prompt for target-only editing and protected-token handling.
"""
from __future__ import annotations

import re
from typing import Sequence

try:
    from prompts.common_prompt import (
        build_common_detection_prompt,
        build_common_normalization_prompt,
        is_protected_token,
    )
except ImportError:  # allow package-local imports
    from ..prompts.common_prompt import (
        build_common_detection_prompt,
        build_common_normalization_prompt,
        is_protected_token,
    )

LANG = "da"

DANISH_RULE_BLOCK = """
Danish-specific guidance:
- Normalize only minimal surface-form deviations under the dataset's annotation style.
- Do not translate or paraphrase Danish words; keep lexical meaning unchanged.
- Danish-specific letters are important: æ, ø, å may be missing or written as ae/oe/aa/plain ASCII.
- Common stable candidates include: vaere→være, vaeret→været, ogsa→også, paa/pa→på, sa/saa→så, ikk/ik/ek→ikke.
- Common q-style substitutions include jeq→jeg, oq→og, diq→dig, miq→mig, noqet→noget.
- Some forms may require splitting: idag→i dag, minpige→min pige.
- Abbreviations may be expanded or punctuated when supported by the dataset: pga→pga., fx→f.eks.
- Be conservative with very short tokens such as p, s, a, r, t, n, gr, fr, fa, pa, sa, ma: they are highly context-dependent.
- Do not normalize hashtags, mentions, URLs, emojis, numbers, times, alphanumeric entities, or names.
- If a token is a plausible named entity, username, title, brand, or event tag, preserve it unless the evidence supports only a minimal casing correction.
""".strip()

FEW_SHOT_EXAMPLES = [
    {
        "raw_sentence": "jeg kan ikke vaere med idag",
        "tokens": ["jeg", "kan", "ikke", "vaere", "med", "idag"],
        "target_index": 3,
        "target": "vaere",
        "label": 1,
        "normalized": "være",
        "notes": "vaere is a Danish ASCII spelling of være.",
    },
    {
        "raw_sentence": "jeg er pa vej",
        "tokens": ["jeg", "er", "pa", "vej"],
        "target_index": 2,
        "target": "pa",
        "label": 1,
        "normalized": "på",
        "notes": "pa is often normalized to på, but still check local context.",
    },
    {
        "raw_sentence": "jeq oq diq ses idag",
        "tokens": ["jeq", "oq", "diq", "ses", "idag"],
        "target_index": 0,
        "target": "jeq",
        "label": 1,
        "normalized": "jeg",
        "notes": "q-style spelling is normalized to standard Danish spelling.",
    },
    {
        "raw_sentence": "jeg er #TeamDK kl 20.00",
        "tokens": ["jeg", "er", "#TeamDK", "kl", "20.00"],
        "target_index": 2,
        "target": "#TeamDK",
        "label": 0,
        "normalized": "#TeamDK",
        "notes": "Hashtags are protected social-media entities and must be preserved.",
    },
    {
        "raw_sentence": "fx er det pga vejret",
        "tokens": ["fx", "er", "det", "pga", "vejret"],
        "target_index": 0,
        "target": "fx",
        "label": 1,
        "normalized": "f.eks.",
        "notes": "Dataset examples support abbreviation punctuation/expansion for fx.",
    },
]

# Strongly data-driven candidate lexicon. These are not all direct replacements;
# most should still be checked by MFR confidence or target-token prompt.
COMMON_DA_CANDIDATES = {
    "pa", "paa", "p", "sa", "saa", "s", "ogsa", "ogs", "ikk", "ik", "ek",
    "vaere", "vre", "vaeret", "vaek", "sadan", "ma", "gar", "nar", "ar", "har", "made",
    "taenker", "taenke", "hjaelp", "hjaelpe", "laekker", "naesten", "saeson", "desvaerre",
    "idag", "maneder", "hvornr", "forsta", "laenge", "haevn", "paen", "naeste",
    "glaede", "glaeder", "vaerd", "blir", "fler", "stte", "fet", "jalouxi", "minpige",
    "jeq", "oq", "diq", "miq", "noqet", "sku", "squ", "sq", "hva", "ka", "ha", "pga", "fx",
    "brn", "mde", "flelser", "nsten", "prvet", "kraefter", "laes", "arets", "mal",
}

CONTEXT_SENSITIVE = {
    "a", "p", "pa", "s", "sa", "ma", "r", "n", "t", "gr", "fr", "fa", "os", "sku", "la", "har", "ar", "nar", "nr", "ka", "ha", "va", "i", "o", "v", "j",
}

DANISH_ASCII_PATTERNS = [
    re.compile(r"ae", re.IGNORECASE),
    re.compile(r"oe", re.IGNORECASE),
    re.compile(r"aa", re.IGNORECASE),
]

Q_SUBSTITUTION_RE = re.compile(r"q", re.IGNORECASE)
REPEATED_RE = re.compile(r"([A-Za-zÆØÅæøå])\1{2,}")


def looks_like_danish_ascii_spelling(token: str) -> bool:
    s = token.strip()
    low = s.lower()
    if any(p.search(s) for p in DANISH_ASCII_PATTERNS):
        return True
    # common loss of Danish letters in data: ar→år, nar→når, gar→går, made→måde, etc.
    if low in {"ar", "nar", "gar", "har", "made", "mal", "la", "fa", "brn", "mde", "krft", "flelser", "nsten", "prvet"}:
        return True
    return False


def is_likely_danish_candidate(token: str) -> bool:
    if token is None:
        return False
    s = str(token).strip()
    if not s or is_protected_token(s):
        return False
    low = s.lower()
    if low in COMMON_DA_CANDIDATES:
        return True
    if looks_like_danish_ascii_spelling(s):
        return True
    if Q_SUBSTITUTION_RE.search(s) and re.search(r"[A-Za-zÆØÅæøå]", s):
        return True
    if REPEATED_RE.search(s):
        return True
    if re.search(r"^[a-z]{1,3}$", low) and low in CONTEXT_SENSITIVE:
        return True
    return False


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_likely_danish_candidate(str(tok))]


def build_da_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=DANISH_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )


def build_da_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        language_rule_block=DANISH_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )
