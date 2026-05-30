"""Indonesian and Indonesian-English code-mixed rules on the common prompt."""
from __future__ import annotations

import re
from typing import Sequence

try:
    from prompts.common_prompt import (
        build_common_detection_prompt,
        build_common_normalization_prompt,
        find_protected_indices,
        is_protected_token,
    )
except ImportError:  # pragma: no cover
    from common_prompt import (  # type: ignore
        build_common_detection_prompt,
        build_common_normalization_prompt,
        find_protected_indices,
        is_protected_token,
    )

LANGS = {"id", "iden"}

COMMON_ID_NORMALIZATION_CUES: dict[str, str] = {
    "yg": "yang", "aja": "saja", "bgt": "banget/sangat", "ga": "enggak/tidak", "gak": "enggak/tidak", "gk": "enggak", "nggak": "enggak/tidak",
    "kalo": "kalau", "klo": "kalau", "liat": "lihat", "pake": "pakai", "udah": "sudah", "udh": "sudah", "org": "orang",
    "jd": "jadi", "jg": "juga", "tp": "tapi", "sm": "sama", "dr": "dari", "utk": "untuk", "dlm": "dalam", "krn": "karena",
    "bs": "bisa", "bsa": "bisa", "ky": "kayak/seperti", "kyk": "kayak/seperti", "gini": "begini", "gitu": "begitu",
    "emg": "memang", "emang": "memang", "gue": "saya", "gua": "saya", "gw": "saya/gue", "lo": "kamu", "lu": "kamu",
    "kk": "kakak", "ka": "kak", "mba": "mbak", "ny": "nya", "nnti": "nanti", "kl": "kalau", "pnjang": "panjang", "bnr": "benar",
    "se7": "setuju", "wkt": "waktu",
}

AMBIGUOUS_OR_CONTEXTUAL_TOKENS = {
    "d", "k", "bg", "kt", "km", "da", "smp", "make", "kaya", "aja", "y", "n",
    "a", "b", "c", "e", "g", "i", "j", "l", "m", "o", "p", "q", "r", "s", "t", "w", "x", "z",
}

INDONESIAN_RULE_BLOCK = """
Indonesian-specific guidance:
- Indonesian social media uses many colloquial forms: yg->yang, aja->saja, bgt->banget/sangat, kalo/klo->kalau, liat->lihat, pake->pakai, udah/udh->sudah.
- Language code matters. In id, ga/gak often normalize to enggak; in iden, ga/gak often normalize to tidak.
- Pronouns such as gue/gua/gw/lo/lu are context-sensitive and differ between id and iden annotation style.
- Character repetition may be reduced when clearly lexical, e.g. cantikkk->cantik, bangettt->banget, but preserve expressive lengthening if uncertain.
- Reduplication markers can indicate plural/repetition: anak2->anak-anak, orang2->orang-orang, kata2nya->kata-katanya. Avoid naive rules for forms like bersama2.
- Informal affixes (-ny, -x, -'y, -q, ng-/nge-) are candidates, not automatic replacements, unless dictionary evidence is strong.
- In iden, English contractions such as i'm/im->i am, it's/its->it is, dont/don't->do not may normalize, but ordinary English words, titles, names, hashtags, and mentions should be preserved.
- Do not translate named entities, product names, hashtags, mentions, URLs, numbers, or alphanumeric entities.
""".strip()

FEW_SHOT_EXAMPLES: dict[str, list[dict[str, object]]] = {
    "id": [
        {"tokens": ["aku", "gak", "liat", "yg", "baru"], "target_index": 1, "target": "gak", "label": 1, "normalized": "enggak", "notes": "Indonesian-only style: gak often normalizes to enggak."},
        {"tokens": ["aku", "gak", "liat", "yg", "baru"], "target_index": 2, "target": "liat", "label": 1, "normalized": "lihat", "notes": "Colloquial clipping."},
        {"tokens": ["anak2", "itu", "cantikkk", "bgt"], "target_index": 0, "target": "anak2", "label": 1, "normalized": "anak-anak", "notes": "Reduplication marker."},
        {"tokens": ["SMP", "Negeri", "1"], "target_index": 0, "target": "SMP", "label": 0, "normalized": "SMP", "notes": "Acronym/entity-like token is protected; do not normalize to sampai."},
    ],
    "iden": [
        {"tokens": ["i'm", "gak", "sure", "yg", "itu"], "target_index": 0, "target": "i'm", "label": 1, "normalized": "i am", "notes": "English contraction in code-mixed text."},
        {"tokens": ["i'm", "gak", "sure", "yg", "itu"], "target_index": 1, "target": "gak", "label": 1, "normalized": "tidak", "notes": "In iden, ga/gak often normalize to tidak."},
        {"tokens": ["The", "Lion", "King", "bagus", "bgt"], "target_index": 0, "target": "The", "label": 0, "normalized": "The", "notes": "Title/name-like English token is preserved."},
        {"tokens": ["#Movie2024", "bagus", "bgt"], "target_index": 0, "target": "#Movie2024", "label": 0, "normalized": "#Movie2024", "notes": "Hashtag is protected."},
    ],
}


def _validate_lang(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"Unsupported Indonesian module lang={lang!r}; expected one of {sorted(LANGS)}")
    return lang


def has_repeated_characters(token: str) -> bool:
    return bool(re.search(r"([A-Za-z])\1{2,}", token or ""))


def has_reduplication_marker(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]+(?:2|['\"]{1,2})[A-Za-z]*$", token or ""))


def looks_like_informal_affix(token: str) -> bool:
    lower = (token or "").lower()
    return (len(lower) > 3 and re.search(r"(?:ny|x|'y|q)$", lower) is not None) or (len(lower) > 4 and lower.startswith(("nge", "ng")))


def has_vowel_deletion_or_digit_substitution(token: str) -> bool:
    lower = (token or "").lower()
    if re.search(r"\d", lower) and re.search(r"[a-z]", lower):
        return True
    return lower in {"yg", "jd", "jg", "tp", "dr", "utk", "dlm", "krn", "bsa", "wkt", "bnr", "nnti", "kl"}


def is_english_contraction_candidate(token: str) -> bool:
    return (token or "").lower() in {"i'm", "im", "it's", "its", "dont", "don't", "you're", "youre", "gonna", "wanna"}


def is_id_likely_candidate(token: str, *, lang: str = "id", tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    lang = _validate_lang(lang)
    if not token or is_protected_token(token):
        return False
    lower = token.lower()
    if lower in COMMON_ID_NORMALIZATION_CUES:
        return True
    if lang == "iden" and is_english_contraction_candidate(token):
        return True
    if has_repeated_characters(token):
        return True
    if has_reduplication_marker(token):
        return True
    if looks_like_informal_affix(token):
        return True
    if has_vowel_deletion_or_digit_substitution(token):
        return True
    if lower in AMBIGUOUS_OR_CONTEXTUAL_TOKENS:
        return False
    return False


def candidate_indices(sentence_tokens: Sequence[str], *, lang: str = "id") -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_id_likely_candidate(tok, lang=lang, tokens=sentence_tokens, index=i)]


def build_id_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, lang: str = "id", raw_sentence: str | None = None) -> str:
    lang = _validate_lang(lang)
    return build_common_detection_prompt(
        lang=lang,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=INDONESIAN_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES[lang],
    )


def build_id_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, lang: str = "id", raw_sentence: str | None = None) -> str:
    lang = _validate_lang(lang)
    return build_common_normalization_prompt(
        lang=lang,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=INDONESIAN_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES[lang],
    )

build_id_target_prompt = build_id_target_detection_prompt
build_id_normalization_prompt = build_id_target_normalization_prompt

__all__ = [
    "LANGS", "INDONESIAN_RULE_BLOCK", "COMMON_ID_NORMALIZATION_CUES", "AMBIGUOUS_OR_CONTEXTUAL_TOKENS", "FEW_SHOT_EXAMPLES",
    "has_repeated_characters", "has_reduplication_marker", "looks_like_informal_affix", "has_vowel_deletion_or_digit_substitution",
    "is_english_contraction_candidate", "is_id_likely_candidate", "candidate_indices", "build_id_target_detection_prompt",
    "build_id_target_normalization_prompt", "build_id_target_prompt", "build_id_normalization_prompt",
]
