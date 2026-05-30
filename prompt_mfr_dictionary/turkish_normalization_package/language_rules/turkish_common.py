"""Shared Turkish normalization cues for tr and trde."""
from __future__ import annotations

import re
from typing import Sequence

try:
    from prompts.common_prompt import is_protected_token
except Exception:  # pragma: no cover
    from ..prompts.common_prompt import is_protected_token

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
ASCII_TURKISH_SIGNAL = re.compile(r"(?:cok|degil|guzel|nasil|icin|hic|boyle|artik|baska|insallah|canim|ask|sey|suan)", re.I)
VOWELLESS_SIGNAL = re.compile(r"^[bcçdfgğhjklmnprsştvyzBCÇDFGĞHJKLMNPRSŞTVYZ]{3,}$")
REPETITION_RE = re.compile(r"(.)\1{2,}", re.UNICODE)
TURKISH_SPOKEN_SUFFIX_RE = re.compile(
    r"(iyom|iyon|iyosun|iyo|iyolar|cem|cam|caz|cez|yom|yon|yolar|miyon|miyom|musun|müsün|dicem|ıcam|icem|ucam|ücem)$",
    re.I,
)
CLITIC_SPACING_FORMS = {
    "bide", "bende", "sende", "bizde", "sizde", "yada", "suan", "bisey", "bişey", "hersey", "herşey",
    "yokmu", "varmi", "varmı", "naber", "nolur", "napıyon", "napiyon", "geliyonmu", "gidiyonmu",
}
COMMON_TR_ABBREVIATIONS = {
    "bi", "cnm", "tmm", "slm", "nbr", "kib", "amk", "aq", "mk", "yaaa", "yaa", "valla", "inş", "ins", "mrb",
}
PROPER_SUFFIX_RE = re.compile(
    r"^[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]+(?:da|de|dan|den|ta|te|tan|ten|ya|ye|a|e|ı|i|u|ü|nın|nin|nun|nün|ın|in|un|ün)$"
)

TURKISH_COMMON_RULE_BLOCK = """
Turkish-family guidance:
- Turkish is morphologically rich and agglutinative; do not assume a long or suffixed word is non-standard.
- Important normalization cues include deasciification, vowel restoration, spoken/accented suffix normalization, clitic separation, proper-noun apostrophe restoration, and repeated-character reduction.
- Deasciification examples: cok→çok, degil→değil, guzel→güzel, nasil→nasıl, icin→için, hic→hiç.
- Vowel restoration examples: cnm→canım, tmm→tamam, nbr→ne haber.
- Spoken/accent forms may require canonical written forms: geliyo→geliyor, gidicem→gideceğim, yapcam→yapacağım.
- Clitic/spacing examples: bide→bir de, bende→ben de, yada→ya da, bisey→bir şey, suan→şu an.
- Proper nouns may require capitalization and apostrophe restoration, e.g. almanyada→Almanya'da. Apply this only when context supports a proper noun.
- Repetition can mark emphasis; normalize only when the dataset style supports reduction.
- Preserve Twitter-specific/protected tokens: hashtags, mentions, URLs, emoticons, RT, DM, numbers, and alphanumeric entities.
""".strip()


def has_turkish_deasciification_signal(tok: str) -> bool:
    s = tok.strip()
    if not s or is_protected_token(s):
        return False
    if TURKISH_CHARS.intersection(s):
        return False
    return bool(ASCII_TURKISH_SIGNAL.search(s))


def is_turkish_likely_candidate(tok: str) -> bool:
    s = str(tok).strip()
    if not s or is_protected_token(s):
        return False
    low = s.lower()
    if low in CLITIC_SPACING_FORMS or low in COMMON_TR_ABBREVIATIONS:
        return True
    if has_turkish_deasciification_signal(s):
        return True
    if VOWELLESS_SIGNAL.fullmatch(s) and len(s) <= 8:
        return True
    if TURKISH_SPOKEN_SUFFIX_RE.search(s):
        return True
    if REPETITION_RE.search(s):
        return True
    if PROPER_SUFFIX_RE.match(s):
        return True
    if any(ch in s for ch in "$ßµ"):
        return True
    return False


def candidate_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_turkish_likely_candidate(str(tok))]
