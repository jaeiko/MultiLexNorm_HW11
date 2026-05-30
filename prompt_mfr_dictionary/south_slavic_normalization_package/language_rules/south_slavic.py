"""Shared South Slavic rule block for sr/hr/sl.

This module contains only language-specific cues and candidate selection.
The common prompt contract is defined in prompts/common_prompt.py.
"""
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

SUPPORTED_LANGS = {"sr", "hr", "sl"}

LATIN_WORD_RE = re.compile(r"^[A-Za-zÀ-ž][A-Za-zÀ-ž'’.-]*$")
REPEATED_LATIN_RE = re.compile(r"([A-Za-zÀ-ž])\1{2,}")
DIACRITIC_LESS_RE = re.compile(r"[cszCSZ]|(?:dj|DJ|Dj)")
APOSTROPHE_OMISSION_RE = re.compile(r"[A-Za-zÀ-ž]+['’]$")

SOUTH_SLAVIC_RULE_BLOCK = """
South Slavic shared guidance for Serbian (sr), Croatian (hr), and Slovenian (sl):
- These datasets are Twitter/CMC lexical-normalization data. Apply minimal lexical intervention only.
- Do not correct syntax, word order, agreement, style, punctuation, usernames, hashtags, URLs, emojis, or ellipsis.
- Normalize spelling/orthographic deviations when dataset evidence supports them: missing diacritics, final vowel dropping, phonetic spelling, obvious typos, shortened function words, abbreviations, and repeated letters.
- Missing diacritics are common: c→č/ć, s→š, z→ž, dj/dj→đ may be needed, but choose the target only from context and language-specific examples.
- Character repetitions in words/interjections may be reduced, but laughter and expressive repetitions should be treated conservatively.
- Colloquial words should not be semantically translated into standard synonyms unless the dataset-style examples show that exact lexical replacement.
- Foreign words, names, and code-mixed tokens should not be translated. If a language-specific guideline says to normalize a foreign spelling variant, keep the closest surface form rather than translating its meaning.
- Context-sensitive short tokens such as ko, bi, jel, k, sm, and sam must not be resolved without context.
""".strip()

LANGUAGE_OVERRIDES = {
    "sr": """
Serbian-specific guidance:
- Shared-task Serbian data is Latin-script; do not convert Latin to Cyrillic.
- Strong cues include diacritic restoration: sto→što, sta→šta, vise→više, nesto→nešto, jos→još, cu→ću, ce→će.
- Common abbreviations include jbg→jebiga, jbt→jebote, aj→hajde, al→ali.
- Serbian often keeps phonetically transcribed foreign words/names; do not force Croatian-style original spelling.
- ko, bi, jel, l', k'o and similar short forms are context-sensitive.
""".strip(),
    "hr": """
Croatian-specific guidance:
- Strong cues include sto/sta/šta→što, cu→ću, ce→će, jos→još, al→ali, ak→ako, di→gdje, kak→kako, san→sam, uvik→uvijek.
- Croatian often normalizes final-vowel dropping and short infinitives: gledat→gledati, smislit→smisliti, pofotkat→pofotkati.
- Croatian may normalize phonetically transcribed foreign words to original spelling, e.g. fešn→fashion, fejsbuk→facebook, tviteraš→twitteraš, when examples support it.
- ko is context-sensitive: it can mean kao or tko. Do not resolve it without context.
- Do not translate colloquial lexical choices such as komp into standard semantic equivalents unless exact dataset evidence supports it.
""".strip(),
    "sl": """
Slovenian-specific guidance:
- Slovenian nonstandard tweets often involve orthographic variation, missing diacritics, vowel dropping, missing spaces, and colloquial spellings.
- Strong cues include sm→sem, tud→tudi, blo→bilo, sej→saj, jst/js→jaz, al→ali, tko→tako, mal→malo, zdej→zdaj, kej→kaj.
- k is highly ambiguous and can map to ko, ki, kot, ker, kam, etc.; leave it unchanged if context is insufficient.
- sm and sam are context-sensitive in short contexts, even though sm→sem and sam→samo are frequent.
- Foreign elements adapted to Slovenian morphology should be normalized to the most frequent adapted spelling variant, not translated into English.
- Mentions, hashtags, URLs, emoticons and emojis are exempt from normalization.
""".strip(),
}

CUES = {
    "sr": {
        "sto", "sta", "vise", "nesto", "cu", "ce", "jos", "zasto", "nece", "moze", "necu", "bas", "kaze", "znaci",
        "jbg", "jbt", "aj", "al", "kuci", "zivot", "skole", "kauc", "decka", "mrs", "dovidjenja", "budzet", "ko", "bi", "jel", "l'", "k'o",
    },
    "hr": {
        "sto", "sta", "šta", "cu", "ce", "jos", "vise", "al", "ak", "di", "kak", "tak", "san", "uvik", "gledat", "smislit",
        "pofotkat", "moš", "nes", "neš", "isprid", "cili", "posudit", "neko", "kauc", "noz", "tviteras", "fešn", "fejsbuk", "ko", "bi", "jel", "dal", "kaj",
    },
    "sl": {
        "sm", "tud", "blo", "sej", "jst", "js", "al", "tko", "mal", "zdej", "tut", "mam", "kej", "dobr", "ce", "pr", "lahk", "nč", "zarad", "dons", "vec", "sreco", "motis", "druzbeno", "skodo", "javla", "k", "sam", "ma", "neki",
    },
}

PRESERVE_BY_DEFAULT = {
    "lol", "LOL", "haha", "hahaha", "hehe", "ahah", "xD", "XD",
}


def get_rule_block(lang: str) -> str:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported South Slavic language: {lang}")
    return SOUTH_SLAVIC_RULE_BLOCK + "\n\n" + LANGUAGE_OVERRIDES[lang]


def has_latin_word_shape(token: str) -> bool:
    return bool(LATIN_WORD_RE.fullmatch(token or ""))


def has_repeated_latin(token: str) -> bool:
    return bool(REPEATED_LATIN_RE.search(token or ""))


def looks_diacriticless_candidate(token: str) -> bool:
    if not has_latin_word_shape(token):
        return False
    # Require at least two letters and one ASCII letter that often stands for a diacritic.
    return len(token) >= 2 and bool(DIACRITIC_LESS_RE.search(token))


def is_common_candidate(token: str, lang: str) -> bool:
    t = str(token)
    if not t or is_protected_token(t):
        return False
    if t in PRESERVE_BY_DEFAULT:
        return False
    tl = t.lower()
    if tl in CUES.get(lang, set()):
        return True
    if has_repeated_latin(t):
        return True
    if APOSTROPHE_OMISSION_RE.fullmatch(t):
        return True
    if looks_diacriticless_candidate(t) and len(t) <= 14:
        # Candidate only. The prompt/model decides whether it really changes.
        return True
    return False


def candidate_indices(sentence_tokens: Sequence[str], lang: str) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_common_candidate(str(tok), lang)]


def build_target_detection_prompt(lang: str, sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, fewshot_examples=None) -> str:
    return build_common_detection_prompt(
        lang=lang,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=get_rule_block(lang),
        fewshot_examples=fewshot_examples,
    )


def build_target_normalization_prompt(lang: str, sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None, fewshot_examples=None) -> str:
    return build_common_normalization_prompt(
        lang=lang,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=get_rule_block(lang),
        fewshot_examples=fewshot_examples,
    )


__all__ = [
    "SUPPORTED_LANGS", "SOUTH_SLAVIC_RULE_BLOCK", "LANGUAGE_OVERRIDES", "CUES", "PRESERVE_BY_DEFAULT",
    "get_rule_block", "has_latin_word_shape", "has_repeated_latin", "looks_diacriticless_candidate",
    "is_common_candidate", "candidate_indices", "build_target_detection_prompt", "build_target_normalization_prompt",
]
