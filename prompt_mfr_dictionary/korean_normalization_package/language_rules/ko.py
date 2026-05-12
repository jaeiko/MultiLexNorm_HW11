"""Korean-specific rules built on top of the common target-only prompt."""
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

LANG = "ko"

KOREAN_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")
REPEATED_KO_RE = re.compile(r"([가-힣ㄱ-ㅎㅏ-ㅣ])\1{2,}")
DIGIT_LATIN_MIXED_KO_RE = re.compile(r"(?=.*[가-힣ㄱ-ㅎㅏ-ㅣ])(?=.*[A-Za-z0-9])")
LAUGHTER_EMOTION_RE = re.compile(r"^(?:[ㅋㅎ]{1,}|[ㅠㅜ]+|ㅇㅇ|ㄴㄴ|ㄷㄷ|ㄱㄱ|ㅊㅋ|ㄱㅅ|ㅇㅋ)$")

COMMON_KO_NORMALIZATION_CUES: dict[str, str] = {
    "ㄹㅇ": "진짜", "ㅇㄱㄹㅇ": "진짜", "ㅅㅂ": "이런", "ㅆㅂ": "이런", "시발": "이런", "씨발": "이런",
    "ㅈㄴ": "매우", "존나": "매우", "좆나": "매우", "존1나": "매우",
    "걍": "그냥", "병신": "바보", "병신들": "바보들", "새끼들": "친구들", "새끼": "친구",
    "한녀": "한국여자", "한남": "한국남자", "짱깨": "중국", "조센징": "한국인", "조센": "한국",
    "중딩": "중학생", "초딩": "초등학생", "고딩": "고등학생", "커엽": "귀엽", "노잼": "재미없음",
    "비추": "비추천", "존잘": "매우 잘생김", "킹치만": "하지만",
}

PRESERVE_OR_CONTEXTUAL = {
    "ㅋㅋ", "ㅋㅋㅋ", "ㅋ", "ㅎㅎ", "ㅎㅎㅎ", "ㅎ", "ㅠㅠ", "ㅜㅜ", "ㅠ", "ㅜ",
    "ㅇㅇ", "ㄴㄴ", "ㄷㄷ", "ㄱㄱ", "ㅊㅋ", "ㄱㅅ", "ㅇㅋ", "ㅇㄷ",
    "노", "누", "긔", "임", "함", "듯", "듯요",
}

KOREAN_RULE_BLOCK = """
Korean-specific guidance:
- Korean social-media text often contains intentional text gaming, Yaminjeongeum, leetspeak, profanity avoidance, memes, and community jargon.
- Do NOT mark a token as needing normalization merely because it is slang-like, profane, expressive, meme-like, or noisy-looking.
- Stable empirical mappings include ㄹㅇ->진짜, ㅅㅂ/ㅆㅂ/시발/씨발->이런, ㅈㄴ->매우, 걍->그냥, 병신->바보, 새끼들->친구들, 한녀->한국여자, 짱깨->중국.
- Intensifiers such as 존나, 개, 씹, 좆 are context-dependent; do not direct-replace without high-confidence evidence.
- Preserve laughter/emotion/backchannel tokens such as ㅋㅋ, ㅎㅎ, ㅠㅠ, ㅜㅜ, ㅇㅇ, ㄴㄴ, ㄷㄷ unless the exact dataset example says otherwise.
- Jamo abbreviations and mixed Hangul+digit/Latin forms can be candidates, but they still require context.
- Do not translate or rewrite named entities, hashtags, mentions, usernames, numbers, or social-media tags.
""".strip()

FEW_SHOT_EXAMPLES = [
    {"tokens": ["ㄹㅇ", "귀엽다"], "target_index": 0, "target": "ㄹㅇ", "label": 1, "normalized": "진짜", "notes": "ㄹㅇ is a high-signal normalization candidate."},
    {"tokens": ["ㅅㅂ", "이거", "뭐냐"], "target_index": 0, "target": "ㅅㅂ", "label": 1, "normalized": "이런", "notes": "Profanity abbreviation is often euphemized in the project data."},
    {"tokens": ["존나", "좋네", "ㅋㅋ"], "target_index": 0, "target": "존나", "label": 1, "normalized": "매우", "notes": "존나 can normalize as an intensifier, but context matters."},
    {"tokens": ["존나", "좋네", "ㅋㅋ"], "target_index": 2, "target": "ㅋㅋ", "label": 0, "normalized": "ㅋㅋ", "notes": "Laughter/emotion token is preserved by default."},
    {"tokens": ["#dcinside", "2024", "ㅇㅇ"], "target_index": 0, "target": "#dcinside", "label": 0, "normalized": "#dcinside", "notes": "Hashtags and numbers are protected."},
]


def has_korean_script(token: str) -> bool:
    return bool(KOREAN_RE.search(token or ""))


def is_laughter_or_emotion_token(token: str) -> bool:
    return bool(LAUGHTER_EMOTION_RE.fullmatch(token or ""))


def has_repeated_korean_characters(token: str) -> bool:
    return bool(REPEATED_KO_RE.search(token or ""))


def has_digit_or_latin_mixed_korean(token: str) -> bool:
    return bool(DIGIT_LATIN_MIXED_KO_RE.search(token or ""))


def has_jamo_abbreviation(token: str) -> bool:
    token = token or ""
    return bool(JAMO_RE.search(token)) and len(token) <= 8


def is_ko_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    if not token or is_protected_token(token):
        return False
    if token in COMMON_KO_NORMALIZATION_CUES:
        return True
    if token in PRESERVE_OR_CONTEXTUAL and is_laughter_or_emotion_token(token):
        return False
    if re.search(r"(시발|씨발|ㅅㅂ|ㅆㅂ|존나|좆나|ㅈㄴ|병신|새끼|놈|년|틀딱|짱깨|조센|한녀|한남|개씹|개쩔|노잼|존잘|커엽|킹치|ㅇㅈㄹ)", token):
        return True
    if has_jamo_abbreviation(token) and not is_laughter_or_emotion_token(token):
        return True
    if has_digit_or_latin_mixed_korean(token):
        return True
    if has_repeated_korean_characters(token):
        return True
    return False


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_ko_likely_candidate(tok, tokens=sentence_tokens, index=i)]


def build_ko_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=KOREAN_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )


def build_ko_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=KOREAN_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )

build_ko_target_prompt = build_ko_target_detection_prompt
build_ko_normalization_prompt = build_ko_target_normalization_prompt

__all__ = [
    "LANG", "KOREAN_RULE_BLOCK", "COMMON_KO_NORMALIZATION_CUES", "PRESERVE_OR_CONTEXTUAL", "FEW_SHOT_EXAMPLES",
    "has_korean_script", "is_laughter_or_emotion_token", "has_repeated_korean_characters", "has_digit_or_latin_mixed_korean",
    "has_jamo_abbreviation", "is_ko_likely_candidate", "candidate_indices", "build_ko_target_detection_prompt",
    "build_ko_target_normalization_prompt", "build_ko_target_prompt", "build_ko_normalization_prompt",
]
