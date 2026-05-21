"""Korean-specific lexical-normalization rules for MultiLexNorm-style data.

Version: ko_v2_paper_informed

This module is deliberately conservative. Korean UGT contains intentional text
noise, memes, fillers, profanity-avoidance forms, compatibility-jamo reactions,
proper nouns, coinages, and internet slang. A token that looks noisy is not
necessarily a normalization target.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

try:
    from prompts.common_prompt import (
        build_common_detection_prompt,
        build_common_normalization_prompt,
        find_protected_indices,
        is_protected_token,
    )
except Exception:  # pragma: no cover - allows standalone inspection
    build_common_detection_prompt = None
    build_common_normalization_prompt = None
    find_protected_indices = None
    is_protected_token = None

LANG = "ko"

HANGUL_RE = re.compile(r"[가-힣]")
JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")
KOREAN_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
URL_MENTION_HASH_RE = re.compile(r"^(?:https?://|www\.|@\S+|#\S+)")
REPEATED_KO_RE = re.compile(r"([가-힣ㄱ-ㅎㅏ-ㅣ])\1{2,}")
DIGIT_LATIN_MIX_RE = re.compile(r"(?=.*[가-힣ㄱ-ㅎㅏ-ㅣ])(?=.*[A-Za-z0-9])")
QWERTY_KOREAN_LIKE_RE = re.compile(r"^(?:Tlqkf|tlqkf|qt|qk|wlsWk|Rk|rk|Rkwl|qkqh|whs나|[A-Za-z0-9^]+ㅣ발)$")

# Korean compatibility-jamo laughter/emotion/backchannel forms. These are often
# emoji-like tokens and should be preserved unless an exact dataset example says otherwise.
LAUGHTER_EMOTION_RE = re.compile(
    r"^(?:[ㅋㅎ]{1,}|[ㅠㅜ]+|(?:ㅇㅇ|ㄴㄴ|ㄷㄷ|ㄱㄱ|ㅊㅋ|ㄱㅅ|ㅇㅋ|ㅇㄷ|ㅁㅊ|ㅉㅉ|ㅍㅌㅊ))$"
)

# Empirical project-data cues. Direct replacement must still come from the MFR dictionary.
COMMON_KO_NORMALIZATION_CUES: dict[str, str] = {
    "ㄹㅇ": "진짜",
    "ㅇㄱㄹㅇ": "진짜",
    "ㅅㅂ": "이런",
    "ㅆㅂ": "이런",
    "ㅆ1발": "이런",
    "시발": "이런",
    "씨발": "이런",
    "Tlqkf": "이런",
    "ㅈㄴ": "매우",
    "존나": "매우",
    "좆나": "매우",
    "존1나": "매우",
    "걍": "그냥",
    "한녀": "한국여자",
    "한남": "한국남자",
    "짱깨": "중국",
    "조센징": "한국인",
    "조센": "한국",
    "병신": "바보",
    "병신들": "바보들",
    "새끼들": "친구들",
    "새끼": "친구",
    "놈": "사람",
    "중딩": "중학생",
    "초딩": "초등학생",
    "고딩": "고등학생",
    "커엽": "귀엽",
    "노잼": "재미없음",
    "비추": "비추천",
    "존잘": "매우 잘생겼다",
    "개웃기네": "매우 웃기네",
    "킹치만": "하지만",
}

# Preserve by default. These are not "never normalize"; rather, exact MFR or
# strong target-level evidence is required before changing them.
PRESERVE_BY_DEFAULT = {
    "ㅋㅋ", "ㅋㅋㅋ", "ㅋㅋㅋㅋ", "ㅋ", "ㅎㅎ", "ㅎㅎㅎ", "ㅎ", "ㅠㅠ", "ㅠㅠㅠ", "ㅠ", "ㅜㅜ", "ㅜㅜㅜ", "ㅜ",
    "ㅇㅇ", "ㄴㄴ", "ㄷㄷ", "ㄱㄱ", "ㅊㅋ", "ㄱㅅ", "ㅇㅋ", "ㅇㄷ", "ㅁㅊ", "ㅉㅉ",
    "흠", "음", "아", "어", "헐", "...", "…", "!!", "!?", "?", "~", "ㅡㅡ", "^^", "^_^",
}

# Tokens frequently needing context. These should usually go to prompt/model fallback.
CONTEXT_SENSITIVE_TOKENS = {
    "존나", "걍", "개", "새끼", "씹", "좆", "틀딱", "갓본", "조센", "한녀", "한남", "짱깨", "놈", "년",
    "노", "누", "긔", "임", "함", "듯", "듯요", "커엽", "노잼", "비추", "존잘", "킹치만",
    "시발", "씨발", "ㅅㅂ", "ㅆㅂ", "병신", "새끼들", "이새끼", "조센징",
}

# User-generated Korean endings / particles from K-MT-style observation.
# They are candidates only when attached to a longer lexical stem; they are not
# automatically normalized.
ONLINE_STYLE_ENDING_RE = re.compile(
    r"(?:용|여|당|넹|구용|구여|군용|군여|나용|나여|네용|네여|세용|세여|합니당|입니당|답니당|랍니당)$"
)

KOREAN_RULE_BLOCK = """
Korean-specific guidance:
- Korean social-media text often contains intentional text gaming, not just accidental typos. Do not normalize a token only because it looks noisy, offensive, meme-like, or community-specific.
- Intentionally noisy Korean forms can function as tricks, memes, fillers, or codes. Strategies include morphological changes, morpho-phonological spelling, optical substitution such as 야민정음/leetspeak, and semantic/code-mixed tricks.
- Compatibility-jamo reactions such as ㅋㅋ, ㅎㅎ, ㅠㅠ, ㅜㅜ, ㄷㄷ, ㅇㅇ, ㄴㄴ usually function like emojis/backchannels. Preserve them by default unless the exact dataset evidence says otherwise.
- Korean UGT often includes proper nouns, newly coined terms, internet slang, foreign characters, spacing errors, emojis, and Korean grapheme characters. Avoid over-segmenting, translating, or rewriting proper nouns, names, community terms, and coinages.
- Offensive or identity-related expressions are culturally and contextually sensitive. The project data sometimes euphemizes explicit profanity, e.g. ㅅㅂ/ㅆㅂ/시발/씨발->이런, 병신->바보, 새끼들->친구들. However, do not blindly euphemize every offensive-looking word; use sentence context and dataset-style evidence.
- High-signal project mappings include: ㄹㅇ->진짜, ㅅㅂ/ㅆㅂ/시발/씨발->이런, ㅈㄴ/존나->매우, 걍->그냥, 병신->바보, 새끼들->친구들, 한녀->한국여자, 짱깨->중국.
- Intensifiers such as 존나, 개, 씹, 좆 are context-dependent. They may normalize to 매우/진짜/엄청 only when used as intensifiers under this dataset style; otherwise preserve.
- Meme/community jargon such as 커엽, 노잼, 킹치만, 갓본, 조센 may be candidates, but many should be preserved if the annotation style does not change them.
- Mixed Hangul + digit/Latin/QWERTY forms such as 존1나, ㅆ1발, Tlqkf are candidates for target-level checking, but do not rewrite the entire sentence.
- Preserve URLs, mentions, hashtags, named entities, product/event/group names, pure numbers, and alphanumeric entities by default.
""".strip()

FEW_SHOT_EXAMPLES: list[dict[str, object]] = [
    {
        "raw_sentence": "ㄹㅇ 귀엽다 ㅋㅋ",
        "tokens": ["ㄹㅇ", "귀엽다", "ㅋㅋ"],
        "target_index": 0,
        "target": "ㄹㅇ",
        "label": 1,
        "normalized": "진짜",
        "notes": "ㄹㅇ is a high-signal dataset mapping to 진짜.",
    },
    {
        "raw_sentence": "ㄹㅇ 귀엽다 ㅋㅋ",
        "tokens": ["ㄹㅇ", "귀엽다", "ㅋㅋ"],
        "target_index": 2,
        "target": "ㅋㅋ",
        "label": 0,
        "normalized": "ㅋㅋ",
        "notes": "ㅋㅋ is an emoji-like laughter token and should be preserved by default.",
    },
    {
        "raw_sentence": "ㅅㅂ 이거 뭐냐",
        "tokens": ["ㅅㅂ", "이거", "뭐냐"],
        "target_index": 0,
        "target": "ㅅㅂ",
        "label": 1,
        "normalized": "이런",
        "notes": "The project data often euphemizes explicit profanity abbreviations to 이런.",
    },
    {
        "raw_sentence": "존나 귀엽다",
        "tokens": ["존나", "귀엽다"],
        "target_index": 0,
        "target": "존나",
        "label": 1,
        "normalized": "매우",
        "notes": "존나 can be an intensifier and may normalize to 매우 when the local context supports it.",
    },
    {
        "raw_sentence": "존나 ㅋㅋㅋㅋ",
        "tokens": ["존나", "ㅋㅋㅋㅋ"],
        "target_index": 0,
        "target": "존나",
        "label": 0,
        "normalized": "존나",
        "notes": "When the context is too weak or exclamatory, preserve rather than force euphemization.",
    },
    {
        "raw_sentence": "김유미의 팬이다",
        "tokens": ["김유미의", "팬이다"],
        "target_index": 0,
        "target": "김유미의",
        "label": 0,
        "normalized": "김유미의",
        "notes": "Proper names and attached particles should not be over-segmented or rewritten by the prompt.",
    },
    {
        "raw_sentence": "ㅆ1발 진짜",
        "tokens": ["ㅆ1발", "진짜"],
        "target_index": 0,
        "target": "ㅆ1발",
        "label": 1,
        "normalized": "이런",
        "notes": "Digit/optical profanity avoidance is a candidate, but only the target token should be normalized.",
    },
    {
        "raw_sentence": "#dcinside ㅠㅠ",
        "tokens": ["#dcinside", "ㅠㅠ"],
        "target_index": 0,
        "target": "#dcinside",
        "label": 0,
        "normalized": "#dcinside",
        "notes": "Hashtags and social-media entities are protected.",
    },
]


def has_korean_script(token: str) -> bool:
    return bool(KOREAN_RE.search(token or ""))


def is_url_mention_hashtag(token: str) -> bool:
    return bool(URL_MENTION_HASH_RE.search(token or ""))


def is_laughter_or_emotion_token(token: str) -> bool:
    return bool(LAUGHTER_EMOTION_RE.fullmatch(token or ""))


def has_repeated_korean_characters(token: str) -> bool:
    return bool(REPEATED_KO_RE.search(token or ""))


def has_digit_or_latin_mixed_korean(token: str) -> bool:
    return bool(DIGIT_LATIN_MIX_RE.search(token or ""))


def has_jamo_abbreviation(token: str) -> bool:
    token = token or ""
    return bool(JAMO_RE.search(token)) and len(token) <= 10


def is_contextual_ending_or_particle(token: str) -> bool:
    token = token or ""
    return bool(ONLINE_STYLE_ENDING_RE.search(token)) and len(token) >= 3


def is_common_protected(token: str) -> bool:
    if is_protected_token is None:
        return is_url_mention_hashtag(token)
    return bool(is_protected_token(token, extra_protected=PRESERVE_BY_DEFAULT))


def is_ko_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    """Return True when token should be sent to prompt/model resolution.

    This identifies candidates; it does not mean direct replacement is safe.
    """
    if not token or is_common_protected(token):
        return False
    if token in PRESERVE_BY_DEFAULT:
        return False
    if token in COMMON_KO_NORMALIZATION_CUES:
        return True
    if token in CONTEXT_SENSITIVE_TOKENS:
        return True
    # Offensive, identity, intensifier, and community-jargon stems.
    if re.search(r"(시발|씨발|ㅅㅂ|ㅆㅂ|존나|좆나|ㅈㄴ|병신|새끼|놈|년|틀딱|짱깨|조센|한녀|한남|개씹|개쩔|개꼴|노잼|존잘|커엽|킹치|ㅇㅈㄹ|이지랄|쌉가능|개소리)", token):
        return True
    # Compatibility jamo abbreviations, except pure laughter/backchannel forms.
    if has_jamo_abbreviation(token) and not is_laughter_or_emotion_token(token):
        return True
    # Optical/leetspeak/QWERTY/code-mixed avoidance.
    if has_digit_or_latin_mixed_korean(token) or QWERTY_KOREAN_LIKE_RE.search(token):
        return True
    # Repetition can be expressive; treat as candidate only, never direct rule.
    if has_repeated_korean_characters(token) and not is_laughter_or_emotion_token(token):
        return True
    # Stylistic endings used in UGT are candidates only when attached to a lexical stem.
    if is_contextual_ending_or_particle(token):
        return True
    return False


def candidate_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_ko_likely_candidate(tok, tokens=tokens, index=i)]


def _fallback_detection_prompt(sentence_tokens: Sequence[str], target_index: int | None = None) -> str:
    return f"""
You are a Korean lexical-normalization detector for MultiLexNorm-style data.
Return 1 only when the target token's gold normalized form differs from raw. Preserve noisy-looking but expressive or entity-like Korean text unless dataset evidence supports a change.

{KOREAN_RULE_BLOCK}

Tokens: {json.dumps(list(sentence_tokens), ensure_ascii=False)}
Target index: {target_index}
Return only JSON: {{"label":0 or 1}}
""".strip()


def build_ko_target_detection_prompt(
    sentence_tokens: Sequence[str],
    target_index: int,
    *,
    raw_sentence: str | None = None,
    protected_indices: Sequence[int] | None = None,
    include_fewshot: bool = True,
) -> str:
    if build_common_detection_prompt is None:
        return _fallback_detection_prompt(sentence_tokens, target_index)
    if protected_indices is None and find_protected_indices is not None:
        protected_indices = find_protected_indices(sentence_tokens, extra_protected=PRESERVE_BY_DEFAULT)
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        language_rule_block=KOREAN_RULE_BLOCK,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        fewshot_examples=FEW_SHOT_EXAMPLES if include_fewshot else None,
    )


def build_ko_target_normalization_prompt(
    sentence_tokens: Sequence[str],
    target_index: int,
    *,
    raw_sentence: str | None = None,
    protected_indices: Sequence[int] | None = None,
    include_fewshot: bool = True,
) -> str:
    if build_common_normalization_prompt is None:
        return _fallback_detection_prompt(sentence_tokens, target_index)
    if protected_indices is None and find_protected_indices is not None:
        protected_indices = find_protected_indices(sentence_tokens, extra_protected=PRESERVE_BY_DEFAULT)
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        language_rule_block=KOREAN_RULE_BLOCK,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=protected_indices,
        fewshot_examples=FEW_SHOT_EXAMPLES if include_fewshot else None,
    )


# Backward-compatible aliases used by earlier packages.
def build_ko_target_prompt(sentence_tokens: Sequence[str], target_index: int) -> str:
    return build_ko_target_detection_prompt(sentence_tokens, target_index)


def build_ko_detection_prompt(sentence_tokens: Sequence[str], *, include_fewshot: bool = True) -> str:
    # Legacy all-token prompt. New pipeline should prefer target prompts.
    examples = FEW_SHOT_EXAMPLES if include_fewshot else None
    labels_hint = "Return only JSON: {\"labels\":[0 or 1 for each token]}"
    return f"""
You are a Korean lexical-normalization detector for MultiLexNorm-style data.
For each token, decide whether the gold normalized form differs from raw.
Prefer the target-token functions in this module for production use.

{KOREAN_RULE_BLOCK}

Few-shot examples:
{examples}

Input tokens:
{json.dumps(list(sentence_tokens), ensure_ascii=False)}

{labels_hint}
""".strip()


__all__ = [
    "LANG",
    "KOREAN_RULE_BLOCK",
    "COMMON_KO_NORMALIZATION_CUES",
    "PRESERVE_BY_DEFAULT",
    "CONTEXT_SENSITIVE_TOKENS",
    "FEW_SHOT_EXAMPLES",
    "has_korean_script",
    "is_laughter_or_emotion_token",
    "has_repeated_korean_characters",
    "has_digit_or_latin_mixed_korean",
    "has_jamo_abbreviation",
    "is_contextual_ending_or_particle",
    "is_ko_likely_candidate",
    "candidate_indices",
    "build_ko_target_detection_prompt",
    "build_ko_target_normalization_prompt",
    "build_ko_target_prompt",
    "build_ko_detection_prompt",
]
