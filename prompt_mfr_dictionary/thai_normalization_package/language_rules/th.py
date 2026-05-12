"""Thai-specific rules built on top of the common target-only prompt."""
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

LANG = "th"

THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")
REPEATED_THAI_RE = re.compile(r"([\u0E00-\u0E7F])\1{2,}")
REPEATED_5_RE = re.compile(r"^(?:5|๕){2,}$")
MAIYAMOK_RE = re.compile(r"ๆ")

COMMON_TH_NORMALIZATION_CUES: dict[str, str] = {
    "เค้า": "เขา", "นึง": "หนึ่ง", "มั้ย": "ไหม", "จ้า": "จ้ะ", "ละ": "แล้ว",
    "งี้": "อย่างนี้", "ยังไง": "อย่างไร", "อ่ะ": "อะ", "ก้": "ก็", "ก้อ": "ก็", "คับ": "ครับ",
    "เปน": "เป็น", "จิง": "จริง", "โครต": "โคตร", "หรอ": "หรือ", "ไร": "อะไร",
    "จริงๆ": "จริง ๆ", "มากๆ": "มาก ๆ", "ดีๆ": "ดี ๆ", "หลายๆ": "หลาย ๆ",
    "คอน": "คอนเสิร์ต", "เว็บ": "เว็บไซต์", "โพส": "โพสต์", "แอพ": "แอป", "ทวิต": "ทวิตเตอร์",
    "มือ2": "มือสอง", "มากกก": "มาก", "โว้ยย": "โว้ย", "ฮือออ": "ฮือ", "แงงง": "แง",
    "แฟมิลี่": "แฟมิลี",
}

PRESERVE_BY_DEFAULT = {
    "555", "5555", "55555", "๕๕๕", "๕๕๕๕",
    "ค่ะ", "คะ", "ครับ", "นะ", "น้า", "จ้า", "จ้ะ",
}

THAI_RULE_BLOCK = """
Thai-specific guidance:
- Thai is a tonal, largely unsegmented language; use the raw sentence as context and keep token alignment.
- Thai misspellings may encode emotion, emphasis, politeness, friendliness, or identity. Do not blindly normalize noisy-looking text.
- Stable project-data candidates include เค้า->เขา, นึง->หนึ่ง, มั้ย->ไหม, จริงๆ->จริง ๆ, มากๆ->มาก ๆ, อ่ะ->อะ, ก้->ก็, เปน->เป็น, คับ->ครับ, แฟมิลี่->แฟมิลี.
- Repeated Thai characters may normalize when the dataset clearly reduces length, e.g. มากกก->มาก, but unseen repetition may carry emphasis and should be preserved if uncertain.
- Laughter tokens such as 555, 5555, and ๕๕๕ should be preserved by default.
- Politeness particles such as คะ, ค่ะ, ครับ, คับ, จ้า, นะ, น้า are context-dependent.
- Do not translate Thai loanwords, names, idols, group names, fandom tags, event names, hashtags, mentions, numbers, dates, or times.
- Hashtags such as #BNK48 and alphanumeric names such as BNK48 must remain unchanged.
""".strip()

FEW_SHOT_EXAMPLES = [
    {"raw_sentence": "เค้า ชอบ จริงๆ มากกก 555", "tokens": ["เค้า", "ชอบ", "จริงๆ", "มากกก", "555"], "target_index": 0, "target": "เค้า", "label": 1, "normalized": "เขา", "notes": "เค้า is often normalized to เขา."},
    {"raw_sentence": "เค้า ชอบ จริงๆ มากกก 555", "tokens": ["เค้า", "ชอบ", "จริงๆ", "มากกก", "555"], "target_index": 4, "target": "555", "label": 0, "normalized": "555", "notes": "555 is Thai laughter and is protected/preserved."},
    {"raw_sentence": "ศิริปันนา แฟมิลี่ แฟร์ #BNK48", "tokens": ["ศิริปันนา", "แฟมิลี่", "แฟร์", "#BNK48"], "target_index": 1, "target": "แฟมิลี่", "label": 1, "normalized": "แฟมิลี", "notes": "Surface spelling variant; do not translate surrounding loanwords."},
    {"raw_sentence": "ศิริปันนา แฟมิลี่ แฟร์ #BNK48", "tokens": ["ศิริปันนา", "แฟมิลี่", "แฟร์", "#BNK48"], "target_index": 3, "target": "#BNK48", "label": 0, "normalized": "#BNK48", "notes": "Hashtag is a protected social-media entity."},
    {"raw_sentence": "โชว์ จาก 6 สาว BNK48", "tokens": ["โชว์", "จาก", "6", "สาว", "BNK48"], "target_index": 0, "target": "โชว์", "label": 0, "normalized": "โชว์", "notes": "Loanword should not be semantically translated to แสดง unless exact dataset evidence exists."},
]


def has_thai_script(token: str) -> bool:
    return bool(THAI_CHAR_RE.search(token or ""))


def has_repeated_thai_characters(token: str) -> bool:
    if REPEATED_5_RE.fullmatch(token or ""):
        return False
    return bool(REPEATED_THAI_RE.search(token or ""))


def has_repetition_marker_spacing_candidate(token: str) -> bool:
    return bool(MAIYAMOK_RE.search(token or ""))


def is_th_laughter_token(token: str) -> bool:
    return bool(REPEATED_5_RE.fullmatch(token or ""))


def is_th_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    if not token or is_protected_token(token):
        return False
    if is_th_laughter_token(token):
        return False
    if token == "ชั้น" and tokens is not None and index is not None and index + 1 < len(tokens) and is_protected_token(tokens[index + 1]):
        return False
    if token in COMMON_TH_NORMALIZATION_CUES:
        return True
    if token in PRESERVE_BY_DEFAULT:
        return False
    if has_repetition_marker_spacing_candidate(token):
        return True
    if has_repeated_thai_characters(token):
        return True
    if has_thai_script(token) and re.search(r"(?:มั้ย|หรอ|ก้|ก้อ|เปน|จิง|โครต|แท๊ก|คอน|เว็บ|โพส|แอพ|ทวิต|ไร)$", token):
        return True
    return False


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_th_likely_candidate(tok, tokens=sentence_tokens, index=i)]


def build_th_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=THAI_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )


def build_th_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=THAI_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )

build_th_target_prompt = build_th_target_detection_prompt
build_th_normalization_prompt = build_th_target_normalization_prompt

__all__ = [
    "LANG", "THAI_RULE_BLOCK", "COMMON_TH_NORMALIZATION_CUES", "PRESERVE_BY_DEFAULT", "FEW_SHOT_EXAMPLES",
    "has_thai_script", "has_repeated_thai_characters", "has_repetition_marker_spacing_candidate", "is_th_laughter_token",
    "is_th_likely_candidate", "candidate_indices", "build_th_target_detection_prompt", "build_th_target_normalization_prompt",
    "build_th_target_prompt", "build_th_normalization_prompt",
]
