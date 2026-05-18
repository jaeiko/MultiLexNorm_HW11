"""Japanese-specific rules built on top of the common target-only prompt."""
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

LANG = "ja"

JAPANESE_RE = re.compile(r"[ぁ-んァ-ン一-龯々〆〤]")
REPEATED_MARK_RE = re.compile(r"(?:ー{2,}|〜{2,}|~{2,}|w{2,}|ｗ{2,})")
SMALL_VOWEL_END_RE = re.compile(r"[ぁぃぅぇぉァィゥェォ]$")

COMMON_JA_NORMALIZATION_CUES: dict[str, str] = {
    "てる": "て いる", "でる": "で いる", "じゃ": "で は", "じゃなくて": "で は なく て", "ちゃう": "て しまう", "ちゃっ": "て しまっ",
    "なきゃ": "なければ", "んで": "ので", "っていう": "という", "ってゆう": "という",
    "やっぱり": "やはり", "やっぱ": "やはり", "ぐらい": "くらい", "ばっか": "ばかり", "どっか": "どこか", "あんまり": "あまり",
    "スマホ": "スマートフォン", "バイト": "アルバイト", "リプ": "リプライ", "RT": "リツイート", "DM": "個別 メッセージ", "TL": "タイムライン",
    "ついった": "ツイッター", "twitter": "ツイッター",
}

AMBIGUOUS_JA_TOKENS = {"て", "って", "ん", "、", "。", "…", "ね", "よ", "た", "か", "で", "に", "の", "と", "いる", "のに", "ください", "です", "ます"}

JAPANESE_RULE_BLOCK = """
Japanese-specific guidance:
- Japanese is largely unsegmented and normalization may involve boundary-aware span conversion; use the indexed tokens for alignment and the raw sentence only as context.
- Use conservative MFR only for stable lexical variants. Many function tokens and punctuation tokens are highly context-dependent.
- Common candidates include てる->て いる, じゃ->で は, ちゃう/ちゃっ->て しまう/て しまっ, んで->ので, っていう->という, やっぱり/やっぱ->やはり, スマホ->スマートフォン, RT->リツイート, DM->個別 メッセージ.
- Do not insert sentence-final punctuation just because a token can appear sentence-final; tokens like です, ます, ね, よ, 。, … are context-dependent.
- Preserve hashtags, mentions, URLs, numbers, dates, times, IDs, usernames, named entities, product names, group names, and alphanumeric entities.
- Do not translate loanwords or names into semantic equivalents unless the exact target token is a known dataset normalization pair.
""".strip()

FEW_SHOT_EXAMPLES = [
    {"tokens": ["ついった", "み", "てる"], "target_index": 0, "target": "ついった", "label": 1, "normalized": "ツイッター", "notes": "Surface spelling variant of Twitter."},
    {"tokens": ["ついった", "み", "てる"], "target_index": 2, "target": "てる", "label": 1, "normalized": "て いる", "notes": "Colloquial contraction can expand."},
    {"tokens": ["スマホ", "便利", "です"], "target_index": 0, "target": "スマホ", "label": 1, "normalized": "スマートフォン", "notes": "Stable lexical abbreviation."},
    {"tokens": ["BNK48", "の", "ライブ", "#MobileBNK48"], "target_index": 0, "target": "BNK48", "label": 0, "normalized": "BNK48", "notes": "Alphanumeric entity is protected."},
    {"tokens": ["BNK48", "の", "ライブ", "#MobileBNK48"], "target_index": 3, "target": "#MobileBNK48", "label": 0, "normalized": "#MobileBNK48", "notes": "Hashtag is protected."},
]


def has_japanese_script(token: str) -> bool:
    return bool(JAPANESE_RE.search(token or ""))


def has_repetition_or_prolongation(token: str) -> bool:
    return bool(REPEATED_MARK_RE.search(token or ""))


def is_ja_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    if not token or is_protected_token(token):
        return False
    if token in AMBIGUOUS_JA_TOKENS:
        return False
    if token in COMMON_JA_NORMALIZATION_CUES:
        return True
    if has_repetition_or_prolongation(token):
        return True
    if SMALL_VOWEL_END_RE.search(token or ""):
        return True
    if has_japanese_script(token) and re.search(r"(?:てる|でる|じゃ|ちゃう|ちゃっ|なきゃ|んで|っていう|ってゆう|やっぱ|ぐらい|ばっか|どっか)$", token):
        return True
    return False


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_ja_likely_candidate(tok, tokens=sentence_tokens, index=i)]


def build_ja_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_common_detection_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=JAPANESE_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )


def build_ja_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, *, raw_sentence: str | None = None) -> str:
    return build_common_normalization_prompt(
        lang=LANG,
        sentence_tokens=sentence_tokens,
        target_index=target_index,
        raw_sentence=raw_sentence,
        protected_indices=find_protected_indices(sentence_tokens),
        language_rule_block=JAPANESE_RULE_BLOCK,
        fewshot_examples=FEW_SHOT_EXAMPLES,
    )

build_ja_target_prompt = build_ja_target_detection_prompt
build_ja_normalization_prompt = build_ja_target_normalization_prompt

__all__ = [
    "LANG", "JAPANESE_RULE_BLOCK", "COMMON_JA_NORMALIZATION_CUES", "AMBIGUOUS_JA_TOKENS", "FEW_SHOT_EXAMPLES",
    "has_japanese_script", "has_repetition_or_prolongation", "is_ja_likely_candidate", "candidate_indices",
    "build_ja_target_detection_prompt", "build_ja_target_normalization_prompt", "build_ja_target_prompt", "build_ja_normalization_prompt",
]
