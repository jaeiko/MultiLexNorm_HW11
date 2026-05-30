"""Turkish-specific prompt rules and candidate selection."""
from __future__ import annotations

from typing import Sequence

try:
    from prompts.common_prompt import build_common_detection_prompt, build_common_normalization_prompt, is_protected_token
    from language_rules.turkish_common import TURKISH_COMMON_RULE_BLOCK, is_turkish_likely_candidate
except Exception:  # pragma: no cover
    from ..prompts.common_prompt import build_common_detection_prompt, build_common_normalization_prompt, is_protected_token
    from .turkish_common import TURKISH_COMMON_RULE_BLOCK, is_turkish_likely_candidate

LANG = "tr"

LANGUAGE_RULE_BLOCK = TURKISH_COMMON_RULE_BLOCK + """

Turkish-only policy:
- Normalize Turkish surface forms only; do not translate Turkish words into English/German or replace them with semantic paraphrases.
- Be conservative with short function tokens such as mi, de, da, ki, ne, bu, o. They can be valid as-is or require Turkish-specific characters/clitic handling depending on context.
- Do not blindly lowercase or uppercase; casing changes should reflect dataset evidence, especially sentence-initial words and proper nouns.
- If a token is a name or could be a name, preserve unless the correction is a minimal Turkish apostrophe/capitalization restoration supported by context.
""".strip()

FEW_SHOT_EXAMPLES = [
    {"raw_sentence": "ben cok mutluyum", "tokens": ["ben", "cok", "mutluyum"], "target_index": 1, "target": "cok", "label": 1, "normalized": "çok", "notes": "Missing Turkish diacritic; surface normalization only."},
    {"raw_sentence": "suan evdeyim", "tokens": ["suan", "evdeyim"], "target_index": 0, "target": "suan", "label": 1, "normalized": "şu an", "notes": "Common Turkish spacing/deasciification normalization."},
    {"raw_sentence": "@user cok iyi yaaa #konu", "tokens": ["@user", "cok", "iyi", "yaaa", "#konu"], "target_index": 0, "target": "@user", "label": 0, "normalized": "@user", "notes": "Mention is protected."},
    {"raw_sentence": "@user cok iyi yaaa #konu", "tokens": ["@user", "cok", "iyi", "yaaa", "#konu"], "target_index": 4, "target": "#konu", "label": 0, "normalized": "#konu", "notes": "Hashtag is protected."},
    {"raw_sentence": "almanyada yaşıyorum", "tokens": ["almanyada", "yaşıyorum"], "target_index": 0, "target": "almanyada", "label": 1, "normalized": "Almanya'da", "notes": "Proper noun + Turkish locative suffix apostrophe restoration."},
    {"raw_sentence": "mi acaba", "tokens": ["mi", "acaba"], "target_index": 0, "target": "mi", "label": 0, "normalized": "mi", "notes": "Short Turkish question particle is context-sensitive; preserve if uncertain."},
]

EXTRA_TR_CANDIDATES = {
    "cok", "degil", "guzel", "nasil", "icin", "hic", "boyle", "artik", "sey", "bi", "bide", "suan", "amk", "aq", "allahın", "allahim", "Allahım",
}

CONTEXT_SENSITIVE_TOKENS = {
    "mi", "mı", "mu", "mü", "de", "da", "ki", "ne", "bu", "o", "a", "e", "i", "ı", "İ", "t", "k", "ben", "sen", "allah", "Allah",
}


def is_tr_likely_candidate(tok: str) -> bool:
    s = str(tok).strip()
    if not s or is_protected_token(s):
        return False
    if s in EXTRA_TR_CANDIDATES or s.lower() in EXTRA_TR_CANDIDATES:
        return True
    if s in CONTEXT_SENSITIVE_TOKENS:
        return True
    return is_turkish_likely_candidate(s)


def candidate_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_tr_likely_candidate(str(tok))]


def build_tr_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, raw_sentence: str | None = None) -> str:
    return build_common_detection_prompt(lang=LANG, sentence_tokens=sentence_tokens, target_index=target_index, raw_sentence=raw_sentence, language_rule_block=LANGUAGE_RULE_BLOCK, fewshot_examples=FEW_SHOT_EXAMPLES)


def build_tr_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, raw_sentence: str | None = None) -> str:
    return build_common_normalization_prompt(lang=LANG, sentence_tokens=sentence_tokens, target_index=target_index, raw_sentence=raw_sentence, language_rule_block=LANGUAGE_RULE_BLOCK, fewshot_examples=FEW_SHOT_EXAMPLES)

# Backward-compatible aliases
build_target_detection_prompt = build_tr_target_detection_prompt
build_target_normalization_prompt = build_tr_target_normalization_prompt
