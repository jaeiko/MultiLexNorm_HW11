"""Japanese language rules for MultiLexNorm-style lexical normalization, v2.

The v2 module is designed around three observations from Japanese LN work:
1. Japanese is normally unsegmented, so normalization is often a boundary-aware
   span extraction + conversion problem rather than a simple token rewrite.
2. Japanese SNS text contains many normalization categories: missing symbols,
   casual/formal endings, script variants, abbreviations, sound/pronunciation
   variants, emoticons, and unknown vocabulary.
3. Some expressive forms encode sentiment or speaker stance, so aggressive
   pronunciation/emotion normalization can be harmful.

This module therefore selects candidate tokens/spans and builds target-only
prompts. It should not be used to rewrite a full sentence directly.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Sequence

LANG = "ja"

# Safe, high-signal cues. They are candidates; direct replacement is handled by MFR.
COMMON_JA_NORMALIZATION_CUES: dict[str, str] = {
    "てる": "て いる",
    "でる": "で いる",
    "じゃ": "で は",
    "じゃなくて": "で は なく て",
    "ちゃっ": "て しまっ",
    "ちゃう": "て しまう / で は ない",
    "なきゃ": "なけれ ば / ない と",
    "んで": "ので",
    "っていう": "という",
    "やっぱ": "やはり",
    "やっぱり": "やはり",
    "ぐらい": "くらい",
    "ばっか": "ばかり",
    "ばっかり": "ばかり",
    "どっか": "どこか",
    "あんまり": "あまり",
    "ホント": "本当に",
    "ほんとに": "本当に",
    "スマホ": "スマートフォン",
    "バイト": "アルバイト",
    "ネット": "インターネット",
    "リプ": "返信",
    "ググっ": "検索し",
    "ワロタ": "笑っ た",
    "コロナ": "新型コロナウイルス感染症",
    "ついった": "ツイッター",
    "twitter": "Twitter",
    "RT": "リツイート",
    "DM": "個別 メッセージ",
    "TL": "タイムライン",
}

BOUNDARY_SENSITIVE_CUES: set[str] = {
    "てる", "でる", "じゃ", "じゃなくて", "ちゃう", "ちゃっ", "なきゃ", "んで", "っていう",
    "とこ", "てか", "とりま", "そういや", "もん", "っす"
}

# These tokens can be normalized in some corpora but are high FP-risk without context.
AMBIGUOUS_OR_CONTEXTUAL_TOKENS: set[str] = {
    "て", "って", "ん", "、", "。", "…", "〜", "～", "！", "?", "？", "ー", "ｗ", "w", "www",
    "ね", "よ", "た", "か", "な", "だ", "や", "わ", "で", "に", "の", "と", "し", "さ", "が", "を", "も",
    "です", "ます", "いる", "ください", "コロナ", "ネット", "RT", "DM", "TL",
}

PRESERVE_BY_DEFAULT_PATTERNS = [
    r"^https?://", r"^www\.", r"^@\w+", r"^#", r"^[0-9０-９]+$", r"^[_＿]+$",
]

JAPANESE_RULE_BLOCK = """
Japanese-specific guidance:
- This is lexical normalization, not translation, paraphrase, style transfer, or full-sentence rewriting.
- Japanese is an unsegmented language in ordinary writing. The target may correspond to a span or a short-unit-word boundary, so use the raw sentence only as context and edit only the marked target token/span.
- Prefer precision over recall. Preserve the target if several normalizations are plausible.
- Boundary-sensitive contractions include てる→て いる, でる→で いる, じゃ→で は, じゃなくて→で は なく て, ちゃう/ちゃっ→て しまう variants, なきゃ→なければ/ないと, んで→ので, っていう→という.
- Script/orthographic variants may be candidates: カワイイ→かわいい/可愛い, 大きぃ→大きい, おいしーい→おいしい, ついった→ツイッター.
- Abbreviations may be candidates when the dataset style supports them: スマホ→スマートフォン, バイト→アルバイト, リプ→返信, TL→タイムライン, DM→個別メッセージ.
- Do not normalize pronunciation variation or emotional spelling merely because it looks informal. Forms such as きゃわ, ふふっ, うわぁ, よーし, www, 笑, 草, emoticons, and emojis may preserve emotion or stance.
- Do not rewrite named entities, titles, group names, product names, hashtags, mentions, URLs, IDs, dates, times, or numbers unless the exact dataset-style pair is strongly supported.
- Punctuation insertion and missing-symbol normalization are context-dependent. Only normalize if the target is clearly sentence-final or title-like in the local context.
""".strip()

FEW_SHOT_EXAMPLES: list[dict[str, object]] = [
    {
        "tokens": ["ついった", "み", "てる"],
        "target_index": 0,
        "label": 1,
        "normalization": "ツイッター",
        "notes": "character/script variant of Twitter; target-only lexical normalization.",
    },
    {
        "tokens": ["ついった", "み", "てる"],
        "target_index": 2,
        "label": 1,
        "normalization": "て いる",
        "notes": "boundary-sensitive colloquial contraction.",
    },
    {
        "tokens": ["やっぱり", "スマホ", "便利"],
        "target_index": 0,
        "label": 1,
        "normalization": "やはり",
        "notes": "stable lexical variant.",
    },
    {
        "tokens": ["やっぱり", "スマホ", "便利"],
        "target_index": 1,
        "label": 1,
        "normalization": "スマートフォン",
        "notes": "abbreviation expansion supported by data/paper taxonomy.",
    },
    {
        "tokens": ["ふふっ", "て", "なっ", "た"],
        "target_index": 0,
        "label": 0,
        "normalization": None,
        "notes": "pronunciation/emotional expression may encode sentiment; preserve unless gold evidence is clear.",
    },
    {
        "tokens": ["www", "#anime", "2025"],
        "target_index": 0,
        "label": 0,
        "normalization": None,
        "notes": "laughter/emotion token is normally preserved in MultiLexNorm-style token alignment.",
    },
    {
        "tokens": ["RT", "@user", "新作", "きた"],
        "target_index": 1,
        "label": 0,
        "normalization": None,
        "notes": "mentions and protected social-media tokens are preserved.",
    },
    {
        "tokens": ["ね", "…"],
        "target_index": 0,
        "label": 0,
        "normalization": None,
        "notes": "sentence-final particles are ambiguous; do not insert punctuation unless local context strongly supports it.",
    },
]


def normalize_width(token: str) -> str:
    return unicodedata.normalize("NFKC", token)


def has_japanese_script(token: str) -> bool:
    return bool(re.search(r"[ぁ-んァ-ン一-龯々〆ヵヶ]", token))


def has_prolonged_or_repeated_marks(token: str) -> bool:
    if re.search(r"[ー〜～]{2,}|[!！?？]{2,}|ｗ{2,}|w{2,}", token):
        return True
    if re.search(r"(.)\1{3,}", token):
        return True
    return False


def looks_like_character_type_variant(token: str) -> bool:
    nfkc = normalize_width(token)
    if nfkc != token and has_japanese_script(token):
        return True
    if re.search(r"[ｦ-ﾟ]", token):
        return True
    if re.search(r"[ぁ-ん][ァ-ン]|[ァ-ン][ぁ-ん]", token):
        return True
    if re.search(r"[A-Za-zａ-ｚＡ-Ｚ]", token) and has_japanese_script(token):
        return True
    if token in {"twitter", "facebook", "amazon", "paypay", "zoom", "pixiv", "bgm"}:
        return True
    return False


def is_preserve_by_default(token: str) -> bool:
    for pat in PRESERVE_BY_DEFAULT_PATTERNS:
        if re.search(pat, token):
            return True
    if token in {"www", "WWW", "w", "ｗ", "笑", "草", "orz"}:
        return True
    return False


def is_sentence_final_position(tokens: Sequence[str], idx: int) -> bool:
    return idx == len(tokens) - 1 or all(t in {"", "。", "！", "!", "?", "？", "…", "〜", "～"} for t in tokens[idx+1:])


def is_ja_likely_candidate(token: str, *, tokens: Sequence[str] | None = None, index: int | None = None) -> bool:
    """Return True when token should be sent to target prompt/model fallback.

    This is a candidate detector, not a direct normalizer.
    """
    if not token or is_preserve_by_default(token):
        return False

    if token in COMMON_JA_NORMALIZATION_CUES:
        return True

    if token in AMBIGUOUS_OR_CONTEXTUAL_TOKENS:
        if tokens is not None and index is not None and is_sentence_final_position(tokens, index):
            return token in {"ね", "よ", "た", "か", "です", "ます", "…", "〜", "～", "！", "。"}
        return False

    if looks_like_character_type_variant(token):
        return True

    # Prolonged/repeated marks are candidates, but final decision should be conservative.
    if has_prolonged_or_repeated_marks(token):
        return True

    # Common colloquial contractions/endings.
    if re.search(r"(てる|でる|じゃん|じゃ|っす|ちゃう|ちゃっ|なきゃ|んで|っていう|って)$", token):
        return True

    # Shortened words and SNS abbreviations.
    if token in {"RT", "DM", "TL", "LV", "M1", "リプ", "スマホ", "バイト", "ネット", "コロナ"}:
        return True

    return False


def candidate_indices(sentence_tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(sentence_tokens) if is_ja_likely_candidate(tok, tokens=sentence_tokens, index=i)]


def candidate_spans(sentence_tokens: Sequence[str], max_span_len: int = 3) -> list[tuple[int, int]]:
    """Boundary-aware candidate spans.

    Returns half-open spans [start, end). Single-token candidates are included,
    and adjacent spans are added when a boundary-sensitive cue appears next to a
    function token. This is useful if the downstream system supports span-level
    prompting. Token-level pipelines can ignore this function.
    """
    spans: set[tuple[int, int]] = {(i, i + 1) for i in candidate_indices(sentence_tokens)}
    n = len(sentence_tokens)
    for i, tok in enumerate(sentence_tokens):
        if tok in BOUNDARY_SENSITIVE_CUES or is_ja_likely_candidate(tok, tokens=sentence_tokens, index=i):
            for start in range(max(0, i - 1), i + 1):
                for end in range(i + 1, min(n, i + max_span_len) + 1):
                    if end > start and end - start <= max_span_len:
                        spans.add((start, end))
    return sorted(spans)


def _format_few_shot(examples: Sequence[dict[str, object]]) -> str:
    blocks = []
    for ex in examples:
        blocks.append(
            "Input tokens: " + json.dumps(ex["tokens"], ensure_ascii=False) + "\n" +
            "Target index: " + str(ex["target_index"]) + "\n" +
            "Output JSON: " + json.dumps({"label": ex["label"], "normalization": ex["normalization"]}, ensure_ascii=False) + "\n" +
            "Reason: " + str(ex.get("notes", ""))
        )
    return "\n\n".join(blocks)


def build_ja_target_detection_prompt(
    tokens: Sequence[str],
    target_index: int,
    *,
    raw_sentence: str | None = None,
    include_fewshot: bool = True,
) -> str:
    marked = [f"<TARGET>{tok}</TARGET>" if i == target_index else tok for i, tok in enumerate(tokens)]
    fewshot = _format_few_shot(FEW_SHOT_EXAMPLES) if include_fewshot else ""
    return f"""
You are a Japanese lexical-normalization detector for MultiLexNorm-style data.

Task:
Decide whether ONLY the TARGET token/span should be normalized.
Return label 1 only if the gold normalized form would differ from the raw target.
Return label 0 if the target should be preserved exactly.

Raw sentence for context:
{raw_sentence if raw_sentence is not None else ""}

Indexed/tokenized sentence with target:
{json.dumps(marked, ensure_ascii=False)}

{JAPANESE_RULE_BLOCK}

Few-shot examples:
{fewshot}

Return only valid JSON:
{{"label": 0 or 1, "reason": "brief"}}
""".strip()


def build_ja_target_normalization_prompt(
    tokens: Sequence[str],
    target_index: int,
    *,
    raw_sentence: str | None = None,
    include_fewshot: bool = True,
) -> str:
    marked = [f"<TARGET>{tok}</TARGET>" if i == target_index else tok for i, tok in enumerate(tokens)]
    target = tokens[target_index]
    fewshot = _format_few_shot(FEW_SHOT_EXAMPLES) if include_fewshot else ""
    return f"""
You are a Japanese lexical-normalization resolver for MultiLexNorm-style data.

Normalize ONLY the TARGET token/span if it should be normalized. Do not output the full sentence.
If the target should be preserved or you are uncertain, return the original target as normalized.

Target index: {target_index}
Target raw: {json.dumps(target, ensure_ascii=False)}
Raw sentence for context:
{raw_sentence if raw_sentence is not None else ""}

Indexed/tokenized sentence with target:
{json.dumps(marked, ensure_ascii=False)}

{JAPANESE_RULE_BLOCK}

Few-shot examples:
{fewshot}

Return only valid JSON:
{{"index": {target_index}, "raw": {json.dumps(target, ensure_ascii=False)}, "normalized": "minimal normalized string", "confidence": 0.0}}
""".strip()

# Backward-compatible aliases
build_ja_detection_prompt = build_ja_target_detection_prompt
build_ja_target_prompt = build_ja_target_normalization_prompt

__all__ = [
    "LANG",
    "COMMON_JA_NORMALIZATION_CUES",
    "BOUNDARY_SENSITIVE_CUES",
    "AMBIGUOUS_OR_CONTEXTUAL_TOKENS",
    "JAPANESE_RULE_BLOCK",
    "FEW_SHOT_EXAMPLES",
    "normalize_width",
    "is_ja_likely_candidate",
    "candidate_indices",
    "candidate_spans",
    "build_ja_target_detection_prompt",
    "build_ja_target_normalization_prompt",
    "build_ja_detection_prompt",
    "build_ja_target_prompt",
]
