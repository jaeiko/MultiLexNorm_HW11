"""Common prompt utilities for MultiLexNorm-style lexical normalization.

Version: v2 unified core prompt

Design goals
------------
1. Give the model enough context: raw sentence + indexed tokenized sentence.
2. Restrict the model's authority: target token/span only.
3. Prevent common failures: translation, paraphrase, full-sentence rewrite,
   entity/hashtag/number corruption, and aggressive capitalization.
4. Support language-specific rule blocks and few-shot examples without
   hard-coding one language's policy into the common layer.
5. Provide code-side guards, because prompt instructions alone are not enough.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Protected-token detection
# ---------------------------------------------------------------------------

PROTECTED_PUNCT = {
    "_", ".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}",
    "-", "—", "–", "…", "...", "..", "....", "\"", "'", "’", "‘", "“", "”",
    "~", "〜", "・", "|", "/", "\\", "*", "+", "=", "<", ">",
}

URL_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
HASHTAG_OR_MENTION_RE = re.compile(r"^[#@]\S+")
PLACEHOLDER_RE = re.compile(r"^(?:\[url\]|\[URL\]|<url>|<URL>|@username|@mention)$")

# Numbers, dates, times, ordinals, decimals, percentages, currencies and common ranges.
NUMBER_DATE_TIME_RE = re.compile(
    r"^(?:[$€£₩¥])?\d+(?:[.,:/\-]\d+)*(?:%|[a-zA-Z]{0,4})?$"
    r"|^\d+(?:st|nd|rd|th)$"
    r"|^\d+\s?(?:am|pm|AM|PM)$",
    re.IGNORECASE,
)

# Alphanumeric entities: BNK48, GPT4, iPhone15, KPOP2024, B2, A7X, etc.
ALNUM_ENTITY_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)")

# Compact emoji coverage plus common text emoticons.
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]",
    flags=re.UNICODE,
)
EMOTICON_RE = re.compile(r"^(?:[:;=xX8][-o*']?[)D(Pp/\\]|[<]?3|T_T|t_t|ㅠㅠ+|ㅜㅜ+|[xX]D+|-_-+|\^_?\^+)$")

# Some tokens are social-media commands or entity-like markers. We do not make
# all uppercase words protected in code because some datasets normalize RT/DM/TL.
# Language-specific modules can mark extra tokens as protected if needed.
DEFAULT_SOCIAL_MARKERS = {"RT", "QT", "MT"}

COMMON_PROTECTED_TOKEN_RULE = """
Protected-token rule:
The following tokens must be preserved exactly unless the language-specific examples explicitly show that this exact token is normalized in this dataset:
- hashtags starting with #
- mentions starting with @
- URLs, emails, and placeholder tokens such as [url] or @username
- emojis and emoticons
- numbers, dates, times, ordinals, measurements, prices, percentages, and numeric ranges
- separator tokens such as "_" when used for alignment
- punctuation-only tokens, punctuation sequences, and ellipsis
- alphanumeric entity tokens such as BNK48, GPT4, iPhone15, B2, A7X
- named entities, usernames, event names, fandom names, product names, song titles, group names, institution names, and social-media tags
""".strip()

COMMON_CORE_RULES = """
Core task definition:
- This is lexical normalization, not translation.
- This is surface-form normalization, not paraphrasing.
- Do not summarize, explain, complete, or rewrite the sentence.
- Do not correct syntax, word order, agreement, case government, tense, or style.
- Do not replace colloquial words with semantic standard equivalents unless the dataset-style examples support that exact lexical normalization.
- Do not translate loanwords, foreign words, names, hashtags, mentions, or entities into semantic equivalents.
- Do not convert numbers into words.
- Do not remove or invent hashtags, mentions, URLs, emojis, separators, or punctuation.
- Use the raw sentence only as context.
- Use the indexed tokenized sentence to preserve alignment.
- If uncertain, preserve the original target.

Target-only editing rule:
- Normalize only the TARGET token/span.
- Do not change any neighboring token.
- If a normalized form contains spaces, return it as one normalized string for the target token/span.
- Do not output the full normalized sentence.

Code-switching rule:
- When a sentence contains multiple languages, normalize only the surface form of the TARGET according to its own language.
- Do not translate a token from one language into another language.
- If the TARGET language is unclear, preserve unless local context strongly supports a minimal surface correction.

Capitalization rule:
- Do not change capitalization only because a token appears sentence-initial.
- Apply capitalization only when the dataset-style examples or local context strongly support it, especially for proper names, German nouns, acronyms, or sentence-initial casing.

Entity and loanword rule:
- Preserve named entities and loanwords unless the expected change is a minimal spelling/orthographic correction, not semantic translation.
- For social-media tags, usernames, fandom tags, and product/event/group names, default to preserve.

Short-token ambiguity rule:
- Short tokens and function-like forms are often ambiguous.
- If several normalizations are plausible, choose preserve in detection or return the raw target in normalization.
""".strip()

DETECTION_TASK_RULES = """
Detection label:
- Output 1 only if the TARGET token/span should have a normalized surface form that differs from the raw form under the dataset annotation style.
- Output 0 if the TARGET token/span should be preserved exactly.
- Do not mark a target as 1 merely because it is informal, expressive, slang-like, dialectal, foreign-looking, code-switched, emoji-like, or social-media-specific.
- Protected targets should receive label 0 unless the language-specific examples explicitly show that this exact target is normalized.
""".strip()

NORMALIZATION_TASK_RULES = """
Normalization rules:
- Return the normalized form of the TARGET token/span only.
- Preserve the original meaning with the minimum necessary surface-form edit.
- If the target is already acceptable under the dataset annotation style, return it unchanged.
- If uncertain, return the raw target unchanged.
- Never return an empty string unless the language-specific examples explicitly show that this exact target should be deleted.
- Do not add surrounding words that do not belong to the target normalization.
""".strip()


def is_protected_token(tok: str | None, *, extra_protected: set[str] | None = None) -> bool:
    """Return True when a token should be protected from automatic normalization.

    This function is intentionally conservative for social-media entities and
    numeric/alphanumeric tokens. It does not protect all uppercase tokens because
    some language-specific modules may normalize RT/DM/TL-like tokens.
    """
    if tok is None:
        return True
    s = str(tok).strip()
    if not s:
        return True
    if extra_protected and s in extra_protected:
        return True
    if s in PROTECTED_PUNCT:
        return True
    if s in DEFAULT_SOCIAL_MARKERS:
        return True
    if PLACEHOLDER_RE.fullmatch(s):
        return True
    if HASHTAG_OR_MENTION_RE.match(s):
        return True
    if URL_RE.match(s):
        return True
    if EMAIL_RE.fullmatch(s):
        return True
    if NUMBER_DATE_TIME_RE.fullmatch(s):
        return True
    if ALNUM_ENTITY_RE.search(s):
        return True
    if "#" in s or "@" in s:
        return True
    if all(ch in PROTECTED_PUNCT for ch in s):
        return True
    if EMOJI_RE.search(s):
        return True
    if EMOTICON_RE.fullmatch(s):
        return True
    return False


def find_protected_indices(tokens: Sequence[str], *, extra_protected: set[str] | None = None) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_protected_token(tok, extra_protected=extra_protected)]


def format_indexed_tokens(tokens: Sequence[str]) -> str:
    return "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens))


def _coerce_target_indices(*, target_index: int | None = None, target_indices: Sequence[int] | None = None) -> list[int]:
    if target_indices is not None:
        indices = [int(i) for i in target_indices]
    elif target_index is not None:
        indices = [int(target_index)]
    else:
        raise ValueError("Either target_index or target_indices must be provided.")
    if not indices:
        raise ValueError("target_indices cannot be empty.")
    return indices


def _target_text(tokens: Sequence[str], indices: Sequence[int], target_text: str | None = None) -> str:
    if target_text is not None:
        return target_text
    return " ".join(tokens[i] for i in indices)


def _validate_indices(tokens: Sequence[str], indices: Sequence[int]) -> None:
    for i in indices:
        if i < 0 or i >= len(tokens):
            raise IndexError(f"target index {i} out of range for {len(tokens)} tokens")


def _format_few_shot(examples: object) -> str:
    if not examples:
        return "(none)"
    if isinstance(examples, str):
        return examples.strip()
    if isinstance(examples, Mapping):
        examples = examples.get("examples", examples)
    if not isinstance(examples, Iterable):
        return str(examples)

    blocks: list[str] = []
    for idx, ex in enumerate(examples, 1):
        if isinstance(ex, str):
            blocks.append(f"Example {idx}:\n{ex}")
            continue
        if not isinstance(ex, Mapping):
            blocks.append(f"Example {idx}:\n{str(ex)}")
            continue
        lines = [f"Example {idx}:"]
        if "raw_sentence" in ex:
            lines.append(f"Raw sentence: {ex['raw_sentence']}")
        if "tokens" in ex:
            lines.append("Tokens: " + json.dumps(ex["tokens"], ensure_ascii=False))
        if "target_index" in ex:
            lines.append(f"Target index: {ex['target_index']}")
        if "target_indices" in ex:
            lines.append("Target indices: " + json.dumps(ex["target_indices"], ensure_ascii=False))
        if "target" in ex:
            lines.append(f"Target: {ex['target']}")
        if "label" in ex:
            lines.append("Detection output: " + json.dumps({"label": ex["label"]}, ensure_ascii=False))
        if "normalized" in ex:
            lines.append("Normalization output: " + json.dumps({"normalized": ex["normalized"]}, ensure_ascii=False))
        if "notes" in ex:
            lines.append(f"Reason: {ex['notes']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_common_detection_prompt(
    *,
    lang: str,
    sentence_tokens: Sequence[str],
    language_rule_block: str,
    target_index: int | None = None,
    target_indices: Sequence[int] | None = None,
    fewshot_examples: object | None = None,
    raw_sentence: str | None = None,
    protected_indices: Sequence[int] | None = None,
    target_text: str | None = None,
) -> str:
    tokens = list(sentence_tokens)
    indices = _coerce_target_indices(target_index=target_index, target_indices=target_indices)
    _validate_indices(tokens, indices)
    target = _target_text(tokens, indices, target_text=target_text)
    protected = list(find_protected_indices(tokens) if protected_indices is None else protected_indices)
    raw_sentence = " ".join(tokens) if raw_sentence is None else raw_sentence
    fewshot = _format_few_shot(fewshot_examples)

    target_descriptor = (
        f"index: {indices[0]}\ntoken/span: {target}"
        if len(indices) == 1
        else f"indices: {json.dumps(indices, ensure_ascii=False)}\ntoken/span: {target}"
    )

    return f"""
You are a lexical-normalization detector for MultiLexNorm-style social media text.

Task:
Decide whether the TARGET token/span should be changed under the dataset's lexical-normalization annotation style.

{COMMON_CORE_RULES}

{DETECTION_TASK_RULES}

{COMMON_PROTECTED_TOKEN_RULE}

Language:
{lang}

Language-specific guidance:
{language_rule_block.strip()}

Language-specific few-shot examples:
{fewshot}

Raw sentence for context:
{raw_sentence}

Indexed tokenized sentence:
{format_indexed_tokens(tokens)}

Protected indices:
{json.dumps(protected, ensure_ascii=False)}

TARGET:
{target_descriptor}

Return only valid JSON:
{{"label": 0 or 1}}
""".strip()


def build_common_normalization_prompt(
    *,
    lang: str,
    sentence_tokens: Sequence[str],
    language_rule_block: str,
    target_index: int | None = None,
    target_indices: Sequence[int] | None = None,
    fewshot_examples: object | None = None,
    raw_sentence: str | None = None,
    protected_indices: Sequence[int] | None = None,
    target_text: str | None = None,
) -> str:
    tokens = list(sentence_tokens)
    indices = _coerce_target_indices(target_index=target_index, target_indices=target_indices)
    _validate_indices(tokens, indices)
    target = _target_text(tokens, indices, target_text=target_text)
    protected = list(find_protected_indices(tokens) if protected_indices is None else protected_indices)
    raw_sentence = " ".join(tokens) if raw_sentence is None else raw_sentence
    fewshot = _format_few_shot(fewshot_examples)

    if len(indices) == 1:
        output_schema = {"index": indices[0], "raw": target, "normalized": "..."}
        target_descriptor = f"index: {indices[0]}\nraw: {target}"
    else:
        output_schema = {"indices": indices, "raw": target, "normalized": "..."}
        target_descriptor = f"indices: {json.dumps(indices, ensure_ascii=False)}\nraw: {target}"

    return f"""
You are a lexical-normalization model for MultiLexNorm-style social media text.

Task:
Normalize ONLY the TARGET token/span.

{COMMON_CORE_RULES}

{NORMALIZATION_TASK_RULES}

{COMMON_PROTECTED_TOKEN_RULE}

Language:
{lang}

Language-specific guidance:
{language_rule_block.strip()}

Language-specific few-shot examples:
{fewshot}

Raw sentence for context:
{raw_sentence}

Indexed tokenized sentence:
{format_indexed_tokens(tokens)}

Protected indices:
{json.dumps(protected, ensure_ascii=False)}

TARGET:
{target_descriptor}

Return only valid JSON:
{json.dumps(output_schema, ensure_ascii=False)}
""".strip()


# ---------------------------------------------------------------------------
# Output parsing and code-side guardrails
# ---------------------------------------------------------------------------


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response.

    The prompt asks for JSON only, but this helper is tolerant of accidental
    wrapping text or code fences.
    """
    if text is None:
        return None
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    while start != -1:
        depth = 0
        for pos in range(start, len(s)):
            if s[pos] == "{":
                depth += 1
            elif s[pos] == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : pos + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return None


def parse_detection_output(text: str) -> int | None:
    obj = extract_first_json_object(text)
    if not obj or "label" not in obj:
        return None
    label = obj.get("label")
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)) and int(label) in {0, 1}:
        return int(label)
    if isinstance(label, str) and label.strip() in {"0", "1"}:
        return int(label.strip())
    return None


def parse_normalization_output(text: str) -> dict[str, Any] | None:
    obj = extract_first_json_object(text)
    if not obj or "normalized" not in obj:
        return None
    return obj


def enforce_protected_output(raw_token: str, normalized_token: str | None, *, allow_empty: bool = False) -> str:
    """Protect tokens and avoid blank outputs by default."""
    if is_protected_token(raw_token):
        return raw_token
    if normalized_token is None:
        return raw_token
    norm = str(normalized_token).strip()
    if not norm and not allow_empty:
        return raw_token
    return norm


def safe_normalization_result(
    *,
    raw_target: str,
    normalized: str | None,
    allow_empty: bool = False,
    max_expansion_tokens: int = 6,
) -> str:
    """Apply conservative post-processing to one target normalization.

    This is intentionally language-agnostic. Language-specific modules may use
    stricter checks after this function.
    """
    norm = enforce_protected_output(raw_target, normalized, allow_empty=allow_empty)
    if norm == raw_target:
        return norm
    if not norm and allow_empty:
        return norm

    # Avoid obvious full-sentence rewrites. One-to-many lexical normalization is
    # allowed, but very long expansions are usually paraphrase/translation errors.
    if len(norm.split()) > max_expansion_tokens:
        return raw_target

    # Protect hashtags/mentions/entities even when embedded in a larger string.
    if raw_target.startswith("#") or raw_target.startswith("@") or "#" in raw_target or "@" in raw_target:
        return raw_target

    return norm


__all__ = [
    "PROTECTED_PUNCT",
    "COMMON_PROTECTED_TOKEN_RULE",
    "COMMON_CORE_RULES",
    "DETECTION_TASK_RULES",
    "NORMALIZATION_TASK_RULES",
    "is_protected_token",
    "find_protected_indices",
    "format_indexed_tokens",
    "build_common_detection_prompt",
    "build_common_normalization_prompt",
    "extract_first_json_object",
    "parse_detection_output",
    "parse_normalization_output",
    "enforce_protected_output",
    "safe_normalization_result",
]
