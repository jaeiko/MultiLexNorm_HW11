"""Common prompt utilities for MultiLexNorm-style lexical normalization.

This shared prompt enforces target-only editing, protected-token handling,
and a strict distinction between lexical normalization and translation/paraphrase.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Sequence

PROTECTED_PUNCT = {
    "_", ".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}",
    "-", "—", "–", "…", "\"", "'", "“", "”", "‘", "’", "~", "〜", "・", "|",
}

URL_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
NUMBER_DATE_TIME_RE = re.compile(r"^[0-9]+(?:[.:/\-][0-9]+)*(?:['’][A-Za-zÀ-ž]+)?$|^[0-9]+(?:st|nd|rd|th)$", re.IGNORECASE)
ALNUM_ENTITY_RE = re.compile(r"(?=.*[A-Za-zÀ-ž])(?=.*[0-9])")
HASHTAG_OR_MENTION_RE = re.compile(r"^[#@]\S+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]",
    flags=re.UNICODE,
)

COMMON_PROTECTED_TOKEN_RULE = """
Protected-token rule:
The following tokens must be preserved exactly unless the language-specific examples explicitly show that this exact token type is normalized:
- hashtags starting with #
- mentions starting with @
- URLs and emails
- emojis and emoticons
- numbers, dates, times, ordinals, measurements
- separator tokens such as "_" when used for alignment
- punctuation-only tokens and ellipsis
- acronyms and alphanumeric entity tokens such as BNK48, GPT4, iPhone15, B2
- named entities, usernames, event names, fandom names, product names, song titles, group names
- social-media entities and tags
""".strip()

COMMON_CORE_RULES = """
Core definition:
- This is lexical normalization.
- This is NOT translation.
- This is NOT paraphrasing.
- This is NOT summarization.
- This is NOT rewriting the whole sentence.
- Do not correct syntax, word order, agreement, case government, or style.
- Do not translate loanwords into semantic equivalents.
- Do not translate names, hashtags, mentions, or foreign words.
- Do not convert numbers into words.
- Do not remove @mentions or hashtags.
- Use the raw sentence only as context.
- Use the indexed tokenized sentence to preserve token alignment.
- If uncertain, preserve the original token.

Code-switching rule:
- When a sentence contains multiple languages, normalize only the surface form of the TARGET token according to its own language.
- Do not translate the TARGET token into another language.
- If the TARGET language is unclear, preserve unless the local context strongly supports a minimal surface correction.

Proper noun / clitic rule:
- Proper names and entities should be preserved unless the annotation style clearly supports a minimal surface correction such as capitalization or apostrophe restoration.
- If the TARGET may require splitting, clitic expansion, or apostrophe insertion, return only the normalized form for the TARGET token/span. Do not rewrite neighboring tokens.
""".strip()


def is_protected_token(tok: str) -> bool:
    if tok is None:
        return True
    s = str(tok).strip()
    if not s:
        return True
    if s in PROTECTED_PUNCT:
        return True
    if HASHTAG_OR_MENTION_RE.match(s):
        return True
    if URL_RE.match(s):
        return True
    if EMAIL_RE.match(s):
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
    return False


def find_protected_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_protected_token(tok)]


def format_indexed_tokens(tokens: Sequence[str]) -> str:
    return "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens))


def _format_few_shot(examples: object) -> str:
    if not examples:
        return "(none)"
    if isinstance(examples, str):
        return examples.strip()
    if isinstance(examples, dict):
        examples = examples.get("examples", examples)
    if not isinstance(examples, Iterable):
        return str(examples)
    blocks: list[str] = []
    for idx, ex in enumerate(examples, 1):
        if isinstance(ex, str):
            blocks.append(f"Example {idx}:\n{ex}")
            continue
        if not isinstance(ex, dict):
            blocks.append(f"Example {idx}:\n{str(ex)}")
            continue
        lines = [f"Example {idx}:"]
        if "raw_sentence" in ex:
            lines.append(f"Raw sentence: {ex['raw_sentence']}")
        if "tokens" in ex:
            lines.append("Tokens: " + json.dumps(ex["tokens"], ensure_ascii=False))
        if "target_index" in ex:
            lines.append(f"Target index: {ex['target_index']}")
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


def build_common_detection_prompt(*, lang: str, sentence_tokens: Sequence[str], target_index: int, language_rule_block: str, fewshot_examples: object | None = None, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None, target_text: str | None = None) -> str:
    tokens = list(sentence_tokens)
    if target_index < 0 or target_index >= len(tokens):
        raise IndexError(f"target_index={target_index} out of range for {len(tokens)} tokens")
    target_text = tokens[target_index] if target_text is None else target_text
    protected = list(find_protected_indices(tokens) if protected_indices is None else protected_indices)
    raw_sentence = " ".join(tokens) if raw_sentence is None else raw_sentence
    fewshot = _format_few_shot(fewshot_examples)
    return f"""
You are a lexical-normalization detector for MultiLexNorm-style social media text.

Task:
Decide whether the TARGET token/span should be changed under the dataset's lexical-normalization annotation style.

{COMMON_CORE_RULES}

Detection label:
- Output 1 only if the TARGET token/span should have a normalized surface form that differs from the raw form.
- Output 0 if the TARGET token/span should be preserved exactly.
- Do not infer that a token needs normalization only because it is informal, expressive, slang-like, foreign-looking, code-switched, or social-media-specific.

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
index: {target_index}
token/span: {target_text}

Return only valid JSON:
{{"label": 0 or 1}}
""".strip()


def build_common_normalization_prompt(*, lang: str, sentence_tokens: Sequence[str], target_index: int, language_rule_block: str, fewshot_examples: object | None = None, raw_sentence: str | None = None, protected_indices: Sequence[int] | None = None, target_text: str | None = None) -> str:
    tokens = list(sentence_tokens)
    if target_index < 0 or target_index >= len(tokens):
        raise IndexError(f"target_index={target_index} out of range for {len(tokens)} tokens")
    target_text = tokens[target_index] if target_text is None else target_text
    protected = list(find_protected_indices(tokens) if protected_indices is None else protected_indices)
    raw_sentence = " ".join(tokens) if raw_sentence is None else raw_sentence
    fewshot = _format_few_shot(fewshot_examples)
    return f"""
You are a lexical-normalization model for MultiLexNorm-style social media text.

Task:
Normalize ONLY the TARGET token/span.

{COMMON_CORE_RULES}

Normalization rules:
- Do not change any token other than the TARGET token/span.
- Preserve the original meaning with the minimum necessary surface-form edit.
- If the target is already acceptable under the dataset annotation style, return it unchanged.
- If uncertain, return the raw target unchanged.

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
index: {target_index}
raw: {target_text}

Return only valid JSON:
{{"index": {target_index}, "raw": {json.dumps(target_text, ensure_ascii=False)}, "normalized": "..."}}
""".strip()


def enforce_protected_output(raw_token: str, normalized_token: str) -> str:
    return raw_token if is_protected_token(raw_token) else normalized_token


__all__ = [
    "PROTECTED_PUNCT", "COMMON_PROTECTED_TOKEN_RULE", "COMMON_CORE_RULES",
    "is_protected_token", "find_protected_indices", "format_indexed_tokens",
    "build_common_detection_prompt", "build_common_normalization_prompt", "enforce_protected_output",
]
