"""Turkish-German code-switched prompt rules and candidate selection."""
from __future__ import annotations

import re
from typing import Sequence

try:
    from prompts.common_prompt import build_common_detection_prompt, build_common_normalization_prompt, is_protected_token
    from language_rules.turkish_common import TURKISH_COMMON_RULE_BLOCK, is_turkish_likely_candidate
except Exception:  # pragma: no cover
    from ..prompts.common_prompt import build_common_detection_prompt, build_common_normalization_prompt, is_protected_token
    from .turkish_common import TURKISH_COMMON_RULE_BLOCK, is_turkish_likely_candidate

LANG = "trde"

LANGUAGE_RULE_BLOCK = TURKISH_COMMON_RULE_BLOCK + """

Turkish-German code-switching policy:
- First decide whether the TARGET is Turkish, German, mixed Turkish+German, named entity, or protected/other.
- Normalize Turkish tokens using Turkish surface-normalization rules.
- Normalize German tokens using German surface-normalization rules, especially minimal spelling or capitalization correction when supported by dataset evidence.
- Do NOT translate Turkish into German or German into Turkish.
- German nouns may require capitalization, e.g. schule→Schule, but only when the target is clearly a German noun or dataset examples support it.
- Mixed tokens may contain a German stem plus Turkish suffix; handle only minimal surface/apostrophe/diacritic correction and preserve if uncertain.
- Ambiguous tokens such as ne, i, de, da, oda, party, foto should be routed to context-aware prompt/model fallback, not blind MFR.
""".strip()

FEW_SHOT_EXAMPLES = [
    {"raw_sentence": "ich hab cok özledim", "tokens": ["ich", "hab", "cok", "özledim"], "target_index": 1, "target": "hab", "label": 1, "normalized": "habe", "notes": "German informal form normalized in German, not translated into Turkish."},
    {"raw_sentence": "ich hab cok özledim", "tokens": ["ich", "hab", "cok", "özledim"], "target_index": 2, "target": "cok", "label": 1, "normalized": "çok", "notes": "Turkish deasciification inside code-switched sentence."},
    {"raw_sentence": "almanyada Schule var", "tokens": ["almanyada", "Schule", "var"], "target_index": 0, "target": "almanyada", "label": 1, "normalized": "Almanya'da", "notes": "Turkish proper noun + locative suffix."},
    {"raw_sentence": "ich gehe zur schule", "tokens": ["ich", "gehe", "zur", "schule"], "target_index": 3, "target": "schule", "label": 1, "normalized": "Schule", "notes": "German noun capitalization, not Turkish translation."},
    {"raw_sentence": "ne oldu", "tokens": ["ne", "oldu"], "target_index": 0, "target": "ne", "label": 0, "normalized": "ne", "notes": "Ambiguous across Turkish/German; preserve unless context clearly supports German eine."},
    {"raw_sentence": "@username ich cok mutluyum #Berlin", "tokens": ["@username", "ich", "cok", "mutluyum", "#Berlin"], "target_index": 4, "target": "#Berlin", "label": 0, "normalized": "#Berlin", "notes": "Hashtag is protected."},
]

GERMAN_CANDIDATE_FORMS = {
    "ich", "hab", "nich", "ma", "schule", "theorie", "ethik", "deutsch", "deutsche", "deutschen", "almanca", "deutschland", "berlin", "semester", "uni",
}
AMBIGUOUS_TRDE_FORMS = {"ne", "i", "de", "da", "oda", "party", "u", "foto", "bende", "sende", "das", "DAS"}
GERMAN_NOUN_LOWER_RE = re.compile(r"^(schule|theorie|ethik|semester|universität|uni|bahn|deutsch|deutschland|sprache)$", re.I)


def is_trde_likely_candidate(tok: str) -> bool:
    s = str(tok).strip()
    if not s or is_protected_token(s):
        return False
    low = s.lower()
    if low in GERMAN_CANDIDATE_FORMS:
        return True
    if s in AMBIGUOUS_TRDE_FORMS or low in AMBIGUOUS_TRDE_FORMS:
        return True
    if GERMAN_NOUN_LOWER_RE.match(s):
        return True
    if is_turkish_likely_candidate(s):
        return True
    return False


def candidate_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if is_trde_likely_candidate(str(tok))]


def build_trde_target_detection_prompt(sentence_tokens: Sequence[str], target_index: int, raw_sentence: str | None = None) -> str:
    return build_common_detection_prompt(lang=LANG, sentence_tokens=sentence_tokens, target_index=target_index, raw_sentence=raw_sentence, language_rule_block=LANGUAGE_RULE_BLOCK, fewshot_examples=FEW_SHOT_EXAMPLES)


def build_trde_target_normalization_prompt(sentence_tokens: Sequence[str], target_index: int, raw_sentence: str | None = None) -> str:
    return build_common_normalization_prompt(lang=LANG, sentence_tokens=sentence_tokens, target_index=target_index, raw_sentence=raw_sentence, language_rule_block=LANGUAGE_RULE_BLOCK, fewshot_examples=FEW_SHOT_EXAMPLES)

# Backward-compatible aliases
build_target_detection_prompt = build_trde_target_detection_prompt
build_target_normalization_prompt = build_trde_target_normalization_prompt
