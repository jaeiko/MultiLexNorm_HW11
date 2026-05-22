"""LLM correction prompt 빌더 어댑터 — 최종 파이프라인의 prompt 단계.

PromptMFRResources: prompt_mfr_dictionary의 17개 언어 패키지를 로드하고
build_normalization_prompt()로 LLM user prompt(언어별 본체 + MFR hint +
dynamic fewshot 블록)를 조립한다. llm_correct_local.py가 사용.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class LanguagePackageConfig:
    package_dir: str
    rule_file: str
    dictionary_file: str
    module_lang: str


LANGUAGE_PACKAGES: dict[str, LanguagePackageConfig] = {
    "da": LanguagePackageConfig(
        "danish_normalization_package",
        "language_rules/da.py",
        "mfr_dictionaries/da_mfr_dictionary.json",
        "da",
    ),
    "de": LanguagePackageConfig(
        "german_normalization_package",
        "language_rules/de.py",
        "mfr_dictionaries/de_mfr_dictionary.json",
        "de",
    ),
    "en": LanguagePackageConfig(
        "english_normalization_package",
        "language_rules/en.py",
        "mfr_dictionaries/en_mfr_dictionary.json",
        "en",
    ),
    "es": LanguagePackageConfig(
        "spanish_normalization_package",
        "language_rules/es.py",
        "mfr_dictionaries/es_mfr_dictionary.json",
        "es",
    ),
    "hr": LanguagePackageConfig(
        "south_slavic_normalization_package",
        "language_rules/hr.py",
        "mfr_dictionaries/hr_mfr_dictionary.json",
        "hr",
    ),
    "id": LanguagePackageConfig(
        "indonesian_normalization_package",
        "language_rules/id.py",
        "mfr_dictionaries/id_mfr_dictionary.json",
        "id",
    ),
    "iden": LanguagePackageConfig(
        "indonesian_normalization_package",
        "language_rules/id.py",
        "mfr_dictionaries/id_mfr_dictionary.json",
        "id",
    ),
    "it": LanguagePackageConfig(
        "italian_normalization_package",
        "language_rules/it.py",
        "mfr_dictionaries/it_mfr_dictionary.json",
        "it",
    ),
    "ja": LanguagePackageConfig(
        "japanese_normalization_package",
        "language_rules/ja.py",
        "mfr_dictionaries/ja_mfr_dictionary.json",
        "ja",
    ),
    "ko": LanguagePackageConfig(
        "korean_normalization_package",
        "language_rules/ko.py",
        "mfr_dictionaries/ko_mfr_dictionary.json",
        "ko",
    ),
    "nl": LanguagePackageConfig(
        "dutch_normalization_package",
        "language_rules/nl.py",
        "mfr_dictionaries/nl_mfr_dictionary.json",
        "nl",
    ),
    "sl": LanguagePackageConfig(
        "south_slavic_normalization_package",
        "language_rules/sl.py",
        "mfr_dictionaries/sl_mfr_dictionary.json",
        "sl",
    ),
    "sr": LanguagePackageConfig(
        "south_slavic_normalization_package",
        "language_rules/sr.py",
        "mfr_dictionaries/sr_mfr_dictionary.json",
        "sr",
    ),
    "th": LanguagePackageConfig(
        "thai_normalization_package",
        "language_rules/th.py",
        "mfr_dictionaries/th_mfr_dictionary.json",
        "th",
    ),
    "tr": LanguagePackageConfig(
        "turkish_normalization_package",
        "language_rules/tr.py",
        "mfr_dictionaries/tr_mfr_dictionary.json",
        "tr",
    ),
    "trde": LanguagePackageConfig(
        "turkish_normalization_package",
        "language_rules/trde.py",
        "mfr_dictionaries/trde_mfr_dictionary.json",
        "trde",
    ),
    "vi": LanguagePackageConfig(
        "vietnamese_normalization_package",
        "language_rules/vi.py",
        "mfr_dictionaries/vi_mfr_dictionary.json",
        "vi",
    ),
}


@contextmanager
def _temporary_import_scope(package_dir: Path):
    """Load one language package without leaking its local language_rules package."""
    old_path = list(sys.path)
    old_language_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "language_rules" or name.startswith("language_rules.")
    }

    for name in old_language_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(package_dir))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == "language_rules" or name.startswith("language_rules."):
                sys.modules.pop(name, None)
        sys.modules.update(old_language_modules)


def _import_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _looks_context_sensitive(value: Any) -> bool:
    return isinstance(value, dict) and bool(
        value.get("context_sensitive")
        or value.get("ambiguous")
        or value.get("is_context_sensitive")
    )


def _norm_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        norm = value.get("norm") or value.get("normalized") or value.get("top_norm")
        return norm if isinstance(norm, str) else None
    return None


def _iter_pair_items(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        yield from value.items()
        return

    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            raw = item.get("raw") or item.get("token")
            if isinstance(raw, str):
                yield raw, item


def _candidate_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        summary: dict[str, Any] = {}
        for key in (
            "norm",
            "normalized",
            "top_norm",
            "count",
            "best_count",
            "top_count",
            "changed_count",
            "changed_total",
            "raw_total",
            "total_count",
            "confidence",
            "changed_confidence",
            "raw_confidence",
            "top_confidence_among_changed",
            "change_ratio",
            "change_rate",
            "raw_count",
        ):
            if key in value:
                summary[key] = value[key]
        norm = _norm_from_value(value)
        if norm is not None:
            summary["norm"] = norm
        candidates = value.get("candidates") or value.get("variants")
        if candidates is not None:
            summary["candidates"] = candidates
        return summary
    if isinstance(value, str):
        return {"norm": value}
    return {}


class PromptMFRResources:
    """Adapter around prompt_mfr_dictionary.

    The language packages were delivered as small standalone packages. This
    adapter gives the main pipeline one stable API:
    protected-token policy, high-confidence MFR, candidate selection, prompt
    building, output parsing, and safe post-processing.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initializes the PromptMFRResources adapter.

        Args:
            base_dir: Path to the prompt_mfr_dictionary directory. If None, it
                resolves to the directory of this script.

        Raises:
            FileNotFoundError: If the prompt_mfr_dictionary directory does not exist.
        """
        repo_root = Path(__file__).resolve().parent
        self.base_dir = Path(base_dir) if base_dir is not None else repo_root / "prompt_mfr_dictionary"
        if not self.base_dir.exists():
            raise FileNotFoundError(f"prompt_mfr_dictionary not found: {self.base_dir}")

        self.common_prompt = self._load_common_prompt()
        self.rule_modules: dict[str, Any] = {}
        self.high_confidence_pairs: dict[str, dict[str, str]] = {}
        self.prompt_hints: dict[str, dict[str, list[dict[str, Any]]]] = {}

        self._load_all_resources()

    @staticmethod
    def normalize_lang(lang: str | None) -> str:
        if lang is None:
            return ""
        lang = str(lang).strip().lower()
        aliases = {
            "id-en": "iden",
            "id_en": "iden",
            "tr-de": "trde",
            "tr_de": "trde",
        }
        return aliases.get(lang, lang)

    def _load_common_prompt(self):
        common_path = self.base_dir / "common_prompt_v2_package" / "prompts" / "common_prompt.py"
        common_module = _import_module_from_file("prompt_mfr_common_prompt_v2", common_path)

        prompts_package = types.ModuleType("prompts")
        prompts_package.__path__ = []  # type: ignore[attr-defined]
        sys.modules["prompts"] = prompts_package
        sys.modules["prompts.common_prompt"] = common_module
        return common_module

    def _load_all_resources(self) -> None:
        for lang, config in LANGUAGE_PACKAGES.items():
            self.rule_modules[lang] = self._load_rule_module(lang, config)
            self._load_mfr_dictionary(lang, config)

    def _load_rule_module(self, lang: str, config: LanguagePackageConfig):
        package_dir = self.base_dir / config.package_dir
        rule_path = package_dir / config.rule_file
        with _temporary_import_scope(package_dir):
            return _import_module_from_file(f"prompt_mfr_rule_{lang}", rule_path)

    def _load_mfr_dictionary(self, lang: str, config: LanguagePackageConfig) -> None:
        dictionary_path = self.base_dir / config.package_dir / config.dictionary_file
        if not dictionary_path.exists():
            self.high_confidence_pairs[lang] = {}
            self.prompt_hints[lang] = {}
            return

        with dictionary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.high_confidence_pairs[lang] = self._extract_high_confidence_pairs(payload, lang)
        self.prompt_hints[lang] = self._extract_prompt_hints(payload, lang)

    def _language_section(self, payload: dict[str, Any], lang: str) -> dict[str, Any]:
        languages = payload.get("languages")
        if isinstance(languages, dict):
            section = languages.get(lang)
            if isinstance(section, dict):
                return section
            config = LANGUAGE_PACKAGES.get(lang)
            if config is not None:
                section = languages.get(config.module_lang)
                if isinstance(section, dict):
                    return section
        return payload

    def _extract_high_confidence_pairs(self, payload: dict[str, Any], lang: str) -> dict[str, str]:
        section = self._language_section(payload, lang)
        raw_pairs = section.get("high_confidence_pairs", {})
        pairs: dict[str, str] = {}

        for raw, value in _iter_pair_items(raw_pairs):
            if _looks_context_sensitive(value):
                continue
            norm = _norm_from_value(value)
            if norm is not None:
                pairs[raw] = norm

        return pairs

    def _extract_prompt_hints(self, payload: dict[str, Any], lang: str) -> dict[str, list[dict[str, Any]]]:
        section = self._language_section(payload, lang)
        hints: dict[str, list[dict[str, Any]]] = {}

        for field in ("review_pairs", "ambiguous_pairs", "context_sensitive_pairs"):
            for raw, value in _iter_pair_items(section.get(field, {})):
                hints.setdefault(raw, []).append(
                    {
                        "source": field,
                        **_candidate_summary(value),
                    }
                )

        # High-confidence entries marked context-sensitive should not be applied
        # automatically, but they are valuable context for the LLM prompt.
        for raw, value in _iter_pair_items(section.get("high_confidence_pairs", {})):
            if _looks_context_sensitive(value):
                hints.setdefault(raw, []).append(
                    {
                        "source": "high_confidence_context_sensitive",
                        **_candidate_summary(value),
                    }
                )

        return hints

    def find_protected_indices(self, tokens: Sequence[str]) -> list[int]:
        """Finds indices of protected tokens in the given list of tokens.

        Args:
            tokens: Sequence of token strings.

        Returns:
            A list of 0-based token indices that are protected from MFR normalization.
        """
        return list(self.common_prompt.find_protected_indices(tokens))

    def apply_high_confidence_mfr(
        self,
        tokens: Sequence[str],
        lang: str | None,
        *,
        protected_indices: Iterable[int] | None = None,
    ) -> tuple[list[str], list[int]]:
        """Applies high confidence MFR replacements on the token list.

        Args:
            tokens: Sequence of token strings.
            lang: Language identifier.
            protected_indices: Iterable of token indices that are protected.

        Returns:
            A tuple containing:
                - The list of normalized tokens.
                - The list of indices that were modified.
        """
        normalized_lang = self.normalize_lang(lang)
        pairs = self.high_confidence_pairs.get(normalized_lang, {})
        protected = set(protected_indices or [])
        output = list(tokens)
        changed: list[int] = []

        for idx, token in enumerate(tokens):
            if idx in protected:
                continue
            replacement = pairs.get(str(token))
            if replacement is None:
                continue
            output[idx] = replacement
            if replacement != token:
                changed.append(idx)

        return output, changed

    def high_confidence_normalization(self, token: str, lang: str | None) -> str | None:
        """Return one high-confidence MFR normalization for a token, if present."""
        normalized_lang = self.normalize_lang(lang)
        return self.high_confidence_pairs.get(normalized_lang, {}).get(str(token))

    def candidate_indices(self, tokens: Sequence[str], lang: str | None) -> list[int]:
        """Identifies candidate indices for LLM-based normalizations.

        Args:
            tokens: Sequence of token strings.
            lang: Language identifier.

        Returns:
            Sorted list of 0-based token indices that are normalization candidates.
        """
        normalized_lang = self.normalize_lang(lang)
        module = self.rule_modules.get(normalized_lang)
        candidates: set[int] = set()

        if module is not None and hasattr(module, "candidate_indices"):
            try:
                if normalized_lang == "iden":
                    candidates.update(module.candidate_indices(tokens, lang="iden"))
                else:
                    candidates.update(module.candidate_indices(tokens))
            except TypeError:
                candidates.update(module.candidate_indices(tokens))

        hints = self.prompt_hints.get(normalized_lang, {})
        for idx, token in enumerate(tokens):
            if str(token) in hints:
                candidates.add(idx)

        return sorted(i for i in candidates if 0 <= i < len(tokens))

    def mfr_fallback_candidate_indices(
        self,
        tokens: Sequence[str],
        lang: str | None,
        *,
        min_confidence: float = 0.30,
        max_candidates: int | None = 3,
    ) -> list[int]:
        """Return suspicious token indices using MFR package candidate logic.

        This is intentionally used only when sentence-level LLM detection says
        "change needed" but XLM-R finds no token. It narrows the old full-sentence
        fallback to tokens selected by language rules and prompt hints.

        min_confidence and max_candidates are kept only for compatibility with
        older experiment configs. Suspicious candidates are not confidence-gated
        because the prompt_mfr_dictionary packages use mixed schemas across
        languages, and many useful ambiguous tokens do not expose a confidence key.
        """
        _ = (min_confidence, max_candidates)
        return self.candidate_indices(tokens, lang)



    def build_normalization_prompt(
        self,
        tokens: Sequence[str],
        target_index: int,
        lang: str | None,
        *,
        dynamic_fewshot_pairs: dict[str, list[dict[str, Any]]] | None = None,
    ) -> str:
        """Constructs a normalization prompt for the LLM.

        Args:
            tokens: Sequence of token strings.
            target_index: Index of the target token to normalize.
            lang: Language identifier.
            dynamic_fewshot_pairs: Dictionary of positive and negative few-shot
                examples.

        Returns:
            The complete prompt string ready for LLM generation.
        """
        return self._build_prompt(
            tokens,
            target_index,
            lang,
            prompt_type="normalization",
            dynamic_fewshot_pairs=dynamic_fewshot_pairs,
        )

    def _build_prompt(
        self,
        tokens: Sequence[str],
        target_index: int,
        lang: str | None,
        *,
        prompt_type: str,
        dynamic_fewshot_pairs: dict[str, list[dict[str, Any]]] | None = None,
    ) -> str:
        normalized_lang = self.normalize_lang(lang)
        module = self.rule_modules.get(normalized_lang)
        if module is None:
            raise ValueError(f"Unsupported language for prompt_mfr_dictionary: {lang!r}")

        module_lang = LANGUAGE_PACKAGES[normalized_lang].module_lang
        names = [
            f"build_{module_lang}_target_{prompt_type}_prompt",
            f"build_{normalized_lang}_target_{prompt_type}_prompt",
            f"build_target_{prompt_type}_prompt",
            f"build_{prompt_type}_prompt",
        ]

        builder = None
        for name in names:
            builder = getattr(module, name, None)
            if builder is not None:
                break
        if builder is None:
            raise AttributeError(f"No {prompt_type} prompt builder for {normalized_lang}")

        raw_sentence = " ".join(str(token) for token in tokens)
        try:
            if normalized_lang == "iden":
                prompt = builder(tokens, target_index, lang="iden", raw_sentence=raw_sentence)
            else:
                prompt = builder(tokens, target_index, raw_sentence=raw_sentence)
        except TypeError:
            prompt = builder(tokens, target_index)

        hint_block = self._format_prompt_hint(str(tokens[target_index]), normalized_lang)
        if hint_block:
            prompt = (
                f"{prompt}\n\n{hint_block}\n\n"
                "Return only the valid JSON object requested above."
            )

        dyn_block = self._format_dynamic_fewshot_block(dynamic_fewshot_pairs)
        if dyn_block:
            prompt = (
                f"{prompt}\n\n{dyn_block}\n\n"
                "Use these similar cases as soft evidence. "
                "Return only the valid JSON object requested above."
            )
        return prompt

    @staticmethod
    def _format_dynamic_fewshot_block(
        pairs: dict[str, list[dict[str, Any]]] | None,
    ) -> str:
        if not pairs:
            return ""

        positive = pairs.get("positive") or []
        negative = pairs.get("negative") or []
        if not positive and not negative:
            return ""

        lines = ["Similar cases from training data:"]
        if positive:
            lines.append("  Positive (these tokens were normalized):")
            for pair in positive:
                lines.append("    " + json.dumps(pair, ensure_ascii=False))
        if negative:
            lines.append("  Negative (these tokens were preserved as-is, raw == norm):")
            for pair in negative:
                lines.append("    " + json.dumps(pair, ensure_ascii=False))
        return "\n".join(lines)

    def _format_prompt_hint(self, token: str, lang: str) -> str:
        hints = self.prompt_hints.get(lang, {}).get(token)
        if not hints:
            return ""

        compact = json.dumps(hints[:5], ensure_ascii=False)
        return (
            "Dataset MFR review hint:\n"
            "- The target appears in review/ambiguous/context-sensitive MFR data.\n"
            "- Use this only as evidence; preserve the raw token if context is uncertain.\n"
            f"- Candidate evidence: {compact}"
        )



    def parse_normalization_output(self, text: str) -> dict[str, Any] | None:
        return self.common_prompt.parse_normalization_output(text)




