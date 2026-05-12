from __future__ import annotations

from typing import List, Sequence

from prompt_mfr_adapter import PromptMFRResources


class LexicalNormalizationPipeline:
    def __init__(
        self,
        detector,
        dictionary=None,
        llm=None,
        *,
        prompt_mfr_resources: PromptMFRResources | None = None,
        use_prompt_mfr_dictionary: bool = True,
        run_target_detection: bool = False,
        detection_max_new_tokens: int = 16,
        normalization_max_new_tokens: int = 64,
    ):
        self.detector = detector
        self.dictionary = dictionary
        self.llm = llm
        self.use_prompt_mfr_dictionary = use_prompt_mfr_dictionary
        self.run_target_detection = run_target_detection
        self.detection_max_new_tokens = detection_max_new_tokens
        self.normalization_max_new_tokens = normalization_max_new_tokens
        self.prompt_mfr_resources = (
            prompt_mfr_resources
            if prompt_mfr_resources is not None
            else PromptMFRResources()
        )

    def process_batch(
        self,
        batch_tokens: List[List[str]],
        gemma_flags: List[int],
        batch_langs: List[str],
    ) -> List[List[str]]:
        if len(batch_tokens) != len(gemma_flags) or len(batch_tokens) != len(batch_langs):
            raise ValueError(
                "batch_tokens, gemma_flags, and batch_langs must have the same length. "
                "gemma_flags must come from prediction.txt in validation row order."
            )

        all_corrected_sentences = []

        for tokens, g_flag, lang in zip(batch_tokens, gemma_flags, batch_langs):
            if self.use_prompt_mfr_dictionary:
                corrected_sentence = self._process_sentence_with_prompt_mfr(
                    tokens=tokens,
                    gemma_flag=g_flag,
                    lang=lang,
                )
            else:
                corrected_sentence = self._process_sentence_legacy(
                    tokens=tokens,
                    gemma_flag=g_flag,
                    lang=lang,
                )

            all_corrected_sentences.append(corrected_sentence)

        return all_corrected_sentences

    def _process_sentence_with_prompt_mfr(
        self,
        *,
        tokens: Sequence[str],
        gemma_flag: int,
        lang: str,
    ) -> list[str]:
        resources = self.prompt_mfr_resources
        raw_tokens = list(tokens)

        if not resources.supports_language(lang):
            return self._process_sentence_legacy(
                tokens=raw_tokens,
                gemma_flag=gemma_flag,
                lang=lang,
            )

        protected_indices = set(resources.find_protected_indices(raw_tokens))
        corrected_tokens = list(raw_tokens)

        xlmr_flags = self._predict_detector_flags(raw_tokens)
        detector_candidates = {
            idx for idx, flag in enumerate(xlmr_flags) if int(flag) == 1
        }
        candidate_indices = self._prediction_txt_candidates(
            raw_tokens=raw_tokens,
            gemma_flag=gemma_flag,
            detector_candidates=detector_candidates,
        )
        candidate_indices.difference_update(protected_indices)

        for idx in sorted(candidate_indices):
            raw_token = raw_tokens[idx]

            dict_result = resources.high_confidence_normalization(raw_token, lang)
            if dict_result is not None:
                corrected_tokens[idx] = dict_result
                continue

            if not self._llm_says_change(raw_tokens, idx, lang):
                continue

            normalized = self._llm_normalize_token(raw_tokens, idx, lang)
            if normalized is None:
                continue

            corrected_tokens[idx] = resources.safe_normalization_result(
                raw_token,
                normalized,
            )

        if len(corrected_tokens) != len(raw_tokens):
            return raw_tokens
        return corrected_tokens

    @staticmethod
    def _prediction_txt_candidates(
        *,
        raw_tokens: Sequence[str],
        gemma_flag: int,
        detector_candidates: set[int],
    ) -> set[int]:
        """Mirror colab_MultiLexNorm_HW11.ipynb prediction.txt behavior.

        prediction.txt is a sentence-level 0/1 file aligned with validation rows.
        If the sentence-level detector says 1 but XLM-R finds no token, the old
        Colab pipeline opened the whole sentence. Otherwise, it trusted XLM-R's
        token-level flags.
        """
        if int(gemma_flag) == 1 and not detector_candidates:
            return set(range(len(raw_tokens)))
        return set(detector_candidates)

    def _llm_says_change(self, tokens: Sequence[str], target_index: int, lang: str) -> bool:
        if not self.run_target_detection:
            return True
        if self.llm is None:
            return False

        prompt = self.prompt_mfr_resources.build_detection_prompt(tokens, target_index, lang)
        raw_output = self._generate(prompt, max_new_tokens=self.detection_max_new_tokens)
        label = self.prompt_mfr_resources.parse_detection_output(raw_output)
        return label == 1

    def _llm_normalize_token(
        self,
        tokens: Sequence[str],
        target_index: int,
        lang: str,
    ) -> str | None:
        if self.llm is None:
            return None

        prompt = self.prompt_mfr_resources.build_normalization_prompt(
            tokens,
            target_index,
            lang,
        )
        raw_output = self._generate(prompt, max_new_tokens=self.normalization_max_new_tokens)
        parsed = self.prompt_mfr_resources.parse_normalization_output(raw_output)
        if parsed is None:
            return None

        normalized = parsed.get("normalized")
        return normalized if isinstance(normalized, str) else None

    def _generate(self, prompt: str, *, max_new_tokens: int) -> str:
        if hasattr(self.llm, "generate"):
            return str(self.llm.generate(prompt, max_new_tokens=max_new_tokens))
        if hasattr(self.llm, "correct_from_prompt"):
            return str(self.llm.correct_from_prompt(prompt, max_new_tokens=max_new_tokens))
        raise AttributeError("LLM object must provide generate(prompt, max_new_tokens=...).")

    def _predict_detector_flags(self, tokens: Sequence[str]) -> list[int]:
        if self.detector is None:
            return [0] * len(tokens)

        try:
            flags = list(self.detector.predict(list(tokens)))
        except Exception:
            return [0] * len(tokens)

        if len(flags) != len(tokens):
            return [0] * len(tokens)

        return [1 if int(flag) == 1 else 0 for flag in flags]

    def _process_sentence_legacy(
        self,
        *,
        tokens: Sequence[str],
        gemma_flag: int,
        lang: str,
    ) -> list[str]:
        raw_tokens = list(tokens)
        xlmr_flags = self._predict_detector_flags(raw_tokens)

        if (
            self.dictionary is not None
            and int(gemma_flag) == 1
            and sum(xlmr_flags) == 0
        ):
            final_flags = [
                1 if self.dictionary.has_entry(token, lang) else 0
                for token in raw_tokens
            ]
        else:
            final_flags = xlmr_flags

        corrected_sentence: list[str] = []

        for idx, token in enumerate(raw_tokens):
            if final_flags[idx] == 0:
                corrected_sentence.append(token)
                continue

            dict_result = None
            if self.dictionary is not None:
                dict_result = self.dictionary.correct_if_confident(
                    token,
                    lang,
                    min_confidence=0.75,
                    min_count=2,
                    min_margin=0.50,
                )

            if dict_result is not None:
                corrected_sentence.append(dict_result)
                continue

            if self.llm is None or not hasattr(self.llm, "correct"):
                corrected_sentence.append(token)
                continue

            llm_result = self.llm.correct(token, raw_tokens, lang=lang)

            if len(llm_result) > max(len(token) * 3, 20):
                corrected_sentence.append(token)
            else:
                corrected_sentence.append(llm_result)

        if len(corrected_sentence) != len(raw_tokens):
            return raw_tokens
        return corrected_sentence
