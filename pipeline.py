from __future__ import annotations

from typing import List


class LexicalNormalizationPipeline:
    def __init__(self, detector, dictionary, llm):
        self.detector = detector
        self.dictionary = dictionary
        self.llm = llm

    def process_batch(
        self,
        batch_tokens: List[List[str]],
        gemma_flags: List[int],
        batch_langs: List[str],
    ) -> List[List[str]]:
        all_corrected_sentences = []

        for tokens, g_flag, lang in zip(batch_tokens, gemma_flags, batch_langs):
            xlmr_flags = self.detector.predict(tokens)

            if len(xlmr_flags) != len(tokens):
                xlmr_flags = [0] * len(tokens)

            if g_flag == 1 and sum(xlmr_flags) == 0:
                final_flags = [
                    1 if self.dictionary.has_entry(token, lang) else 0
                    for token in tokens
                ]
            else:
                final_flags = xlmr_flags

            corrected_sentence = []

            for idx, token in enumerate(tokens):
                if final_flags[idx] == 0:
                    corrected_sentence.append(token)
                    continue

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

                llm_result = self.llm.correct(token, tokens, lang=lang)

                if len(llm_result) > max(len(token) * 3, 20):
                    corrected_sentence.append(token)
                else:
                    corrected_sentence.append(llm_result)

            if len(corrected_sentence) != len(tokens):
                corrected_sentence = list(tokens)

            all_corrected_sentences.append(corrected_sentence)

        return all_corrected_sentences
