from __future__ import annotations

import re
import warnings
from typing import List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

warnings.filterwarnings("ignore")

class MultilingualCorrector:
    def __init__(
        self,
        model_checkpoint: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        load_in_4bit: bool = True,
        use_chat_template: bool = True,
    ) -> None:
        self.model_checkpoint = model_checkpoint
        self.use_chat_template = use_chat_template

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            device_map={"": 0},
            quantization_config=quantization_config,
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 64) -> str:
        model_input = self._format_prompt(prompt)
        outputs = self.generator(
            model_input,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return str(outputs[0]["generated_text"]).strip()

    def correct_from_prompt(self, prompt: str, *, max_new_tokens: int = 64) -> str:
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def correct(self, noisy_word: str, context: List[str], lang: str | None = None) -> str:
        """
        [핵심 업데이트] Qwen의 과잉 교정을 막기 위한 '보수적 편집자' 프롬프트와 
        언어별 동적 퓨샷(Dynamic Few-shot)을 주입합니다.
        """
        full_sentence = " ".join(context)
        
        # 1. 강력한 통제 규칙 (System Instruction)
        system_instruction = (
            "You are a STRICT and CONSERVATIVE Lexical Normalizer for social media text.\n"
            "Your ONLY task is to normalize severe internet slang, abbreviations, or typos into standard words.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT touch standard words: If the target token is already a valid word, output it EXACTLY as is (e.g., 'went' MUST remain 'went', do not change tense to 'gone').\n"
            "2. DO NOT touch punctuation, symbols, or emojis: If the token is '.', '?', '!', 'ㅋㅋ', output it EXACTLY as is.\n"
            "3. DO NOT translate: Keep the output in the original language.\n"
            "Output ONLY the final normalized token. No explanations, no quotes."
        )

        # 2. 언어별 퓨샷 예시 (정상 단어 유지 예시를 비표준어 교정 예시와 섞어 배치)
        few_shot_examples = {
            "en": (
                "Example 1 (Slang): Input: 'im', Output: i'm\n"
                "Example 2 (Standard - DO NOT TOUCH): Input: 'went', Output: went\n"
                "Example 3 (Punctuation - DO NOT TOUCH): Input: '.', Output: .\n"
                "Example 4 (Abbreviation): Input: 'tmrw', Output: tomorrow\n"
                "Example 5 (Standard - DO NOT TOUCH): Input: 'apple', Output: apple"
            ),
            "ko": (
                "Example 1 (Slang): Input: '글구', Output: 그리고\n"
                "Example 2 (Standard - DO NOT TOUCH): Input: '학교에', Output: 학교에\n"
                "Example 3 (Punctuation - DO NOT TOUCH): Input: '!!', Output: !!\n"
                "Example 4 (Consonants - DO NOT TOUCH): Input: 'ㅋㅋ', Output: ㅋㅋ\n"
                "Example 5 (Standard - DO NOT TOUCH): Input: '갔다', Output: 갔다"
            ),
            "nl": (
                "Example 1 (Standard - DO NOT TOUCH): Input: 'vrouw', Output: vrouw\n"
                "Example 2 (Slang): Input: 'ff', Output: even\n"
                "Example 3 (Punctuation - DO NOT TOUCH): Input: ':p', Output: :p"
            ),
            "ja": (
                "Example 1 (Standard - DO NOT TOUCH): Input: '行く', Output: 行く\n"
                "Example 2 (Slang): Input: 'りょ', Output: 了解\n"
                "Example 3 (Punctuation - DO NOT TOUCH): Input: '。', Output: 。"
            ),
            "id": (
                "Example 1 (Standard - DO NOT TOUCH): Input: 'makan', Output: makan\n"
                "Example 2 (Slang): Input: 'yg', Output: yang\n"
                "Example 3 (Punctuation - DO NOT TOUCH): Input: ',', Output: ,"
            ),
            # 예시가 없는 언어를 위한 기본 방어 예시
            "default": (
                "Example 1 (Standard - DO NOT TOUCH): Input: 'apple', Output: apple\n"
                "Example 2 (Slang): Input: 'u', Output: you\n"
                "Example 3 (Punctuation - DO NOT TOUCH): Input: '?', Output: ?\n"
                "Example 4 (Standard - DO NOT TOUCH): Input: 'went', Output: went"
            )
        }

        lang_code = lang if lang in few_shot_examples else "default"
        examples = few_shot_examples[lang_code]

        # 3. 최종 프롬프트 조립
        prompt = f"{system_instruction}\n\n[Language: {lang} Examples]\n{examples}\n\nContext: {full_sentence}\nTarget token: {noisy_word}\nAnswer:"

        corrected_word = self.generate(prompt, max_new_tokens=32)
        return self._legacy_safety_filter(noisy_word, full_sentence, corrected_word)

    def _format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    @staticmethod
    def _legacy_safety_filter(noisy_word: str, full_sentence: str, corrected_word: str) -> str:
        corrected_word = corrected_word.strip()

        if "\n" in corrected_word:
            return noisy_word

        if len(corrected_word.split()) > 5:
            return noisy_word

        protected_symbols = ["*", "%", "#", "@", '"', "'", ":", "`", "(", ")"]
        for symbol in protected_symbols:
            if symbol in corrected_word and symbol not in noisy_word:
                return noisy_word

        cjk_pattern = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
        if cjk_pattern.search(corrected_word) and not cjk_pattern.search(full_sentence):
            return noisy_word

        return corrected_word