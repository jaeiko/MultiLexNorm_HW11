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
    """Thin Hugging Face generation wrapper for lexical normalization prompts.

    The current full pipeline builds prompts outside this class with
    prompt_mfr_dictionary. This class is intentionally responsible only for
    loading the model and generating text.
    """

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
        """Generate raw model output for a fully-built prompt."""
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
        """Backward-compatible alias used by some notebooks."""
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def correct(self, noisy_word: str, context: List[str], lang: str | None = None) -> str:
        """Legacy one-token correction API.

        New code should prefer prompt_mfr_adapter + generate(). This method is
        kept so older notebook cells can still run.
        """
        full_sentence = " ".join(context)
        lang_hint = f"Language code: {lang}\n" if lang else ""
        prompt = f"""
You are a multilingual lexical normalization API.

Normalize only the noisy target token. Do not translate, paraphrase, or rewrite
the full sentence. Return only the normalized target token, with no explanation.

{lang_hint}Context: {full_sentence}
Noisy target token: {noisy_word}
Answer:
""".strip()
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
