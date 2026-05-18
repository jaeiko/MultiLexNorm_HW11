from __future__ import annotations
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
        mfr_adapter,  # [핵심] 외부에서 만능 어댑터 객체를 주입받습니다.
        model_checkpoint: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        load_in_4bit: bool = True,
        use_chat_template: bool = True,
    ) -> None:
        self.mfr_adapter = mfr_adapter
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

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        model_input = self._format_prompt(prompt)
        outputs = self.generator(
            model_input,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return str(outputs[0]["generated_text"]).strip()

    def correct(self, noisy_word: str, context: List[str], lang: str | None = None, target_index: int | None = None) -> str:
        if target_index is None:
            try:
                target_index = context.index(noisy_word)
            except ValueError:
                target_index = 0

        lang_code = lang or "en"

        try:
            # 1. 어댑터에게 프롬프트 생성을 완벽하게 위임 (언어별 퓨샷 자동 매핑)
            prompt = self.mfr_adapter.build_normalization_prompt(context, target_index, lang_code)
        except Exception as e:
            print(f"[Warning] 프롬프트 생성 실패 ({lang_code}): {e}")
            return noisy_word

        # 2. LLM 생성
        raw_output = self.generate(prompt, max_new_tokens=64)

        # 3. 어댑터에게 JSON 파싱 및 안전 필터링 위임
        parsed_json = self.mfr_adapter.parse_normalization_output(raw_output)
        
        if parsed_json and "normalized" in parsed_json:
            corrected_word = str(parsed_json["normalized"])
        else:
            corrected_word = noisy_word

        # 이모지, 기호 등이 망가지지 않도록 최종 안전 보호
        final_word = self.mfr_adapter.safe_normalization_result(noisy_word, corrected_word)
        return final_word

    def _format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
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
