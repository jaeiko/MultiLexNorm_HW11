import torch
import re
import warnings
from typing import List
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 불필요한 경고 메시지 숨김
warnings.filterwarnings('ignore')

class MultilingualCorrector:
    """수민님의 '언어별 프롬프트 분리' 전략이 적용된 고도화된 다국어 단어 교정 모듈."""
    
    def __init__(self, model_checkpoint: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        print(f"[System] {model_checkpoint} 모델을 4-bit로 로드합니다. (안전망 가동)")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
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
            tokenizer=self.tokenizer
        )

    def correct(self, noisy_word: str, context: List[str], lang: str | None = None) -> str:
        full_sentence = " ".join(context)
        lang_hint = f"Language code: {lang}\n" if lang else ""
        
        # 1. 공통 시스템 규칙 (Base Rules)
        system_rules = [
            "You are a highly accurate multilingual lexical normalization API.",
            "Analyze the context and normalize the 'Noisy word' into its standard form in the exact same language.",
            "\nCRITICAL RULES:",
            "1. OUTPUT ONLY THE NORMALIZED FORM. No explanations.",
            "2. DO NOT TRANSLATE to English. Maintain the original language.",
            "3. The output may contain spaces if the noisy token expands into a phrase.",
            "4. Preserve token-level alignment: return one string for the given noisy token."
        ]
        
        # 2. 언어별 동적 규칙 및 Few-shot 라우팅 (Routing Logic)
        specific_examples = []
        
        if lang == "sr":
            # 세르비아어: 문자 체계(라틴/키릴) 변환 금지 규칙 강화
            system_rules.append("5. DO NOT TRANSLITERATE SCRIPTS. If the input is in the Latin alphabet, the output MUST remain in the Latin alphabet. Do not convert to Cyrillic.")
            specific_examples = [
                {"role": "user", "content": "Language code: sr\nContext: 'da li je to ok'\nNoisy word: 'da'"},
                {"role": "assistant", "content": "da"}
            ]
        elif lang == "hr":
            # 크로아티아어: 과잉 교정 방지 규칙 강화
            system_rules.append("5. PREVENT OVER-CORRECTION. If the word is already a valid standard form, return it exactly as it is.")
            specific_examples = [
                {"role": "user", "content": "Language code: hr\nContext: 'kad dolazis'\nNoisy word: 'kad'"},
                {"role": "assistant", "content": "kad"}
            ]
        elif lang == "ja":
            specific_examples = [
                {"role": "user", "content": "Language code: ja\nContext: 'これマジでやば이'\nNoisy word: 'マジ'"},
                {"role": "assistant", "content": "本当に"}
            ]
        elif lang == "vi":
            specific_examples = [
                {"role": "user", "content": "Language code: vi\nContext: 'tôi ko biết'\nNoisy word: 'ko'"},
                {"role": "assistant", "content": "không"}
            ]
        else:
            # 영어 및 기타 언어: 구문 확장 예시 제공
            specific_examples = [
                {"role": "user", "content": "Language code: en\nContext: 'I lov u so much'\nNoisy word: 'lov'"},
                {"role": "assistant", "content": "love"},
                {"role": "user", "content": "Language code: en\nContext: 'I wanna go home'\nNoisy word: 'wanna'"},
                {"role": "assistant", "content": "want to"}
            ]

        # 3. 최종 메시지 구성
        messages = [
            {"role": "system", "content": "\n".join(system_rules)},
            *specific_examples,
            {"role": "user", "content": f"{lang_hint}Context: '{full_sentence}'\nNoisy word: '{noisy_word}'"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self.generator(
            prompt, 
            max_new_tokens=20, 
            return_full_text=False,
            do_sample=False, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        corrected_word = outputs[0]["generated_text"].strip()
        
        # 4. 다단계 조건부 안전망 (Safety Net)
        if "\n" in corrected_word:
            return noisy_word

        if len(corrected_word.split()) > 5:
            return noisy_word

        sys_symbols = ['*', '%', '#', '@', '"', "'", ':', '`', '(', ')']
        for sym in sys_symbols:
            if sym in corrected_word and sym not in noisy_word:
                return noisy_word
        
        cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
        if cjk_pattern.search(corrected_word) and not cjk_pattern.search(full_sentence):
            return noisy_word

        return corrected_word