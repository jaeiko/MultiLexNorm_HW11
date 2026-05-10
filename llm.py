import torch
import torch
import re
import warnings
from typing import List
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 불필요한 경고 메시지 숨김
warnings.filterwarnings('ignore')

class MultilingualCorrector:
    """프롬프트 경량화 및 조건부 안전망(Safety Net)이 적용된 다국어 단어 교정 모듈."""
    
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
        
        # 프롬프트 전면 적용
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly accurate multilingual lexical normalization API. "
                    "Analyze the context and normalize the 'Noisy word' into its standard form "
                    "in the exact same language.\n\n"
                    "CRITICAL RULES:\n"
                    "1. OUTPUT ONLY THE NORMALIZED FORM. No explanations.\n"
                    "2. DO NOT TRANSLATE to English. Maintain the original language.\n"
                    "3. The output may contain spaces if the noisy token expands into a phrase.\n"
                    "4. Preserve token-level alignment: return one string for the given noisy token."
                ),
            },
            # English
            {"role": "user", "content": "Language code: en\nContext: 'I lov u so much'\nNoisy word: 'lov'"},
            {"role": "assistant", "content": "love"},
            # German
            {"role": "user", "content": "Language code: de\nContext: 'und ich hab mir heut früh'\nNoisy word: 'hab'"},
            {"role": "assistant", "content": "habe"},
            # Korean
            {"role": "user", "content": "Language code: ko\nContext: '글구 오늘 넘 피곤해'\nNoisy word: '글구'"},
            {"role": "assistant", "content": "그리고"},
            # Japanese: phrase output example
            {"role": "user", "content": "Language code: ja\nContext: 'これマジでやばい'\nNoisy word: 'マジ'"},
            {"role": "assistant", "content": "本当に"},
            # Vietnamese
            {"role": "user", "content": "Language code: vi\nContext: 'tôi ko biết'\nNoisy word: 'ko'"},
            {"role": "assistant", "content": "không"},
            # Phrase expansion example
            {"role": "user", "content": "Language code: en\nContext: 'I wanna go home'\nNoisy word: 'wanna'"},
            {"role": "assistant", "content": "want to"},
            # Target Data
            {
                "role": "user",
                "content": f"{lang_hint}Context: '{full_sentence}'\nNoisy word: '{noisy_word}'",
            },
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
        
        # =====================================================================
        # [최적화 3] 교차 검증 기반의 다단계 조건부 안전망 (Safety Net)
        # =====================================================================
        
        # 방어 1: 줄바꿈은 설명문 생성 가능성이 높으므로 reject
        if "\n" in corrected_word:
            return noisy_word

        # 방어 2: 너무 긴 phrase는 환각/설명문 가능성이 높으므로 reject
        if len(corrected_word.split()) > 5:
            return noisy_word

        # 방어 3: 시스템/특수 기호 교차 검사
        sys_symbols = ['*', '%', '#', '@', '"', "'", ':', '`', '(', ')']
        for sym in sys_symbols:
            if sym in corrected_word and sym not in noisy_word:
                return noisy_word
        
        # 교정된 단어에 아시아 문자가 생겼는데, 원본 문장(Context) 전체에 아시아 문자가 단 하나도 없었다면 100% 환각
        cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
        if cjk_pattern.search(corrected_word) and not cjk_pattern.search(full_sentence):
            return noisy_word

        # 모든 방어막을 통과한 순수하고 안전한 단어만 리턴
        return corrected_word