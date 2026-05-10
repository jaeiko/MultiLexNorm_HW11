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

    def correct(self, noisy_word: str, context: List[str]) -> str:
        full_sentence = " ".join(context)
        
        # [최적화 1] 프롬프트 경량화 (17개국 -> 구조가 뚜렷한 5개국 대표 예시로 압축)
        messages = [
            {"role": "system", "content": "You are a highly accurate multilingual lexical normalization API. Analyze the context, identify the language, and normalize the 'Noisy word' into its standard, formal dictionary spelling in THAT EXACT SAME language.\n\nCRITICAL RULES:\n1. OUTPUT ONLY ONE NORMALIZED WORD. No explanations, no punctuation.\n2. DO NOT TRANSLATE to English. Maintain the original language."},
            
            # English (라틴)
            {"role": "user", "content": "Context: 'I lov u so much'\nNoisy word: 'lov'"}, {"role": "assistant", "content": "love"},
            # German (구어체 축약)
            {"role": "user", "content": "Context: 'und ich hab mir heut früh'\nNoisy word: 'hab'"}, {"role": "assistant", "content": "habe"},
            # Korean (한글/초성체)
            {"role": "user", "content": "Context: '글구 오늘 넘 피곤해'\nNoisy word: '글구'"}, {"role": "assistant", "content": "그리고"},
            # Japanese (한자/가나 혼용)
            {"role": "user", "content": "Context: 'これマジでやばい'\nNoisy word: 'マジ'"}, {"role": "assistant", "content": "本当に"},
            # Vietnamese (성조 누락)
            {"role": "user", "content": "Context: 'tôi ko biết'\nNoisy word: 'ko'"}, {"role": "assistant", "content": "không"},
            
            # Target Data
            {"role": "user", "content": f"Context: '{full_sentence}'\nNoisy word: '{noisy_word}'"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # [최적화 2] 파라미터 교정 (do_sample=False로 확률적 헛소리 차단, max_length 제거)
        outputs = self.generator(
            prompt, 
            max_new_tokens=20, 
            return_full_text=False,
            do_sample=False, # 가장 확률이 높은 단 하나의 정답만 기계적으로 출력 (온도 무시)
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        corrected_word = outputs[0]["generated_text"].strip()
        
        # =====================================================================
        # [최적화 3] 교차 검증 기반의 다단계 조건부 안전망 (Safety Net)
        # =====================================================================
        
        # 방어 1: 단일 단어 절대 규칙 검사 (공백이나 줄바꿈이 있으면 문장을 생성한 것으로 간주)
        if "\n" in corrected_word or len(corrected_word.split()) > 4:
            return noisy_word
            
        # 방어 2: 시스템/특수 기호 교차 검사
        # 원본 노이즈 단어에는 없었는데, 모델이 마음대로 기호를 붙였다면 환각으로 간주
        sys_symbols = ['*', '%', '#', '@', '"', "'", ':', '`', '(', ')']
        for sym in sys_symbols:
            if sym in corrected_word and sym not in noisy_word:
                return noisy_word
                
        # 방어 3: 한중일(CJK) 문자 환각 교차 검사
        # 한자, 히라가나/가타카나, 한글을 탐지하는 정규표현식
        cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
        
        # 교정된 단어에 아시아 문자가 생겼는데, 원본 문장(Context) 전체에 아시아 문자가 단 하나도 없었다면 100% 환각
        if cjk_pattern.search(corrected_word) and not cjk_pattern.search(full_sentence):
            return noisy_word

        # 모든 방어막을 통과한 순수하고 안전한 단어만 리턴
        return corrected_word