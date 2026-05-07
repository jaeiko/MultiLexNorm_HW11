import torch
from typing import List
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LlamaCorrector:
    """프롬프트 엔지니어링 기반의 Llama-3/Gemma 단어 교정 모듈."""
    
    def __init__(self, model_checkpoint: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> None:
        print(f"[System] {model_checkpoint} 모델을 로드합니다.")
        
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
        
        # [핵심] Instruct 모델에 최적화된 시스템 대화 구조(Chat Template)
        messages = [
            # 1. 시스템 역할 부여 (절대 규칙)
            {"role": "system", "content": "You are a strict lexical normalization API. Your ONLY job is to output the normalized word. NO explanations. NO preamble. NO punctuation."},
            
            # 2. 퓨샷(Few-shot) 예시 1
            {"role": "user", "content": "Context: 'yay , und ich hab mir heut früh'\nNoisy word: 'hab'"},
            {"role": "assistant", "content": "habe"},
            
            # 3. 퓨샷(Few-shot) 예시 2
            {"role": "user", "content": "Context: 'I lov u so much'\nNoisy word: 'lov'"},
            {"role": "assistant", "content": "love"},
            
            # 4. 실제 처리할 데이터
            {"role": "user", "content": f"Context: '{full_sentence}'\nNoisy word: '{noisy_word}'"}
        ]
        
        # Llama-3/Gemma가 인식하는 특수 토큰 구조(<|system|>, <|user|> 등)로 자동 변환
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self.generator(
            prompt, 
            max_new_tokens=20, # 토큰 공간을 넉넉히 주어도 대화 패턴에 의해 단어 1개만 출력 후 스스로 종료(EOS)함
            return_full_text=False,
            temperature=0.01,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        corrected_word = outputs[0]["generated_text"].strip()
        return corrected_word