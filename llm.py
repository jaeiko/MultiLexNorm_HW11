import torch
import warnings
from typing import List
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 불필요한 경고 메시지 숨김
warnings.filterwarnings('ignore')

class MultilingualCorrector:
    """17개국 다국어 데이터 정규화에 최적화된 프롬프트 엔지니어링 기반 교정 모듈."""
    
    # 다국어 처리에 압도적인 성능을 보이는 Qwen2.5-7B를 기본 모델로 세팅
    def __init__(self, model_checkpoint: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        print(f"[System] {model_checkpoint} 모델을 4-bit로 로드합니다. (다국어 최적화)")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            device_map={"": 0}, # GPU 0번 강제 할당 (OOM 방지)
            quantization_config=quantization_config,
        )
        
        self.generator = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def correct(self, noisy_word: str, context: List[str]) -> str:
        full_sentence = " ".join(context)
        
        # [핵심] 17개국 다국어 정규화를 위한 엄격한 Chat Template 구성
        messages = [
            {"role": "system", "content": "You are a highly accurate multilingual lexical normalization API. Your task is to analyze the context, identify the exact language, and normalize the 'Noisy word' into its standard, formal dictionary spelling in THAT EXACT SAME language.\n\nCRITICAL RULES:\n1. OUTPUT ONLY THE NORMALIZED WORD. No explanations, no preamble, no punctuation.\n2. DO NOT TRANSLATE to English. You MUST maintain the original language.\n3. If it is a code-switched text, normalize it to the target language used in context."},
            
            # English (EN)
            {"role": "user", "content": "Context: 'I lov u so much'\nNoisy word: 'lov'"}, {"role": "assistant", "content": "love"},
            # Korean (KO)
            {"role": "user", "content": "Context: '글구 오늘 넘 피곤해'\nNoisy word: '글구'"}, {"role": "assistant", "content": "그리고"},
            # Japanese (JA)
            {"role": "user", "content": "Context: 'これマジでやばい'\nNoisy word: 'マジ'"}, {"role": "assistant", "content": "本当に"},
            # Thai (TH)
            {"role": "user", "content": "Context: 'ไปกินข้าวกันมั้ย'\nNoisy word: 'มั้ย'"}, {"role": "assistant", "content": "ไหม"},
            # Vietnamese (VI)
            {"role": "user", "content": "Context: 'tôi ko biết'\nNoisy word: 'ko'"}, {"role": "assistant", "content": "không"},
            # German (DE)
            {"role": "user", "content": "Context: 'und ich hab mir heut früh'\nNoisy word: 'hab'"}, {"role": "assistant", "content": "habe"},
            # Spanish (ES)
            {"role": "user", "content": "Context: 'hola q tal estas'\nNoisy word: 'q'"}, {"role": "assistant", "content": "qué"},
            # Danish (DA)
            {"role": "user", "content": "Context: 'jeg kommer os i morgen'\nNoisy word: 'os'"}, {"role": "assistant", "content": "også"},
            # Croatian (HR)
            {"role": "user", "content": "Context: 'ja nez šta da radim'\nNoisy word: 'nez'"}, {"role": "assistant", "content": "ne znam"},
            # Italian (IT)
            {"role": "user", "content": "Context: 'xk non mi rispondi'\nNoisy word: 'xk'"}, {"role": "assistant", "content": "perché"},
            # Dutch (NL)
            {"role": "user", "content": "Context: 'ik kom dr zo aan'\nNoisy word: 'dr'"}, {"role": "assistant", "content": "er"},
            # Slovenian (SL)
            {"role": "user", "content": "Context: 'zaka pa ne'\nNoisy word: 'zaka'"}, {"role": "assistant", "content": "zakaj"},
            # Serbian (SR)
            {"role": "user", "content": "Context: 'ja msm da je to ok'\nNoisy word: 'msm'"}, {"role": "assistant", "content": "mislim"},
            # Turkish (TR)
            {"role": "user", "content": "Context: 'slm nbr'\nNoisy word: 'slm'"}, {"role": "assistant", "content": "selam"},
            # Indonesian (ID)
            {"role": "user", "content": "Context: 'buku yg bagus'\nNoisy word: 'yg'"}, {"role": "assistant", "content": "yang"},
            # Indonesian-English (ID-EN)
            {"role": "user", "content": "Context: 'pls bantu aku'\nNoisy word: 'pls'"}, {"role": "assistant", "content": "please"},
            # Turkish-German (TR-DE)
            {"role": "user", "content": "Context: 'vllt gelirim'\nNoisy word: 'vllt'"}, {"role": "assistant", "content": "vielleicht"},
            
            # Target Data
            {"role": "user", "content": f"Context: '{full_sentence}'\nNoisy word: '{noisy_word}'"}
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
            temperature=0.01,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        corrected_word = outputs[0]["generated_text"].strip()
        return corrected_word