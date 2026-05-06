import torch
from typing import List
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LlamaCorrector:
    """프롬프트 엔지니어링 기반의 Llama-3 단어 교정 모듈.
    
    파이프라인의 최종 단계에서, 사전에 없는 난해한 노이즈 단어를
    주변 문맥(Context)을 활용하여 LLM의 추론 능력으로 정규화한다.
    """
    
    def __init__(self, model_checkpoint: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> None:
        print(f"[System] {model_checkpoint} 모델을 4-bit 양자화로 로드합니다. (VRAM 절약)")
        
        # T4 GPU 메모리 초과를 막기 위한 4-bit 양자화 설정 (필수)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            device_map="auto",
            quantization_config=quantization_config,
        )
        
        self.generator = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def correct(self, noisy_word: str, context: List[str]) -> str:
        full_sentence = " ".join(context)
        
        prompt = f"""You are a lexical normalization expert.
Read the following sentence and normalize the noisy word into a standard word.
Provide ONLY the corrected word as your answer, with no additional explanation.

Sentence: "{full_sentence}"
Noisy word: "{noisy_word}"
Corrected word: """

        outputs = self.generator(
            prompt, 
            max_new_tokens=10, 
            return_full_text=False,
            temperature=0.1,
            do_sample=True, # temperature를 사용할 때 필수
            pad_token_id=self.tokenizer.eos_token_id # 경고 메시지 방지
        )
        
        corrected_word = outputs[0]["generated_text"].strip()
        return corrected_word