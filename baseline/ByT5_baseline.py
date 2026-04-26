import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

class ByT5Baseline:
    """ByT5(ÚFAL) 기반의 인코더-디코더 정규화 베이스라인."""
    
    def __init__(self, model_checkpoint: str = "google/byt5-small") -> None:
        print(f"[System] {model_checkpoint} 모델을 로드합니다. (시간 소요 가능)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.model.eval() # 추론 모드로 설정
        
        # 논문 구현에 따른 특수 태그 설정 (예시)
        self.open_tag = "<w>"
        self.close_tag = "</w>"

    def predict(self, sentence_tokens: List[str]) -> List[str]:
        corrected_tokens = []
        
        with torch.no_grad():
            for word in sentence_tokens:
                # 입력 단어를 시작 및 종료 태그로 묶어서 독립적으로 처리
                prompt = f"{self.open_tag}{word}{self.close_tag}"
                
                # 바이트 단위 토큰화
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                # 모델 생성 추론
                outputs = self.model.generate(**inputs, max_new_tokens=20)
                
                # 결과 디코딩 및 리스트 추가
                corrected_word = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                corrected_tokens.append(corrected_word)
                
        return corrected_tokens