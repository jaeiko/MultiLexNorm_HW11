import sys
import os
from typing import List

class MFRBaseline:
    """훈련 데이터의 빈도수(MFR) 기반 통계적 교정 베이스라인."""
    
    def __init__(self, train_data) -> None:
        # 1. 현재 파일(MFR_baseline.py)의 절대 경로를 찾음
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 한 단계 위인 부모 폴더의 경로를 계산함
        parent_dir = os.path.dirname(current_dir)
        
        # 3. 파이썬이 모듈을 찾을 때 부모 폴더도 검색하도록 경로 추가
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)

        try:
            from utils import counting
            self.counts = counting(train_data)
            print("[System] MFR Baseline 초기화 완료 (빈도수 사전 구축).")
        except ImportError:
            print(f"[Error] {parent_dir} 에서 utils.py를 불러올 수 없습니다.")

    def predict(self, sentence_tokens: List[str]) -> List[str]:
        corrected_tokens = []
        for word in sentence_tokens:
            if word in self.counts:
                best_correction = max(self.counts[word], key=self.counts[word].get)
                corrected_tokens.append(best_correction)
            else:
                corrected_tokens.append(word)
        return corrected_tokens