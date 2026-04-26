from typing import List

class LAIBaseline:
    """아무런 교정을 수행하지 않고 원본을 반환하는 기본 베이스라인."""
    
    def __init__(self):
        print("[System] LAI Baseline 초기화 완료.")

    def predict(self, sentence_tokens: List[str]) -> List[str]:
        # 입력된 리스트를 변경 없이 그대로 반환
        return sentence_tokens