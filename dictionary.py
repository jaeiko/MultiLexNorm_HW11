import sys

# 베이스라인의 utils.py에서 counting 함수를 가져옵니다.
try:
    from utils import counting
except ImportError:
    print("utils.py 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

class MFRDictionary:
    """훈련 데이터의 빈도수(MFR)를 기반으로 비표준 단어를 즉시 교정하는 사전 모듈.
    
    Detection 모듈에서 '노이즈(1)'로 판별된 단어 중, 과거 훈련 데이터에 
    자주 등장했던 단어는 이 모듈에서 O(1) 시간 복잡도로 빠르게 교정합니다.
    사전에 없는 난해한 단어는 None을 반환하여 다음 파이프라인(LLM)으로 넘깁니다.
    
    Attributes:
        counts (dict): 훈련 데이터를 스캔하여 구축한 '노이즈 단어 -> 표준어' 매핑 딕셔너리.
    """
    
    def __init__(self, train_data) -> None:
        """MFRDictionary 클래스를 초기화하고 룩업 테이블(사전)을 구축합니다.
        
        Args:
            train_data: utils.py의 counting 함수가 처리할 수 있는 형태의 훈련 데이터 리스트.
        """
        # utils.py의 counting 함수를 사용하여 단어별 교정 빈도수 사전을 메모리에 적재합니다.
        self.counts = counting(train_data)
        
    def correct(self, word: str):
        """단어가 사전에 존재하면 가장 빈도가 높은 정규화 단어를 반환하고, 없으면 None을 반환합니다.
        
        Args:
            word (str): 교정이 필요한 비표준 노이즈 단어.
            
        Returns:
            str or None: 사전에 기반한 정규화 단어. 사전에 없으면 None.
        """
        if word in self.counts:
            # 해당 노이즈 단어에 대해 가장 빈도수가 높은 교정 단어를 찾아 반환합니다.
            return max(self.counts[word], key=self.counts[word].get)
        
        # 사전에 없는 단어는 LLM이 문맥을 보고 추론해야 하므로 None을 반환합니다.
        return None

# ==========================================
# 파이프라인 뼈대 검증을 위한 내부 단위 테스트
# ==========================================
if __name__ == "__main__":
    # 임시 훈련 데이터 모형 (dict 포맷)
    mock_train_data = [
        {"raw": ["u", "r", "funny"], "norm": ["you", "are", "funny"]},
        {"raw": ["u", "r", "cute"], "norm": ["you", "are", "cute"]}
    ]
    
    # 1. 사전 모듈 초기화
    dictionary_module = MFRDictionary(mock_train_data)
    
    # 2. 테스트 단어 검증
    test_word_1 = "u"      # 사전에 있는 단어
    test_word_2 = "bcause" # 사전에 없는 단어
    
    print(f"'{test_word_1}' 교정 결과: {dictionary_module.correct(test_word_1)}") # 'you'가 출력되어야 함
    print(f"'{test_word_2}' 교정 결과: {dictionary_module.correct(test_word_2)}") # None이 출력되어야 함