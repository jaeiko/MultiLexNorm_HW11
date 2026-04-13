try:
    from utils import counting
except ImportError:
    print("utils.py 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

class MFRDictionary:
    """훈련 데이터의 빈도수(MFR)를 기반으로 비표준 단어를 즉시 교정하는 사전 모듈.
    
    Attributes:
        counts (dict): 훈련 데이터를 스캔하여 구축한 '노이즈 단어 -> 표준어' 매핑 딕셔너리.
    """
    
    def __init__(self, train_data) -> None:
        """MFRDictionary 클래스를 초기화하고 룩업 테이블(사전)을 구축합니다.
        
        Args:
            train_data: utils.py의 counting 함수가 처리할 수 있는 형태의 훈련 데이터 리스트.
        """
        self.counts = counting(train_data)
        
    def correct(self, word: str):
        """단어가 사전에 존재하면 가장 빈도가 높은 정규화 단어를 반환하고, 없으면 None을 반환합니다.
        
        Args:
            word (str): 교정이 필요한 비표준 노이즈 단어.
            
        Returns:
            str or None: 사전에 기반한 정규화 단어. 사전에 없으면 None.
        """
        if word in self.counts:
            return max(self.counts[word], key=self.counts[word].get)
        return None

if __name__ == "__main__":
    # =====================================================================
    # [데이터 담당자 📌] 
    # 아래 mock_train_data 위치에 직접 정제하신 17개국 통합 데이터셋
    # (또는 언어별 데이터셋)을 로드하여 넣어주시면 됩니다. 
    # 데이터 포맷은 {"raw": [...], "norm": [...]} 형태의 딕셔너리 리스트여야 합니다.
    # =====================================================================
    mock_train_data = [
        {"raw": ["u", "r", "funny"], "norm": ["you", "are", "funny"]},
        {"raw": ["u", "r", "cute"], "norm": ["you", "are", "cute"]}
    ]
    
    dictionary_module = MFRDictionary(mock_train_data)
    
    test_word_1 = "u"      
    test_word_2 = "bcause" 
    
    print(f"'{test_word_1}' 교정 결과: {dictionary_module.correct(test_word_1)}")
    print(f"'{test_word_2}' 교정 결과: {dictionary_module.correct(test_word_2)}")