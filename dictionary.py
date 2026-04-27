import json
try:
    from utils import counting
except ImportError:
    print("utils.py 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

class MFRDictionary:
    """훈련 데이터의 빈도수(MFR)를 기반으로 비표준 단어를 즉시 교정하는 사전 모듈.
    
    Attributes:
        counts (dict): 훈련 데이터를 스캔하여 구축한 '노이즈 단어 -> 표준어' 매핑 딕셔너리.
    """
    
    def __init__(self, json_file_path: str) -> None:
        """
        초기화 시점에 JSON 파일을 읽어 메모리에 캐싱합니다.
        가정된 JSON 구조: {"u": "you", "lov": "love", ...}
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            print(f"[System] {len(self.mapping)}개의 사전 데이터를 성공적으로 로드했습니다.")
        except Exception as e:
            print(f"[Error] 사전 데이터 로드 실패: {e}")
            self.mapping = {}
        
    def correct(self, word: str):
        """단어가 사전에 존재하면 가장 빈도가 높은 정규화 단어를 반환하고, 없으면 None을 반환합니다.
        
        Args:
            word (str): 교정이 필요한 비표준 노이즈 단어.
            
        Returns:
            str or None: 사전에 기반한 정규화 단어. 사전에 없으면 None.
        """
        return self.mapping.get(word, None)

if __name__ == "__main__":
    # 임시 테스트용 JSON 파일 생성
    test_json_path = "mock_dict.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump({"u": "you", "bcause": "because", "lov": "love"}, f)
    
    # 생성한 JSON 파일 경로를 입력하여 사전 모듈 초기화
    dictionary_module = MFRDictionary(test_json_path)
    
    test_word_1 = "u"      
    test_word_2 = "bcause" 
    
    print(f"'{test_word_1}' 교정 결과: {dictionary_module.correct(test_word_1)}")