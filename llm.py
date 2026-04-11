from typing import List

# --- 관점 B: 향후 희진님이 작업할 실제 Llama-3 모델의 뼈대 ---
class LlamaCorrector:
    """
    [향후 구현 예정] Llama-3 기반의 문맥 맞춤형 단어 교정 모듈.
    희진님이 이 클래스 내부의 __init__과 correct 메서드에 실제 허깅페이스 모델 로드 및
    프롬프트 생성 로직을 채워 넣으시면 됩니다.
    """
    def __init__(self, model_checkpoint: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_checkpoint = model_checkpoint
        # TODO: 여기에 4비트 양자화(BitsAndBytesConfig) 및 Llama-3 모델 로드 코드 작성
        pass

    def correct(self, noisy_word: str, context: List[str]) -> str:
        # TODO: context를 기반으로 프롬프트를 구성하고, 모델(generate)을 호출하여 결과 반환
        pass


# --- 파이프라인 테스트를 위해 사용할 더미 모듈 ---
class DummyLLM:
    """
    [현재 테스트용] 파이프라인의 전체 흐름이 끊기지 않는지 검증하기 위한 임시 모듈.
    입력된 노이즈 단어에 단순히 꼬리표를 붙여서 반환합니다.
    """
    def __init__(self):
        print("[System] DummyLLM이 초기화되었습니다. (파이프라인 테스트 모드)")

    def correct(self, noisy_word: str, context: List[str]) -> str:
        """
        사전(Dictionary)에서 찾지 못한 단어를 문맥을 고려하여 교정합니다.
        
        Args:
            noisy_word (str): 교정이 필요한 비표준 단어.
            context (List[str]): 해당 단어가 포함된 전체 문장 토큰 리스트.
            
        Returns:
            str: 교정된 단어. (현재는 테스트를 위해 '_llm_fixed'를 붙여 반환)
        """
        # 실제로는 여기서 문맥(context)을 분석하지만, 현재는 더미 출력을 반환합니다.
        return f"{noisy_word}_llm_fixed"

# ==========================================
# 파이프라인 뼈대 검증을 위한 내부 단위 테스트
# ==========================================
if __name__ == "__main__":
    llm_module = DummyLLM()
    
    # 가상의 입력 데이터 세팅
    test_sentence = ["I", "lov", "u", "so", "much", "."]
    target_noisy_word = "lov"
    
    # LLM 모듈 추론 실행
    result = llm_module.correct(noisy_word=target_noisy_word, context=test_sentence)
    
    print(f"전체 문맥: {test_sentence}")
    print(f"'{target_noisy_word}' 교정 결과: {result}")