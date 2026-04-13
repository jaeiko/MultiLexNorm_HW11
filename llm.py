from typing import List
from transformers import pipeline

class LlamaCorrector:
    """프롬프트 엔지니어링 기반의 Llama-3 단어 교정 모듈.
    
    파이프라인의 최종 단계에서, 사전에 없는 난해한 노이즈 단어를
    주변 문맥(Context)을 활용하여 LLM의 추론 능력으로 정규화한다.

    Attributes:
        generator: Hugging Face의 텍스트 생성 파이프라인 객체.
    """
    
    def __init__(self, model_checkpoint: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> None:
        """LlamaCorrector를 초기화하고 모델을 로드한다.

        Args:
            model_checkpoint (str): 사용할 LLM 모델의 경로 또는 이름.
        """
        print("[System] Llama-3 모델을 로드합니다. (이 작업은 다소 시간이 소요될 수 있습니다)")
        # 모델 담당자가 양자화 없이 기본 모델을 로드하는 설정
        self.generator = pipeline(
            "text-generation", 
            model=model_checkpoint, 
            device_map="auto"
        )

    def correct(self, noisy_word: str, context: List[str]) -> str:
        """[모델 담당자용] 프롬프트를 구성하여 모델에 전달하고 교정된 단어를 반환받는다.

        Args:
            noisy_word (str): 교정이 필요한 비표준 단어.
            context (List[str]): 해당 단어가 포함된 전체 문장 토큰 리스트.

        Returns:
            str: LLM이 교정한 표준 단어.
        """
        full_sentence = " ".join(context)
        
        # 모델 담당자는 이 프롬프트 텍스트를 실험을 통해 최적화(Prompt Engineering)합니다.
        prompt = f"""You are a lexical normalization expert.
Read the following sentence and normalize the noisy word into a standard word.
Provide ONLY the corrected word as your answer, with no additional explanation.

Sentence: "{full_sentence}"
Noisy word: "{noisy_word}"
Corrected word: """

        # 텍스트 생성 추론 실행
        outputs = self.generator(
            prompt, 
            max_new_tokens=10, 
            return_full_text=False,
            temperature=0.1 # 일관된 정답을 위해 창의성(온도)을 낮춤
        )
        
        # 결과값 텍스트 클렌징 (공백 및 줄바꿈 제거)
        corrected_word = outputs[0]["generated_text"].strip()
        return corrected_word

class DummyLLM:
    """[현재 테스트용] 파이프라인의 전체 흐름 검증을 위한 임시 모듈."""
    def __init__(self):
        print("[System] DummyLLM이 초기화되었습니다. (파이프라인 테스트 모드)")

    def correct(self, noisy_word: str, context: List[str]) -> str:
        return f"{noisy_word}_llm_fixed"

if __name__ == "__main__":
    llm_module = DummyLLM()
    test_sentence = ["I", "lov", "u", "so", "much", "."]
    target_noisy_word = "lov"
    result = llm_module.correct(noisy_word=target_noisy_word, context=test_sentence)
    
    print(f"전체 문맥: {test_sentence}")
    print(f"'{target_noisy_word}' 교정 결과: {result}")