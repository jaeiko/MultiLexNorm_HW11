import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple, Dict, Any

class AnomalyDetector:
    """소셜 미디어 텍스트 내 비표준(노이즈) 단어를 식별하는 이진 토큰 분류 모델 클래스.

    XLM-RoBERTa 모델을 기반으로 하며, 입력된 문장의 각 토큰이 정상(0)인지
    교정이 필요한 비표준 단어(1)인지 예측하기 위한 전처리와 추론을 수행한다.

    Attributes:
        tokenizer (AutoTokenizer): XLM-RoBERTa 사전 학습 토크나이저.
        model (AutoModelForTokenClassification): 2개의 클래스(정상, 노이즈)를 분류하는 모델.
    """

    def __init__(self, model_checkpoint: str = "xlm-roberta-base") -> None:
        """AnomalyDetector 클래스를 초기화한다.

        Args:
            model_checkpoint (str): 허깅페이스 허브에 등록된 모델 식별자. 기본값은 'xlm-roberta-base'.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2)

    def align_labels_with_tokens(self, labels: List[int], word_ids: List[int]) -> List[int]:
        """서브워드(Subword) 분할로 인해 틀어진 토큰과 레이블의 길이를 동기화한다.

        Hugging Face 토크나이저는 1개의 단어를 여러 개의 서브워드로 나눌 수 있다. 
        PyTorch 손실 함수(CrossEntropyLoss)가 중복된 서브워드에 대해 패널티를 
        부과하지 않도록, 첫 번째 서브워드에만 원본 레이블을 할당하고 나머지는 
        무시 인덱스인 -100을 할당한다.

        Args:
            labels (List[int]): 원본 단어 단위의 정답 레이블 리스트.
            word_ids (List[int]): 토크나이저가 반환한 서브워드들의 원본 단어 매핑 ID 리스트.

        Returns:
            List[int]: -100이 적절히 패딩(Padding)된 새로운 레이블 리스트.
        """
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # 새로운 단어의 첫 서브워드이므로 원래 레이블을 부여함
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            else:
                # 동일한 단어에서 파생된 두 번째 이후의 서브워드는 무시(-100) 처리함
                new_labels.append(-100)
        return new_labels

    def prepare_data(self, raw_tokens: List[str], norm_tokens: List[str]) -> Tuple[Dict[str, Any], List[int]]:
        """원시 텍스트와 정규화 텍스트를 비교하여 모델 학습용 데이터 구조로 변환한다.

        Args:
            raw_tokens (List[str]): 사용자가 입력한 원본 텍스트의 토큰 리스트.
            norm_tokens (List[str]): 교정이 완료된 정답 텍스트의 토큰 리스트.

        Returns:
            Tuple[Dict[str, Any], List[int]]: 
                - 토큰화된 입력 텐서 딕셔너리 (input_ids, attention_mask 등).
                - 서브워드 정렬이 완료된 타겟 레이블 리스트.
        """
        # List Comprehension을 사용하여 원본과 정답이 같으면 0, 다르면 1(노이즈) 할당
        labels = [0 if r == n else 1 for r, n in zip(raw_tokens, norm_tokens)]
        
        # 텍스트를 모델이 이해할 수 있는 텐서 형태로 변환
        tokenized_inputs = self.tokenizer(raw_tokens, is_split_into_words=True, truncation=True)
        word_ids = tokenized_inputs.word_ids()
        
        # 레이블 길이 동기화
        aligned_labels = self.align_labels_with_tokens(labels, word_ids)
        return tokenized_inputs, aligned_labels