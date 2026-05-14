import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
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
        """[모델 담당자용] 서브워드 분할로 인해 틀어진 토큰과 레이블의 길이를 동기화한다.

        첫 번째 서브워드에만 원본 레이블을 할당하고 나머지는 무시 인덱스인 -100을 할당한다.

        Args:
            labels (List[int]): 원본 단어 단위의 정답 레이블 리스트.
            word_ids (List[int]): 토크나이저가 반환한 서브워드들의 원본 단어 매핑 ID 리스트.

        Returns:
            List[int]: -100이 적절히 패딩된 새로운 레이블 리스트.
        """
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            else:
                new_labels.append(-100)
        return new_labels

    def prepare_data(self, raw_tokens: List[str], norm_tokens: List[str]) -> Tuple[Dict[str, Any], List[int]]:
        """[모델 담당자용] 원시 텍스트와 정규화 텍스트를 비교하여 모델 학습용 데이터 구조로 변환한다.

        Args:
            raw_tokens (List[str]): 사용자가 입력한 원본 텍스트의 토큰 리스트.
            norm_tokens (List[str]): 교정이 완료된 정답 텍스트의 토큰 리스트.

        Returns:
            Tuple[Dict[str, Any], List[int]]: 토큰화된 입력 텐서 딕셔너리 및 정렬된 레이블 리스트.
        """
        labels = [0 if r == n else 1 for r, n in zip(raw_tokens, norm_tokens)]
        tokenized_inputs = self.tokenizer(raw_tokens, is_split_into_words=True, truncation=True)
        word_ids = tokenized_inputs.word_ids()
        aligned_labels = self.align_labels_with_tokens(labels, word_ids)
        return tokenized_inputs, aligned_labels

    def train_model(self, train_dataset, eval_dataset, output_dir: str = "./detection_model"):
        """ 정제된 데이터셋을 이용하여 모델의 가중치를 파인튜닝한다.

        Args:
            train_dataset: 전처리가 완료된 학습용 Hugging Face Dataset 객체.
            eval_dataset: 전처리가 완료된 검증용 Hugging Face Dataset 객체.
            output_dir (str): 학습된 모델 가중치가 저장될 경로.
        """
        # 1. 메모리 최적화를 위한 동적 패딩 콜레이터 설정
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # 2. 훈련 하이퍼파라미터 설정 (Learning Rate, Epoch, Batch Size)
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch", # 매 에폭마다 가중치 저장
            logging_steps=50,
        )

        # 3. Trainer 객체 초기화 및 훈련 시작
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset, # prepare_data로 전처리된 HuggingFace Dataset 객체
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator
        )
        
        trainer.train()
        trainer.save_model(output_dir)

    def predict_proba(self, sentence_tokens: List[str]) -> List[float]:
        """[파이프라인 추론용] 문장 내 각 단어가 노이즈(비표준어)일 확률(0.0~1.0)을 반환한다.

        Args:
            sentence_tokens (List[str]): 띄어쓰기 단위로 분리된 원본 텍스트 리스트.

        Returns:
            List[float]: 단어별 노이즈(클래스 1) 예측 확률 리스트
        """
        tokenized_inputs = self.tokenizer(sentence_tokens, is_split_into_words=True, return_tensors="pt")
        word_ids = tokenized_inputs.word_ids()
        tokenized_inputs = {k: v.to(self.model.device) for k, v in tokenized_inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)

        # 1. argmax 대신 softmax를 적용하여 0.0 ~ 1.0 사이의 확률 분포 획득
        probs = F.softmax(outputs.logits, dim=-1).squeeze()

        if probs.dim() == 1:
            probs = probs.unsqueeze(0)

        # 2. 클래스 1(비표준어/노이즈)의 확률만 추출
        noise_probs = probs[:, 1].tolist()

        # 3. 서브워드 확률을 원본 단어 단위로 병합
        word_probabilities = []
        current_word = None
        
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != current_word:
                word_probabilities.append(noise_probs[idx]) # 첫 번째 서브워드의 확률을 해당 단어의 확률로 채택
                current_word = word_id
                
        return word_probabilities