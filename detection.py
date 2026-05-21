"""XLM-R 기반 비표준 토큰 탐지 (probability detection).

이 모듈은 우리 파이프라인의 표준 XLM-R wrapper다. 모든 XLM-R 호출은
이 모듈의 AnomalyDetector를 경유한다.

- predict_proba(tokens)        : 단일 문장, 토큰별 noise(class 1) 확률
- predict_proba_batch(...)     : 배치 추론 (동일 로직, 효율용)
- predict_labels(...)          : 확률 + lang-specific threshold → 0/1 라벨

threshold 정책: argmax(0/1) 대신 softmax 확률을 lang별 임계값과 비교한다.
임계값이 높을수록 detection이 보수적(LLM을 덜 부름)이다.
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from typing import List, Tuple, Dict, Any, Sequence


# lang-specific detection threshold.
#   ko/en/iden : mini-validation grid search 최적값
#   ja/tr/hr/sr/da/es/it/de : 팀원이 정한 언어학 그룹 기준
#       (A군 교착어/비라틴 0.85, B군 슬라브어 0.75, C군 라틴 고자원 0.60)
#   그 외(id/nl/sl/th/vi/trde) : DEFAULT_DETECTION_THRESHOLD
LANG_DETECTION_THRESHOLDS: Dict[str, float] = {
    # grid search 최적값
    "ko": 0.55, "en": 0.85, "iden": 0.90,
    # 팀원 언어학 그룹 — A군
    "ja": 0.85, "tr": 0.85, "zh": 0.85,
    # 팀원 언어학 그룹 — B군
    "hr": 0.75, "sr": 0.75, "da": 0.75, "bs": 0.75, "ru": 0.75,
    # 팀원 언어학 그룹 — C군
    "es": 0.60, "it": 0.60, "de": 0.60,
    # id, nl, sl, th, vi, trde → 미지정 → DEFAULT(0.60)
}
DEFAULT_DETECTION_THRESHOLD: float = 0.60


def get_lang_threshold(lang: str) -> float:
    """언어 코드에 대한 detection threshold를 반환한다 (미분류 시 기본값)."""
    return LANG_DETECTION_THRESHOLDS.get(str(lang), DEFAULT_DETECTION_THRESHOLD)


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
            model_checkpoint (str): 허깅페이스 허브 식별자 또는 로컬 가중치 폴더 경로.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, num_labels=2
        ).to(device)

    def align_labels_with_tokens(self, labels: List[int], word_ids: List[int]) -> List[int]:
        """[모델 담당자용] 서브워드 분할로 틀어진 토큰과 레이블의 길이를 동기화한다.

        첫 번째 서브워드에만 원본 레이블을 할당하고 나머지는 무시 인덱스 -100을 할당한다.
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
        """[모델 담당자용] 원시/정규화 텍스트를 비교하여 학습용 데이터 구조로 변환한다."""
        labels = [0 if r == n else 1 for r, n in zip(raw_tokens, norm_tokens)]
        tokenized_inputs = self.tokenizer(raw_tokens, is_split_into_words=True, truncation=True)
        word_ids = tokenized_inputs.word_ids()
        aligned_labels = self.align_labels_with_tokens(labels, word_ids)
        return tokenized_inputs, aligned_labels

    def train_model(self, train_dataset, eval_dataset, output_dir: str = "./detection_model"):
        """정제된 데이터셋으로 모델 가중치를 파인튜닝한다."""
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_steps=50,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model(output_dir)

    def predict_proba(self, sentence_tokens: List[str]) -> List[float]:
        """[추론용] 문장 내 각 단어가 노이즈(비표준어)일 확률(0.0~1.0)을 반환한다.

        Args:
            sentence_tokens (List[str]): 띄어쓰기 단위로 분리된 원본 텍스트 리스트.

        Returns:
            List[float]: 단어별 노이즈(클래스 1) 예측 확률 리스트.
        """
        tokenized_inputs = self.tokenizer(sentence_tokens, is_split_into_words=True, return_tensors="pt")
        word_ids = tokenized_inputs.word_ids()
        tokenized_inputs = {k: v.to(self.model.device) for k, v in tokenized_inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)

        # argmax 대신 softmax → 0.0~1.0 확률 분포
        probs = F.softmax(outputs.logits, dim=-1).squeeze()
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)

        # 클래스 1(비표준/노이즈)의 확률만 추출
        noise_probs = probs[:, 1].tolist()

        # 서브워드 확률을 원본 단어 단위로 병합 (첫 서브워드 채택)
        word_probabilities: List[float] = []
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != current_word:
                word_probabilities.append(noise_probs[idx])
                current_word = word_id
        return word_probabilities

    def predict_proba_batch(
        self,
        batch_tokens: Sequence[Sequence[str]],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> List[List[float]]:
        """predict_proba의 배치 버전. 로직(softmax → class 1 → 첫 서브워드)은 동일하다.

        Args:
            batch_tokens: 문장(토큰 리스트)들의 리스트.
            batch_size: 한 번에 forward할 문장 수.
            max_length: truncation 한계 (긴 문장 OOM 방지).

        Returns:
            List[List[float]]: 문장별 단어별 noise 확률.
        """
        self.model.eval()
        all_word_probs: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(batch_tokens), batch_size):
                chunk = [list(t) for t in batch_tokens[start:start + batch_size]]
                enc = self.tokenizer(
                    chunk, is_split_into_words=True, return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length,
                )
                enc_dev = {k: v.to(self.model.device) for k, v in enc.items()}
                logits = self.model(**enc_dev).logits
                noise = F.softmax(logits, dim=-1)[:, :, 1].cpu().tolist()  # (B, T)
                for b in range(len(chunk)):
                    word_ids = enc.word_ids(b)
                    word_probs: List[float] = []
                    current_word = None
                    for idx, word_id in enumerate(word_ids):
                        if word_id is None:
                            continue
                        if word_id != current_word:
                            word_probs.append(noise[b][idx])
                            current_word = word_id
                    # truncation으로 잘린 단어는 0.0 (정상)으로 패딩
                    while len(word_probs) < len(chunk[b]):
                        word_probs.append(0.0)
                    all_word_probs.append(word_probs)
        return all_word_probs

    def predict_labels(
        self,
        batch_tokens: Sequence[Sequence[str]],
        batch_langs: Sequence[str],
        batch_size: int = 16,
        max_length: int = 512,
        threshold: float | None = None,
    ) -> List[List[int]]:
        """배치 추론 후 threshold를 적용해 0/1 라벨을 반환한다.

        Args:
            threshold: None이면 lang-specific threshold(get_lang_threshold) 사용.
                       float이면 모든 lang에 그 고정값 적용.
                       threshold=0.5는 argmax(logits)와 동등하다.

        error_prob >= threshold → 1 (비표준 의심), 그 외 0.

        Returns:
            List[List[int]]: 문장별 단어별 0/1 라벨.
        """
        probs = self.predict_proba_batch(batch_tokens, batch_size=batch_size, max_length=max_length)
        labels: List[List[int]] = []
        for row_probs, lang in zip(probs, batch_langs):
            thr = get_lang_threshold(lang) if threshold is None else threshold
            labels.append([1 if p >= thr else 0 for p in row_probs])
        return labels
