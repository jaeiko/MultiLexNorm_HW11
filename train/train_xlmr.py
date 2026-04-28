# 이 스크립트는 메모리 문제로 인해 Google Colab 환경에서 실행하는 것을 권장함
import os
import sys
import pandas as pd
from datasets import Dataset, DatasetDict

# 1. 상위 폴더 경로 추가 (detection.py 임포트용)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from detection import AnomalyDetector

def run_final_xlmr_training():
    print("[1] 탐지 모델 초기화 중...")
    detector = AnomalyDetector("xlm-roberta-base")
    
    # 2. 로컬 Parquet 파일 경로 설정
    # 파일이 'data' 폴더 안에 있다고 가정합니다.
    train_path = os.path.join(parent_dir, "data", "train-00000-of-00001.parquet")
    val_path = os.path.join(parent_dir, "data", "validation-00000-of-00001.parquet")

    print("[2] 로컬 Parquet 데이터 로드 중...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # pandas 데이터프레임을 허깅페이스 Dataset 객체로 변환
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # 3. 데이터 전처리 함수
    def preprocess_function(examples):
        all_tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        # batched=True 설정으로 인해 examples["raw"]와 examples["norm"]은 리스트의 리스트 형태입니다.
        for raw, norm in zip(examples["raw"], examples["norm"]):
            # prepare_data 함수 내부에서 tokenizer 호출 시 return_tensors="pt"를 쓰지 않으므로
            # 결과값은 이미 순수 파이썬 리스트 형태입니다.
            tokenized_inputs, aligned_labels = detector.prepare_data(raw, norm)
            
            # tokenized_inputs는 {'input_ids': [...], 'attention_mask': [...]} 형태의 딕셔너리입니다.
            # 리스트이므로 .squeeze() 없이 바로 append 합니다.
            all_tokenized_inputs["input_ids"].append(tokenized_inputs["input_ids"])
            all_tokenized_inputs["attention_mask"].append(tokenized_inputs["attention_mask"])
            all_tokenized_inputs["labels"].append(aligned_labels)
            
        return all_tokenized_inputs

    print("[3] 17개국 데이터 전처리 시작 (토큰화 및 레이블 정렬)...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    # 4. 모델 저장 경로 설정
    local_output_dir = os.path.join(parent_dir, "models", "xlmr_finetuned_final")
    
    print(f"[4] 로컬 학습 시작! (저장 위치: {local_output_dir})")
    # 우리가 수정한 detection.py의 train_model 호출
    detector.train_model(
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        output_dir=local_output_dir
    )
    print("학습이 완료되었습니다!")

if __name__ == "__main__":
    run_final_xlmr_training()