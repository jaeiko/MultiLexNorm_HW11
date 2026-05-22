# MultiLexNorm2026 Evaluator Package

이 패키지는 official validation parquet와 internal held-out validation parquet를 같은 방식으로 분석/평가하기 위한 재사용 evaluator입니다.

## 포함 파일

- `multilexnorm_evaluator.py`: dataset statistics + evaluator 통합 CLI
- `requirements.txt`: parquet 로딩을 위한 dependency
- `outputs/stats/*.csv|json`: 현재 첨부 데이터셋 기준으로 미리 계산한 통계
- `outputs/results/*.csv|json`: LAI/MFR baseline 평가 결과 예시

## 권장 평가 기준

- `accuracy`: token-level accuracy
- `lai_accuracy`: Leave-As-Is baseline accuracy
- `err`: Error Reduction Rate
- `macro_err`: language-wise ERR의 단순 평균
- `detection_precision / recall / f1`: `pred != raw`를 수정 탐지로 보고 계산한 값
- `overnormalization_rate / undernormalization_rate`: FP/FN token 비율

## Official validation 통계 생성

```bash
python multilexnorm_evaluator.py stats \
  --gold_parquet /path/to/validation-00000-of-00001.parquet \
  --train_parquet /path/to/train-00000-of-00001.parquet \
  --dataset_name official_validation \
  --out_dir outputs/stats
```

## Internal 17-language validation 통계 생성

```bash
python multilexnorm_evaluator.py stats \
  --gold_parquet /path/to/dataset_17lang/data/validation-00000-of-00001.parquet \
  --train_parquet /path/to/dataset_17lang/data/train-00000-of-00001.parquet \
  --dataset_name internal_validation_17lang \
  --out_dir outputs/stats
```

## 예측 파일 평가

```bash
python multilexnorm_evaluator.py evaluate \
  --gold_parquet /path/to/validation-00000-of-00001.parquet \
  --pred_path /path/to/predictions.json \
  --model_name final_hybrid \
  --dataset_name official_validation \
  --out_dir outputs/results
```

## Built-in baseline 평가

```bash
# LAI
python multilexnorm_evaluator.py evaluate \
  --gold_parquet /path/to/validation-00000-of-00001.parquet \
  --baseline lai \
  --dataset_name official_validation \
  --out_dir outputs/results

# MFR
python multilexnorm_evaluator.py evaluate \
  --gold_parquet /path/to/validation-00000-of-00001.parquet \
  --train_parquet /path/to/train-00000-of-00001.parquet \
  --baseline mfr \
  --dataset_name official_validation \
  --out_dir outputs/results
```

## 주의사항

internal split을 평가할 때는 반드시 `dataset_17lang/data/train-00000-of-00001.parquet`만 MFR dictionary, ngram dictionary, prompt few-shot, detector 학습에 사용해야 합니다. `pseudo_validation_missing_langs.parquet` 또는 `dataset_17lang/data/validation-00000-of-00001.parquet`의 샘플을 학습 리소스에 섞으면 held-out 평가 누수가 발생합니다.
