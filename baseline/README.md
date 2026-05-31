# MultiLexNorm2026 Baseline

이 폴더는 MultiLexNorm2026 과제에서 사용된 baseline 모델 패키지를 포함합니다.
각 하위 폴더는 validation 결과와 예측 코드를 함께 제공합니다.

## 구성

- `smart_guarded_mfr_v1_guard_v1/`
  - MFR 기반 언어 인식 모델
  - Smart Guard v1 규칙 기반 보호기
  - validation 결과: Overall ERR 46.12, Accuracy 94.09

- `smart_guarded_mfr_v2_pth_0.8/`
  - MFR 기반 언어 인식 모델
  - Smart Guard v2 보호기 (protected_threshold = 0.8)
  - validation 결과: Overall ERR 49.19, Accuracy 94.15

## 하위 폴더 내용

각 하위 폴더에는 아래 파일이 포함됩니다.

- `mfr_stats.pkl.gz`: MFR 사전 및 통계
- `smart_guard_mfr_v*.py`: 예측 함수
- `config.json`: 모델/보호기 하이퍼파라미터
- `validation_predictions.json`: 검증 예측 결과
- `validation_overall.json`: 종합 검증 지표
- `validation_language_metrics.csv`: 언어별 검증 지표

## 사용 방법

1. 원하는 하위 폴더로 이동합니다.
2. Python 환경을 활성화합니다.
3. `mfr_stats.pkl.gz`를 로드하고 예측 함수를 호출합니다.

예시:

```python
import gzip
import pickle
from smart_guard_mfr_v1 import predict_smart_guarded_mfr_v1

with gzip.open("mfr_stats.pkl.gz", "rb") as f:
    obj = pickle.load(f)

mfr = obj["mfr"]
stats = obj["stats"]

predictions = predict_smart_guarded_mfr_v1(test_samples, mfr, stats)
```

## 주의

- 이 README는 `baseline` 하위에 포함된 baseline 모델 패키지 정보를 요약한 것입니다.
- 실제 데이터 로딩 및 전체 파이프라인은 루트 프로젝트의 다른 스크립트에서 관리될 수 있습니다.
