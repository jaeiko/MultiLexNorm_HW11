# MultiLexNorm_HW11

MultiLexNorm 2026 lexical normalization 과제용 팀 프로젝트 레포.  
기본 MFR baseline, full pipeline, LLM 기반 탐지모델, 실험 결과 분석 코드를 함께 관리합니다.
기존 README.md파일은 baseline폴더로 옮겼습니다

## 폴더 구조

```text
MultiLexNorm_HW11/
├─ pipeline.py                  # full pipeline 실행 진입점
├─ detection.py                 # 탐지 단계 코드
├─ dictionary.py                # dictionary 기반 후보/보정 로직
├─ llm.py                       # LLM 수정 모델 연동 코드
├─ utils.py                     # 공통 유틸, 평가 함수
├─ baseline/                    # 기본 baseline 관련 파일
├─ multilexnorm2026-dataset/    # train/validation/test parquet 데이터셋
├─ 탐지모델/                    # 탐지모델 개발 공간
│  ├─ LLM-base탐지모델/          # Gemma/Ollama 기반 LLM 탐지 실험 코드
│  └─ naive_detector/           # naive 탐지 baseline
├─ prompt_mfr_dictionary/       # 수민님께 받은 새 prompt + MFR dictionary 패키지
├─ notebooks/                   # Colab / 데모 notebook
├─ tools/analysis/              # 분석 스크립트
├─ reports/                     # validation 실험 요약/분석 결과
├─ outputs/                     # pipeline 출력, 제출용 예측 결과
└─ bin/                         # gitignore
```

## 주로 보면 되는 곳


- 수민님께 받은 prompt와 MFR dictionary는 `prompt_mfr_dictionary/`에 있습니다.
- 실험 결과 요약은 `reports/`, 실제 예측 출력은 `outputs/`에 둡니다.
- 임시 코드나 한 번 쓰고 버리는 스크립트는 `bin/`에 두고 push 대상에서 제외합니다.

## 실행 환경

```bash
pip install -r requirements.txt
```

