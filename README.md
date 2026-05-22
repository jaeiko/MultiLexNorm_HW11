# MultiLexNorm2026

MultiLexNorm 2026 다국어 구어체 텍스트 어휘 정규화(Lexical Normalization) 과제용 팀 프로젝트 레포지토리입니다.  
본 프로젝트는 **[0단계: Trigram] -> [1단계: MFR 사전] -> [2단계: XLM-R 탐지] -> [3단계: LLM 보정]**으로 이어지는 4단계 파이프라인을 효율적으로 연동하고 성능을 극대화하는 것을 목표로 합니다.

---

## 📂 최신 폴더 및 파일 구조

최근 리팩토링 및 공식 평가 패키지 통합에 따른 프로젝트 구조입니다.

```text
MultiLexNorm2026/
├── baseline/
│   ├── LAI_baseline.py          # LAI 기반 베이스라인 모델
│   ├── MFR_baseline.py          # MFR 사전 기반 베이스라인 모델
│   ├── ByT5_baseline.py         # ByT5 생성 모델 베이스라인
│   ├── test_baselines.py        # 베이스라인 단위 테스트
│   └── evaluate_all.py          # [Dashboard] 전체 모델 공식 메트릭 성능 집계 대시보드
├── multilexnorm2026-dataset/
│   ├── dataset_12lang/          # 12개 국어 다국어 Parquet 데이터셋 (신규 경로)
│   └── dataset_17lang/          # 17개 국어 다국어 Parquet 데이터셋 (신규 경로)
├── multilexnorm_eval_package/
│   └── multilexnorm_evaluator.py # [공식 평가 패키지] 수정 절대 금지 (공식 수학 연산식 탑재)
├── prompt_mfr_dictionary/       # 다국어 프롬프트 템플릿 및 언어별 MFR 규칙 자료 리소스
├── paths_config.py              # [중앙 집중 경로 관리] 로컬/Colab 환경 자동 경로 바인딩 모듈
├── evaluation.py                # [통합 평가 브릿지] 공식 평가 패키지와 기존 메트릭 구조 완벽 연동
├── detection.py                 # 2단계: XLM-R 기반 토큰 단위 비정규화 탐지 모듈
├── normalization_fewshot.py     # 3단계: LLM 기반 다국어 퓨샷(Few-shot) 보정 모듈
├── trigram_predictor.py         # 0단계: 문맥 기반 트라이그램(Trigram) 사전 필터링 모델
├── smart_guard_mfr_v2.py        # 1단계: MFR 사전 기반 필터링 및 백업 전략 로직
├── prompt_mfr_adapter.py        # MFR 딕셔너리 및 프롬프트 연동 어댑터
├── run_mfr_xlmr_experiment.py   # MFR 사전 + XLM-R 탐지 결합 실험 실행 스크립트
├── llm_correct_local.py         # LLM 최종 보정 로컬 추론 및 파이프라인 연계 스크립트
├── build_dev_submissions.py     # 최종 제출(Submission) 포맷 빌드 스크립트
├── mine_hard_cases_dev.py       # 오류 분석 및 하드 케이스 필터링용 마이닝 도구
└── requirements.txt             # 프로젝트 주요 의존성 목록
```

---

## 🛠️ 핵심 설계 방식

1. **중앙 집중형 경로 설계 (`paths_config.py`)**:
   * 실행 환경(로컬 또는 Google Colab)을 자동으로 판단하여 필요한 디렉토리를 파이썬 검색 경로(`sys.path`)에 등록합니다. 
   * 개별 스크립트 내에서 경로를 하드코딩하거나 복사해 넣을 필요가 없습니다.
2. **공식 메트릭 일원화 (`evaluation.py`)**:
   * `multilexnorm_eval_package`의 공식 `multilexnorm_evaluator.py` 모듈과 내부 연산이 100% 매핑됩니다.
   * `precision`, `recall`, `f1`, `err` 등의 성능 메트릭을 공식 연산식 그대로 도출하며, 기존 호출 코드와의 호환성을 완벽히 보장합니다.

---

## 🚀 실행 및 실험 방법

### 1. 환경 설정
프로젝트 주요 의존성을 설치합니다.
```bash
pip install -r requirements.txt
```
*(실험 진행에 따라 PyTorch, Transformers, Accelerate 등의 딥러닝 패키지가 추가로 필요합니다.)*

### 2. 베이스라인 성능 대시보드 실행
공식 평가 패키지로 브릿징된 대시보드를 구동하여 현재 베이스라인들의 공식 성능(F1, ERR 등)을 실시간으로 집계해 비교합니다.
```bash
python baseline/evaluate_all.py
```

### 3. 주요 단계별 실험 실행
* **MFR + XLM-R 연동 실험**:
  ```bash
  python run_mfr_xlmr_experiment.py
  ```
* **LLM 로컬 보정 실험**:
  ```bash
  python llm_correct_local.py
  ```

---

## 📝 협업 가이드라인

* **브랜치 관리**: 리팩토링 및 기능 변경은 작업용 브랜치(`reducing_over_normalized`)에서 검증 후 PR(Pull Request)을 거쳐 병합해 주시기 바랍니다.
* **불필요한 파일 커밋 방지**: `.gitignore` 규칙에 의해 `__pycache__` 및 로컬 테스트성 임시 파일들은 자동으로 제외됩니다. 임시 스크립트나 용량이 큰 데이터 파일들은 `bin/` 폴더 또는 로컬 저장소에 두고 커밋 대상에서 제외해 주십시오.
