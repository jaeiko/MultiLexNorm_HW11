# MultiLexNorm 2026

homework 11th team submission codefile

다국어 비표준 어휘(줄임말·구어체·오탈자·방언 등)를 표준형으로 변환하는 **다국어 어휘 정규화(Lexical Normalization)** 인공지능개론 과제 코드입니다.
통계 사전(N-gram·MFR), 탐지기(XLM-R), 로컬/클라우드 LLM을 결합한 파이프라인으로 동작합니다.

```
N-gram  →  MFR  →  XLM-R  →  LLM
```

---

## 파이프라인 실행 흐름

3개 스크립트를 순서대로 실행합니다.

```
[Stage 1] mine_hard_cases_dev.py
    입력 문장 → N-gram·MFR 예측 계산 + XLM-R로 hard case 선별
    출력: hard_cases(.jsonl), ngram_mfr(.json)  ← LLM 직전 N-gram+MFR 합본 예측
        │
        ▼
[Stage 2] llm_correct_local.py
    hard case 토큰에 대해서만 LLM 정규화 호출 (dynamic few-shot)
    출력: llm_corrections(.jsonl)
        │
        ▼
[Stage 3] build_dev_submissions.py
    ngram_mfr 예측 위에 LLM 교정을 overlay → CodaBench 제출용 zip
    출력: submissions/predictions.{json,zip}
```

- **예측 우선순위**: N-gram → MFR → 원본 유지 (N-gram이 바꾸면 그 값, 아니면 MFR, 둘 다 아니면 raw)
- **LLM 게이팅**: N-gram·MFR이 손대지 않았고 XLM-R이 noise로 표시한 토큰(hard case)에만 LLM 호출 → 비용·시간 절약

---

## 폴더 / 파일 설명

### 실행 스크립트 (3-stage)
| 파일 | 단계 | 역할 |
|------|------|------|
| `mine_hard_cases_dev.py` | Stage 1 | N-gram·MFR 예측 계산 + XLM-R hard case 마이닝 |
| `llm_correct_local.py` | Stage 2 | LLM 정규화 교정 (Ollama / OpenAI) |
| `build_dev_submissions.py` | Stage 3 | overlay + 제출 zip 빌드 |

### 모듈 (스크립트가 import)
| 파일 | 역할 |
|------|------|
| `trigram_predictor.py` | N-gram 예측기 (trigram → bigram back-off) |
| `smart_guard_mfr_v2.py` | MFR 최빈 정규화 + 보호 토큰 가드 |
| `detection.py` | XLM-R 토큰 noise 탐지기 (`AnomalyDetector`) |
| `prompt_mfr_adapter.py` | MFR 사전 ↔ LLM 프롬프트 어댑터 |
| `normalization_fewshot.py` | 편집거리 기반 동적 few-shot 검색기 |
| `paths_config.py` | 모든 입출력 경로를 모은 중앙 설정 |
| `evaluation.py` | 공식 평가기 연동 브릿지 |

### 보조 도구
| 파일 | 역할 |
|------|------|
| `build_trigram_stats.py` | train → `outputs/trigram_stats.pkl.gz` 빌드 |
| `ablation_trigram.py` | N-gram 설정 grid ablation 러너 |

### 리소스 / 데이터
| 항목 | 내용 |
|------|------|
| `mfr_stats.pkl.gz` | MFR 통계 사전 (predict용 lookup table) |
| `outputs/trigram_stats.pkl.gz` | N-gram 통계 |
| `multilexnorm2026-dataset/` | train / validation / test parquet |
| `prompt_mfr_dictionary/` | 언어별 MFR 사전 + LLM 프롬프트 리소스 |
| `.env` | `OPENAI_KEY=...` (OpenAI 모델 사용 시) |

### 폴더
| 폴더 | 내용 |
|------|------|
| `baseline/` | LAI · MFR · ByT5 베이스라인 |
| `multilexnorm_eval_package_v2/` | 공식 ERR 평가기 (eval_groups 다중 뷰) |
| `outputs/` | 파이프라인 산출물 |
| `bin/` | 아카이브 (구 실험·데이터·문서) |

---

## 파이프라인 실행 방법

> [!IMPORTANT]
> Stage 2는 로컬 **Ollama 데몬** 또는 OpenAI API 키(`.env`)가 필요합니다.

입출력 경로는 **`paths_config.py` 상단 상수**로 지정합니다.
실험을 바꿀 때는 이 상수들만 편집하고 스크립트를 돌립니다.

```python
# paths_config.py
MINE_INPUT_PATH  = ...  # mine 입력 (.parquet 또는 .json)
HARD_CASES_PATH  = ...  # Stage 1 → 2
NGRAM_MFR_PATH   = ...  # Stage 1 → 3 (N-gram+MFR 합본 예측)
LLM_OUTPUT_PATH  = ...  # Stage 2 → 3
SUBMISSION_DIR   = ...  # Stage 3 최종 출력
```

### 1. 경로 설정
`paths_config.py`에서 위 5개 상수를 이번 실험에 맞게 수정한다.

### 2. Stage 1 — 마이닝
```bash
python mine_hard_cases_dev.py            # N-gram + MFR (기본)
python mine_hard_cases_dev.py --no-trigram   # MFR만
python mine_hard_cases_dev.py --no-mfr       # N-gram만
python mine_hard_cases_dev.py --mfr-first    # MFR 우선순위
```

### 3. Stage 2 — LLM 교정
```bash
# 로컬 Ollama
python llm_correct_local.py --model gemma4:latest --fewshot --pos-k 3 --neg-k 0 --workers 2

# OpenAI (.env의 OPENAI_KEY 사용, 별도 출력파일로)
python llm_correct_local.py --model gpt-5-mini-2025-08-07 --openai \
    --output llm_corrections_gpt5mini.jsonl --fewshot --pos-k 3 --neg-k 0 --workers 8
```
- 중단 후 같은 명령으로 재실행하면 처리분을 건너뛰고 **이어서 진행**(resume)한다.

### 4. Stage 3 — 빌드
```bash
python build_dev_submissions.py          # baseline + LLM overlay
python build_dev_submissions.py --no-llm # baseline만
```
→ `SUBMISSION_DIR/predictions.{json,zip}` 생성.

### 5. 평가 (공식 ERR)
```bash
python multilexnorm_eval_package_v2/multilexnorm_evaluator_v2.py evaluate \
  --gold_parquet multilexnorm2026-dataset/validation-00000-of-00001.parquet \
  --pred_path outputs/<submission>/predictions.json \
  --model_name <name> --dataset_name val \
  --eval_groups all official12 missing5 \
  --out_dir multilexnorm_eval_package_v2/outputs/results
```

---

## N-gram Ablation 사용법

N-gram 예측기의 `variant × conf_min × protect` 조합을 한 번에 평가합니다.
`tri-only`(N-gram 단독)와 `full`(전체 파이프라인) 두 관점을 모두 산출합니다.

```bash
python ablation_trigram.py --mode both          # tri-only + full
python ablation_trigram.py --mode tri-only      # N-gram 단독만
python ablation_trigram.py --mode full          # 전체 파이프라인만
```

- 출력: `outputs/ablation_trigram_results.csv` (조합별 ERR)
- 탐색 범위: variant 4종(`pure`/`tri_biL`/`tri_biR`/`tri_bi_both`), conf_min 0.50~1.00, protect(`non_protect`/`protect`)
- `full` 모드는 기존 LLM 캐시(`outputs/llm_corrections_val.jsonl`)를 재사용한다.

**현재 채택 설정**: `variant=tri_bi_both`, `conf_min=0.70`, `protect=non_protect` (ablation으로 선정)

---
