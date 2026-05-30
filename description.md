# MultiLexNorm_HW11 파일 인덱스 (2026-05-21 평탄화 후)

> 2026-05-21 정리로 trigram_first_package / mfr_first_package를 폐지하고 모든 코드를 루트로 평탄화.
> 옛 실험 스크립트는 `bin/cleanup_20260521/`, 무거운 산출물은 외부 `../_archive_20260521/`.
> 실험 카탈로그·분기점은 `RESEARCH_LOG.md` H-J 섹션, 실행방법은 `RUN.md` 참조.

## 최종 파이프라인 3-stage (루트 .py)

| 파일 | 단계 | 역할 |
|------|------|------|
| `mine_hard_cases_dev.py` | Stage 1 | dev 5972 row에서 hard case 추출 + 전체 baseline 계산 → `outputs/hard_cases_dev.jsonl` + `outputs/baseline_dev.json` |
| `llm_correct_local.py` | Stage 2 | ollama gemma로 hard case 토큰별 LLM 정규화 호출 (fewshot k=3) → `outputs/llm_corrections_*.jsonl` |
| `build_dev_submissions.py` | Stage 3 | `baseline_dev.json` + LLM 결과 overlay → `outputs/submissions_dev_final/predictions.{json,zip}` |

## 의존 모듈 (루트 .py)

| 파일 | 역할 |
|------|------|
| `trigram_predictor.py` | `predict_trigram()` — trigram override (fallback chain tri→biL/biR) |
| `smart_guard_mfr_v2.py` | `predict_smart_guarded_mfr_v2()` — MFR 단계 + protected 토큰 보호 |
| `detection.py` | `AnomalyDetector` — XLM-R wrapper (`predict_labels` argmax/threshold) |
| `prompt_mfr_adapter.py` | `PromptMFRResources` — LLM user prompt 빌더 |
| `normalization_fewshot.py` | `NormalizationFewshot` — 편집거리 동적 fewshot retriever (rapidfuzz) |
| `evaluation.py` | `compute_metrics()` — TP/FP/FN/TN/ERR. **최종 3-stage는 미사용**, 평가·검증 도구용 |

## 데이터 / 캐시

| 경로 | 내용 |
|------|------|
| `mfr_stats.pkl.gz` | MFR 통계 (lang별 token→norm 빈도). smart_guard_mfr_v2가 사용 |
| `outputs/trigram_stats_internal_v1.pkl.gz` | trigram/bigram/unigram 통계 (12MB) |
| `prompt_mfr_dictionary/` | 17개 언어 prompt 패키지 + common_prompt. ko/ja/it는 **v2 prompt** 반영됨 |
| `multilexnorm2026-dataset/` | train/validation/test parquet (dataset_17lang). `dataset_17lang/data/train-00000-of-00001.parquet`는 dynamic fewshot retrieve 풀로도 사용 |
| `../xlmr_finetuned_colab/` | fine-tuned XLM-RoBERTa detection 모델 (1.1GB). **MultiLexNorm_HW11 밖 부모 폴더**. 다른 경로면 `XLMR_MODEL_PATH` 환경변수 |
| `outputs/submission_dev/predictions.json` | codabench dev sample — dev 5972행 raw/lang 단일 소스 (mine 입력) |
| `outputs/hard_cases_dev.jsonl` | Stage 1 산출물 (~4065) — Stage 2 입력 |
| `outputs/baseline_dev.json` | Stage 1 산출물 (5972행 Tri+MFR baseline) — Stage 3 입력 |
| `outputs/submissions_dev_final/predictions.{json,zip}` | 최종 제출물 (build 산출) |
| `baseline/` | LAI/MFR/ByT5 baseline 스크립트 (codabench baseline 데모) |

## 문서

| 파일 | 역할 |
|------|------|
| `RESEARCH_LOG.md` | 연구 로그 H-G1~H-J. **H-J 섹션 = 실험 카탈로그**(dev 제출 #1~6, 분기점, 파일 매핑) |
| `RESEARCH_LOG_mfr_legacy.md` | 옛 mfr_first_package 연구 로그 (H-A~H-F) |
| `description.md` | 본 파일 — 평탄화 후 파일 인덱스 |
| `description_mfr_legacy.md` | 옛 mfr_first_package 파일 식별 |
| `RUN.md` | **최종 파이프라인 실행방법 + 검증 기준값** (코드 수정 후 동작 체크용) |
| `bin/cleanup_20260521/README.md` | bin으로 옮긴 옛 파일 용도 |

## 외부 (`../_archive_20260521/`)

LLM 결과 jsonl(21개) / trace CSV(동료 디버깅용) / 옛 캐시·산출물. 재현 시 LLM 재호출 대신 재활용.
