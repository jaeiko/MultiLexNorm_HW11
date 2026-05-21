# mfr_first_package 파일 식별

> 각 파일이 어떤 실험을 위해 만들어졌고 어떤 결과물을 산출하는지 정리.
> 메모(`research_log.md`)에 적힌 흐름의 재현에 필요한 파일들만 루트에 남아 있고,
> 나머지는 `bin/`에 보존 (참조 가능, 카테고리별 1줄 설명은 `bin/README.md`).

---

## ✅ 인프라 (모든 실험에서 import)

| 파일 | 역할 |
|------|------|
| `smart_guard_mfr_v2.py` | MFR 적용 함수 + protect token 로직 (URL/숫자/@/#/특수문자) |
| `evaluation.py` | TP/FP/FN/TN, Precision/Recall/F1/ERR 표준 계산 |
| `mfr_stats.pkl.gz` | MFR 사전 (`{lang: {token: {top_norm, confidence, top_count, total_count, candidates}}}`) |

---

## 📦 메모 §3 재현 — 유사 MFR 수정정책 (32옵션 그리드)

| 파일 | 역할 | 산출/입력 |
|------|------|------|
| `precompute_retrieve_cache.py` | mini validation set에 대해 lang별 top-k(=20) sentence retrieval cache 빌드 | → `outputs/retrieve_k20_cache_mini.json` |
| `experiment_rules_v3.py` | 32옵션(P1~P4 × k × P3 conf) 그리드 실행. cache 로드 후 retrieve 호출 0번 | ← `retrieve_k20_cache_mini.json` → `outputs/experiment_results/rules_v3_results.json` |
| `outputs/retrieve_k20_cache_mini.json` | mini set 1131 rows × k=20 retrieve 결과 cache | — |
| `outputs/experiment_results/rules_v3_results.json` | 32옵션 ERR/precision/recall/coverage 결과 | best: P2_k20 / P3_k20_c0.60 ERR **0.2467** |

---

## 📦 메모 §3 후반 재현 — P1 활용 + MFR fusion (H-A 시리즈)

| 파일 | 역할 | 산출/입력 |
|------|------|------|
| `precompute_retrieve_cache_v3.py` | internal_v1 9056 rows × k=50 retrieve cache 빌드 (label=1 문장만 필터) | → `outputs/retrieve_k50_cache_internal_v1.json` |
| `experiment_HA1_veto_threshold.py` | t2t 임계 (count_min × conf_min) 그리드. t2t 강할 때 MFR 변환을 raw로 veto | ← `retrieve_k50_cache_internal_v1.json` → `HA1_veto_threshold_internal_v1.json` |
| `experiment_HA2_lang_veto.py` | 언어별 best threshold 적용 (ja: cm=15, cn=0.50 / th: cm=3, cn=0.70) | → `HA2_lang_veto_internal_v1.json` |
| `outputs/retrieve_k50_cache_internal_v1.json` | internal_v1 retrieve cache (~154MB) | — |
| `outputs/experiment_results/HA1_veto_threshold_internal_v1.json` | H-A1 그리드 결과. best ERR **0.5345** (count_min=8, conf=0.60) | +0.0010 vs MFR-only |
| `outputs/experiment_results/HA2_lang_veto_internal_v1.json` | H-A2 언어별 best 결과. ERR **0.5349** | +0.0014 vs MFR-only ⭐ |

---

## 📦 메모 §1 재현 — LLM 수정모델 옵션 변경 시도

| 파일 | 역할 |
|------|------|
| `stage3_llm_correction.py` | 옛 Stage 3 = MFR 적용 → token detection → LLM(Qwen2.5-7B / qwen3 27b) 수정. dynamic fewshot 지원 |
| `pipeline.py` | `LexicalNormalizationPipeline` — detector/dictionary/llm 묶어서 process_batch 실행 |
| `llm.py` | `MultilingualCorrector` wrapper (HuggingFace AutoModelForCausalLM 4-bit 로드 + generate) |
| `prompt_mfr_adapter.py` | `PromptMFRResources` — lang별 prompt 빌더 호출, dynamic fewshot block 삽입 |
| `normalization_fewshot.py` | `NormalizationFewshot` — train에서 target token에 대해 Levenshtein 기반 positive/negative pairs retrieve |
| `prompt_mfr_dictionary/` | 14개 lang별 prompt builder + fewshot 6개 + lang-specific rule block |
| `outputs/stage3_metrics.json` | qwen2.5-7B + token detection(stage2) + dynamic fewshot 결과. ERR **-0.0276** (mini 219 rows, 흐름 = MFR → 탐지 → 수정) |
| `outputs/stage3_summary.md` | 위 결과 markdown 요약 |
| `outputs/stage3_predictions_final.json` | stage 3 토큰별 최종 예측 결과 |

---

## 📝 문서

| 파일 | 역할 |
|------|------|
| `research_log.md` | 메모 (notion 형식, §1 LLM 옵션변경 / §2 파이프라인 순서 변경 / §3 유사 MFR 수정정책) |
| `RESEARCH_LOG.md` | 상세 연구 로그 (Glossary + H-A 시리즈 표 + 인사이트). research_log.md의 백업/세부 자료 |
| `description.md` | 본 파일 — 파일별 역할 식별 |

---

## 📁 폴더

| 폴더 | 역할 |
|------|------|
| `data/` | `train_internal_v1.parquet` 등 train/val parquet (H-A 시리즈에서 직접 로드) |
| `outputs/` | retrieve cache + experiment_results JSON + stage3 결과 |
| `outputs/experiment_results/` | 32옵션 (rules_v3) + H-A1/H-A2 JSON |
| `prompt_mfr_dictionary/` | LLM correction용 lang별 prompt 패키지 |
| `bin/` | 메모 흐름에 직접 안 들어가는 옛 코드/결과 보존 (50개 파일 + README.md) |

---

## 🏆 메모 흐름 핵심 결과

| 단계 | best ERR | 비고 |
|------|---------:|------|
| §3 32옵션 그리드 (Rule-only, mini) | **0.2467** | P2_k20 / P3_k20_c0.60 |
| §3 P1 활용 H-A1 (MFR + veto, internal_v1) | 0.5345 | count_min=8, conf=0.60 |
| §3 P1 활용 H-A2 (언어별, internal_v1) | **0.5349** | ja/th 별도 threshold |
| §1 옛 Stage 3 LLM (mini 219, MFR → 탐지 → 수정) | -0.0276 | sentence-level 과정규화 잔재 |

(MFR-only baseline = 0.5335)

---

## 🔄 재현 절차 요약

### 32옵션 그리드 (mini)
```bash
python precompute_retrieve_cache.py    # mini cache 빌드 (한 번만)
python experiment_rules_v3.py          # 32옵션 그리드 → rules_v3_results.json
```

### H-A 시리즈 (internal_v1)
```bash
python precompute_retrieve_cache_v3.py # internal_v1 k=50 cache (한 번만)
python experiment_HA1_veto_threshold.py
python experiment_HA2_lang_veto.py
```

### Stage 3 LLM
```bash
# GPU + HuggingFace 모델 필요
python stage3_llm_correction.py
```
