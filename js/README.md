# MultiLexNorm 2026 — Final Pipeline

Tri override → MFR (guarded) → XLM-R S5 fallback → (optional) LLM correction on hard cases.

## 결과 요약 (validation 9056 rows, internal_v1)

| Combo | TP | FP | FN | ERR |
|---|---:|---:|---:|---:|
| tri_only | 3391 | 140 | 12764 | 0.2012 |
| mfr_only | 9240 | 621 | 6915 | 0.5335 |
| tri_mfr | 9526 | 685 | 6629 | 0.5473 |
| mfr_xlmr | 9630 | 782 | 6525 | 0.5477 |
| **tri_mfr_xlmr (full)** | **9738** | **829** | **6417** | **0.5515** |

LLM overlay (gemma e4b + dynamic fewshot k=3, hard case 588): ΔERR **+0.058** (full → ~0.609).

---

## 설치

```bash
pip install -r requirements.txt
```

XLM-R detection 모델은 용량이 커서 repo에 포함하지 않음. Google Drive에서 받아 `external/xlmr_finetuned_colab/` 안에 풀어둘 것.

**XLM-R 모델 다운로드**: `<DRIVE_LINK_HERE>`

```bash
# 받은 폴더 구조가 다음과 같아야 함
external/xlmr_finetuned_colab/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Dataset과 통계 pkl은 이미 repo 안에 포함되어 있어 추가 다운로드 불필요.

---

## 환경변수 (override 가능, 평소엔 default로 충분)

| 변수 | Default | 용도 |
|---|---|---|
| `XLMR_MODEL_PATH` | `external/xlmr_finetuned_colab/` | XLM-R 가중치 폴더 |
| `DATA_DIR` | `multilexnorm2026-dataset/` | dataset 루트 |
| `OPENAI_API_KEY` | — | OpenAI 사용 시 필수 |
| `OPENAI_MODEL` | `gpt-5-mini` | OpenAI 모델명 |
| `REASONING_EFFORT` | — | `minimal`/`low`/`medium`/`high` (GPT-5 계열) |
| `HARD_CASES_FILE` | `hard_cases_mini.jsonl` | LLM 호출 입력 JSONL |

---

## 사용법

### 1. Baseline only (XLM-R S5 까지, no LLM)

```bash
cd trigram_first_package
python predict_test_no_llm.py
# → outputs/test_pred_baseline_noLLM.dev  (parquet, .dev 확장자)
```

### 2. Validation 5-combo 재현

```bash
cd trigram_first_package
python eval_val_combos.py
# → ERR full=0.5515, mfr_only=0.5335 등 표 결과 그대로 출력
```

### 3. Hard case mining + LLM correction (Ollama gemma e4b)

Ollama 서버를 띄운 상태에서 (`ollama serve`):

```bash
cd trigram_first_package

# (1) hard case 추출
python mine_hard_cases_mini_realistic.py
# → outputs/hard_cases_mini_realistic.jsonl  (588개)

# (2) gemma e4b + dynamic fewshot k=3 (best 설정)
HARD_CASES_FILE=hard_cases_mini_realistic.jsonl \
python llm_correct_local.py \
    --model gemma4:latest \
    --output llm_corrections_realistic_gemma4_4b_fewshot3.jsonl \
    --workers 2 --fewshot --pos-k 3 --neg-k 0

# (3) baseline + LLM overlay 평가
python eval_llm_realistic.py \
    --input llm_corrections_realistic_gemma4_4b_fewshot3.jsonl \
    --label gemma4_4b_fewshot3
# → ΔERR ~+0.058
```

### 4. LLM correction (OpenAI GPT-5-mini)

```bash
cd trigram_first_package
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-5-mini
export REASONING_EFFORT=minimal

HARD_CASES_FILE=hard_cases_mini_realistic.jsonl \
FEWSHOT_K=3 \
OUTPUT_FILE=llm_corrections_gpt5mini.jsonl \
python llm_correct_oldprompt.py

python eval_llm_realistic.py \
    --input llm_corrections_gpt5mini.jsonl --label gpt5mini
```

### 5. Codabench dev phase submission 빌드 (5-combo)

```bash
cd trigram_first_package
python predict_test_combos.py
# → outputs/submissions_combos/<combo>/predictions.zip  (5개 zip)
```

각 zip 안에 `predictions.json` 하나. Codabench에 업로드.

---

## 파이프라인 구조 (요약)

```
Stage A (모든 토큰)
  Tri override (lang ∉ {it, es, th, id})
    └─ no override → MFR (guarded, protected token 보호)
          └─ MFR top1 == raw → XLM-R S5: XLM-R abnormal(1)이면 MFR top-2 후보
                └─ 그래도 변환 없음 → raw 유지

Stage A.5 — Hard case mining (production-realistic)
  baseline_pred == raw AND XLM-R == 1 AND not protected → JSONL 1줄

Stage B — LLM correction (Stage A.5 결과에만)
  토큰 1개당 LLM call 1회 = JSON {"normalized": "..."}
  - System: "lexical normalization, JSON only, no thinking"
  - User: PromptMFRResources.build_normalization_prompt 결과
      · 본체: 언어별 정적 fewshot 6 + 룰 + JSON 스키마
      · MFR review hint (대상이 사전 ambiguous일 때)
      · Dynamic fewshot block (Levenshtein 유사 train 사례 k개)

Overlay → eval_llm_realistic.py가 baseline 위에 LLM 답 덮어쓰고 TP/FP/FN/TN/ERR 계산
```

---

## 폴더 구조

```
js/
├── README.md                            (this file)
├── requirements.txt
├── mfr_first_package/
│   ├── smart_guard_mfr_v2.py            MFR + protected_indices
│   ├── evaluation.py                    compute_metrics (TP/FP/FN/TN/ERR)
│   ├── prompt_mfr_adapter.py            PromptMFRResources (LLM prompt builder)
│   ├── normalization_fewshot.py         dynamic fewshot retriever (rapidfuzz)
│   ├── mfr_stats.pkl.gz                 우리 MFR 통계
│   ├── data/train_internal_v1.parquet   dynamic fewshot 검색 풀
│   └── prompt_mfr_dictionary/           17개 언어 prompt + MFR 사전
├── trigram_first_package/
│   ├── trigram_predictor.py             trigram chain
│   ├── predict_test_combos.py           test 5-combo + codabench zip 빌드
│   ├── predict_test_no_llm.py           baseline-only test 예측
│   ├── mine_hard_cases_mini_realistic.py
│   ├── count_hard_cases.py
│   ├── llm_correct_local.py             Ollama (gemma/qwen3)
│   ├── llm_correct_oldprompt.py         OpenAI (GPT-5-mini 등)
│   ├── eval_llm_realistic.py            baseline + LLM overlay 평가
│   ├── eval_val_combos.py               validation 5-combo 평가
│   └── outputs/trigram_stats_internal_v1.pkl.gz   우리 trigram 통계
├── multilexnorm2026-dataset/            dataset (16 MB, 포함)
├── outputs/submission_dev/predictions.json    codabench dev sample (2.7 MB)
└── external/
    └── xlmr_finetuned_colab/            ← 비어있음, Drive에서 다운로드해 채울 것
```

---

## 모듈 의존 그래프

```
predict_test_no_llm.py
  ├── trigram_predictor.predict_trigram
  ├── smart_guard_mfr_v2.predict_smart_guarded_mfr_v2
  ├── smart_guard_mfr_v2.find_protected_indices
  └── (HuggingFace) XLM-R AutoModelForTokenClassification

mine_hard_cases_mini_realistic.py
  └── 위와 동일 + JSONL writer

llm_correct_{local,oldprompt}.py
  ├── prompt_mfr_adapter.PromptMFRResources.build_normalization_prompt
  ├── normalization_fewshot.NormalizationFewshot.retrieve   (옵션)
  └── prompt_mfr_dictionary.common_prompt.extract_first_json_object

eval_llm_realistic.py
  ├── trigram_predictor + smart_guard_mfr_v2 + XLM-R (Stage A 재실행)
  └── evaluation.compute_metrics
```
