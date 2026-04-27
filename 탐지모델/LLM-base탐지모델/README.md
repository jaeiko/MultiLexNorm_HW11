# LLM-base 탐지모델

MultiLexNorm 입력 문장이 lexical normalization을 필요로 하는지 `0/1`로
탐지하는 LLM 기반 파이프라인입니다.

라벨 의미:

- `1`: 정규화 필요
- `0`: 정규화 불필요

LLM raw response는 `<label> <reason>` 형식을 사용합니다.
실험 실행 후에는 채점용 `predictions.txt`, 분석용 `debug.jsonl`,
정답지 `answer_labels.txt`, 벤치마크 결과 파일이 실험 폴더 안에 함께 저장됩니다.

## 현재 실행 방식

`run_llm_experiment.py`는 실험 폴더 안의 config 3개를 읽어서 LLM 실행,
예측 저장, 디버그 로그 저장, 정답 라벨 저장, 벤치마크 저장, 요약 저장을
한 번에 수행합니다.

필수 config:

- `LLM_model_config.json`: Ollama 모델 설정
- `prompt_config.json`: 프롬프트와 few-shot 설정
- `run_config.json`: 데이터셋, 언어, 출력 파일 설정

실행 예시:

```powershell
python -X utf8 run_llm_experiment.py experiments/exp_001_retrieval_fewshot_v1
```

Ollama를 호출하지 않고 파일 생성 흐름만 확인하려면:

```powershell
python -X utf8 run_llm_experiment.py experiments/exp_001_retrieval_fewshot_v1 --dry-run
```

## Few-shot 방식

현재 `exp_001_retrieval_fewshot_v1`은 다음 방식을 사용합니다.

```text
same-lang + normalized edit distance + label balance
```

현재 입력과 같은 언어의 train 샘플 중 문자 수정거리가 가까운 예시를 찾고,
`raw != norm` 예시와 `raw == norm` 예시를 균형 있게 프롬프트에 넣습니다.

few-shot 예시는 사람이 작성한 사유가 아니라 데이터셋의 실제 예시를 사용합니다.

```text
Raw: ...
Normalized: ...
```

## 정답지 생성 도구

실험 실행 시 `answer_labels.txt`가 자동 생성됩니다. 별도로 특정 parquet 파일의
정답지만 만들고 싶다면 `create_answer_labels.py`를 사용할 수 있습니다.

```powershell
python -X utf8 create_answer_labels.py ../../multilexnorm2026-dataset/data/validation-00000-of-00001.parquet -o answer_labels/validation.txt
```
