import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


CONFIG_MODEL = "LLM_model_config.json"
CONFIG_PROMPT = "prompt_config.json"
CONFIG_RUN = "run_config.json"


def read_json(path):
    """JSON 설정 파일을 읽어서 dict로 반환한다.

    input:
        path: 읽을 JSON 파일 경로
    output:
        dict: JSON 파일 내용
    """
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_text(path):
    """텍스트 파일을 UTF-8로 읽고 앞뒤 공백을 제거한다.

    input:
        path: 읽을 텍스트 파일 경로
    output:
        str: 파일 내용
    """
    return Path(path).read_text(encoding="utf-8-sig").strip()


def resolve_path(path_value, experiment_dir, project_dir):
    """config에 적힌 상대 경로를 실제 파일 경로로 변환한다.

    input:
        path_value: config 안의 경로 문자열
        experiment_dir: 현재 실험 폴더
        project_dir: LLM-base탐지모델 폴더
    output:
        Path: 존재 여부와 무관하게 해석된 경로
    """
    path = Path(path_value)
    if path.is_absolute():
        return path

    from_experiment = experiment_dir / path
    if from_experiment.exists():
        return from_experiment

    return project_dir / path


def tokens_to_text(value):
    """데이터셋의 토큰 리스트를 프롬프트에 넣기 좋은 문자열로 변환한다.

    input:
        value: list[str], numpy array, 또는 문자열
    output:
        str: 공백으로 이어 붙인 문장
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return " ".join("" if token is None else str(token) for token in value)
    return "" if value is None else str(value)


def normalize_tokens(value):
    """raw/norm 비교를 위해 토큰 값을 안정적인 list 또는 str로 변환한다.

    input:
        value: list[str], numpy array, 또는 문자열
    output:
        list[str] 또는 str: 비교 가능한 값
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return ["" if token is None else str(token) for token in value]
    return "" if value is None else str(value)


def answer_label(raw, norm):
    """raw와 norm을 비교해서 탐지 정답 라벨을 만든다.

    input:
        raw: 원본 토큰열
        norm: 정규화 후 토큰열
    output:
        str: 다르면 "1", 같으면 "0"
    """
    return "1" if normalize_tokens(raw) != normalize_tokens(norm) else "0"


def load_eval_rows(dataset_path, lang=None, limit=None, offset=0):
    """평가 대상 split을 읽고, LLM 입력에 필요한 row 목록을 만든다.

    input:
        dataset_path: validation/test parquet 경로
        lang: 특정 언어만 사용할 때의 언어 코드
        limit: 사용할 row 수
        offset: 앞에서 건너뛸 row 수
    output:
        list[dict]: row_index, lang, raw_text, answer_label 등을 담은 목록
    """
    df = pd.read_parquet(dataset_path)
    if lang:
        df = df[df["lang"].astype(str) == lang]
    if offset:
        df = df.iloc[offset:]
    if limit is not None:
        df = df.head(limit)

    rows = []
    has_norm = "norm" in df.columns
    for original_index, row in df.iterrows():
        rows.append(
            {
                "row_index": int(original_index) + 1,
                "lang": str(row.get("lang", "")),
                "raw": normalize_tokens(row["raw"]),
                "raw_text": tokens_to_text(row["raw"]),
                "answer_label": answer_label(row["raw"], row["norm"]) if has_norm else None,
            }
        )
    return rows


def load_train_examples(train_dataset_path):
    """few-shot 검색에 사용할 train 예시를 메모리에 올린다.

    input:
        train_dataset_path: train parquet 경로
    output:
        list[dict]: lang, raw_text, norm_text, label 등을 담은 train 예시 목록
    """
    df = pd.read_parquet(train_dataset_path)
    examples = []
    for original_index, row in df.iterrows():
        label = answer_label(row["raw"], row["norm"])
        examples.append(
            {
                "row_index": int(original_index) + 1,
                "lang": str(row.get("lang", "")),
                "raw_text": tokens_to_text(row["raw"]),
                "norm_text": tokens_to_text(row["norm"]),
                "label": label,
            }
        )
    return examples


def levenshtein_distance(a, b):
    """두 문자열 사이의 Levenshtein 수정거리를 계산한다.

    input:
        a: 첫 번째 문자열
        b: 두 번째 문자열
    output:
        int: 삽입/삭제/치환으로 변환하는 최소 비용
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def normalized_edit_distance(a, b):
    """문장 길이 차이를 보정한 수정거리 점수를 계산한다.

    input:
        a: 기준 문자열
        b: 비교 문자열
    output:
        float: 0에 가까울수록 비슷한 점수
    """
    denominator = max(len(a), len(b), 1)
    return levenshtein_distance(a, b) / denominator


def select_top_by_label(scored_examples, label, k):
    """특정 라벨의 train 예시 중 가장 가까운 k개를 고른다.

    input:
        scored_examples: score가 붙은 train 예시 목록
        label: "0" 또는 "1"
        k: 선택할 개수
    output:
        list[dict]: 선택된 예시 목록
    """
    if k <= 0:
        return []
    same_label = [example for example in scored_examples if example["label"] == label]
    return same_label[:k]


def retrieve_fewshot_examples(row, train_examples, positive_k=3, negative_k=3):
    """same-lang + normalized edit distance + label balance 방식으로 few-shot을 고른다.

    input:
        row: 현재 평가 입력 row
        train_examples: train 예시 전체 목록
        positive_k: label 1 예시 개수
        negative_k: label 0 예시 개수
    output:
        list[dict]: few-shot으로 사용할 train 예시 목록
    """
    same_lang_examples = [example for example in train_examples if example["lang"] == row["lang"]]
    candidates = same_lang_examples or train_examples

    scored = []
    for example in candidates:
        score = normalized_edit_distance(row["raw_text"], example["raw_text"])
        scored.append({**example, "distance": score})
    scored.sort(key=lambda example: (example["distance"], example["row_index"]))

    selected = []
    selected.extend(select_top_by_label(scored, "1", positive_k))
    selected.extend(select_top_by_label(scored, "0", negative_k))
    selected.sort(key=lambda example: (example["distance"], example["row_index"]))
    return selected


def format_fewshot_block(examples):
    """few-shot 예시를 프롬프트에 들어갈 Raw/Normalized 문자열로 만든다.

    input:
        examples: retrieve_fewshot_examples가 반환한 예시 목록
    output:
        str: 프롬프트에 삽입할 few-shot 블록
    """
    if not examples:
        return ""

    blocks = ["Examples:"]
    for example in examples:
        blocks.append(
            "\n".join(
                [
                    f"Raw: {example['raw_text']}",
                    f"Normalized: {example['norm_text']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(prompt_template, row, fewshot_examples):
    """프롬프트 템플릿에 few-shot과 현재 입력을 끼워 넣는다.

    input:
        prompt_template: prompt_v1.txt 같은 뼈대 프롬프트
        row: 현재 평가 입력 row
        fewshot_examples: 현재 입력에 맞춰 검색된 few-shot 예시
    output:
        str: LLM에 넘길 최종 프롬프트
    """
    fewshot_block = format_fewshot_block(fewshot_examples)
    return prompt_template.format(
        fewshot_block=fewshot_block,
        lang=row["lang"] or "unknown",
        input_text=row["raw_text"],
        row_index=row["row_index"],
    )


def call_ollama_chat(model_config, prompt):
    """Ollama chat API를 호출해서 LLM 응답 문자열을 받는다.

    input:
        model_config: LLM_model_config.json 내용
        prompt: LLM에 넘길 최종 프롬프트
    output:
        str: 모델의 message.content
    """
    payload = {
        "model": model_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": bool(model_config.get("think", False)),
        "options": {
            "temperature": model_config.get("temperature", 0),
            "num_ctx": model_config.get("num_ctx", 16384),
            "num_predict": model_config.get("num_predict", 1024),
        },
    }
    request = urllib.request.Request(
        model_config.get("ollama_chat_url", "http://localhost:11434/api/chat"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=model_config.get("timeout", 120)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama API에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.") from exc

    return data.get("message", {}).get("content", "").strip(), data.get("eval_count", 0)


def parse_response(response, fallback="1"):
    """LLM 응답에서 첫 0/1 라벨과 사유를 분리한다.

    input:
        response: 모델 raw response
        fallback: 파싱 실패 시 사용할 라벨
    output:
        tuple[str, str, bool]: label, reason, strict_parse 여부
    """
    text = "" if response is None else str(response).strip()
    match = re.match(r"^\s*([01])(?:\s+(.*))?$", text, flags=re.DOTALL)
    if match:
        label = match.group(1)
        reason = (match.group(2) or "").strip().replace("\n", " ")
        return label, reason, True

    loose_match = re.search(r"(^|\s)([01])(\s|$)", text)
    if loose_match:
        label = loose_match.group(2)
        return label, text.replace("\n", " "), False

    return fallback, text.replace("\n", " "), False


def confusion_counts(answer_labels, pred_labels):
    """정답과 예측을 비교해서 TP/TN/FP/FN을 계산한다.

    input:
        answer_labels: 정답 라벨 목록
        pred_labels: 예측 라벨 목록
    output:
        tuple[dict, list]: count dict와 오답 목록
    """
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    errors = []
    for index, (answer, pred) in enumerate(zip(answer_labels, pred_labels), start=1):
        if answer == "1" and pred == "1":
            counts["TP"] += 1
        elif answer == "0" and pred == "0":
            counts["TN"] += 1
        elif answer == "0" and pred == "1":
            counts["FP"] += 1
            errors.append({"position": index, "type": "FP"})
        elif answer == "1" and pred == "0":
            counts["FN"] += 1
            errors.append({"position": index, "type": "FN"})
    return counts, errors


def safe_divide(numerator, denominator):
    """0으로 나누는 상황을 피하면서 비율을 계산한다."""
    return numerator / denominator if denominator else 0.0


def benchmark_metrics(answer_labels, pred_labels):
    """탐지모델 성능 지표를 계산한다.

    input:
        answer_labels: 정답 라벨 목록
        pred_labels: 예측 라벨 목록
    output:
        dict: confusion matrix와 accuracy/precision/recall/F1 등
    """
    counts, errors = confusion_counts(answer_labels, pred_labels)
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    total = len(answer_labels)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "total": total,
        **counts,
        "accuracy": safe_divide(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "specificity": safe_divide(tn, tn + fp),
        "errors": errors,
    }


def write_lines(path, lines):
    """문자열 목록을 한 줄씩 파일에 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for line in lines:
            file.write(f"{line}\n")


def write_jsonl(path, records):
    """dict 목록을 JSONL 형식으로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data):
    """dict 데이터를 JSON 파일로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_benchmark_text(path, metrics):
    """benchmark 지표를 사람이 읽기 쉬운 txt 파일로 저장한다."""
    lines = [
        "=== Detection Benchmark ===",
        "Label meaning: 1 = normalization needed, 0 = normalization not needed",
        "",
        f"Total: {metrics['total']}",
        f"TP: {metrics['TP']}",
        f"TN: {metrics['TN']}",
        f"FP: {metrics['FP']}",
        f"FN: {metrics['FN']}",
        "",
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall   : {metrics['recall']:.4f}",
        f"F1       : {metrics['f1']:.4f}",
        f"Specificity: {metrics['specificity']:.4f}",
    ]
    write_lines(path, lines)


def write_summary(path, experiment_dir, model_config, prompt_config, run_config, metrics, parse_failures, total_tokens=0, elapsed_seconds=0):
    """실험 내용을 빠르게 훑어볼 수 있는 summary.md를 만든다."""
    throughput = f"{total_tokens / elapsed_seconds:.1f} tok/s" if elapsed_seconds > 0 else "N/A"
    lines = [
        "# LLM Detection Experiment Summary",
        "",
        f"- Experiment: `{experiment_dir.name}`",
        f"- Model: `{model_config.get('model')}`",
        f"- Prompt template: `{prompt_config.get('prompt_template')}`",
        f"- Dataset: `{run_config.get('dataset')}`",
        f"- Language: `{run_config.get('lang')}`",
        f"- Limit: `{run_config.get('limit')}`",
        f"- Parse failures: `{parse_failures}`",
        "",
        "## Metrics",
        "",
        f"- Total: {metrics['total']}",
        f"- TP: {metrics['TP']}",
        f"- TN: {metrics['TN']}",
        f"- FP: {metrics['FP']}",
        f"- FN: {metrics['FN']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- Specificity: {metrics['specificity']:.4f}",
        "",
        "## Runtime",
        "",
        f"- Total output tokens: {total_tokens}",
        f"- Elapsed time: {elapsed_seconds:.1f}s ({elapsed_seconds / 60:.1f}min)",
        f"- Throughput: {throughput}",
    ]
    write_lines(path, lines)


def output_path(experiment_dir, run_config, key):
    """run_config의 output_files 항목을 실험 폴더 기준 실제 경로로 바꾼다."""
    return experiment_dir / run_config["output_files"][key]


def run_experiment(experiment_dir, dry_run=False):
    """실험 폴더의 config들을 읽고 LLM 탐지 실험 전체를 수행한다.

    input:
        experiment_dir: config와 결과물이 들어갈 실험 폴더
        dry_run: True면 Ollama를 호출하지 않고 더미 응답을 사용
    output:
        None: 결과 파일들을 실험 폴더에 저장
    """
    experiment_dir = Path(experiment_dir)
    project_dir = Path(__file__).resolve().parent

    model_config = read_json(experiment_dir / CONFIG_MODEL)
    prompt_config = read_json(experiment_dir / CONFIG_PROMPT)
    run_config = read_json(experiment_dir / CONFIG_RUN)

    dataset_path = resolve_path(run_config["dataset"], experiment_dir, project_dir)
    prompt_path = resolve_path(prompt_config["prompt_template"], experiment_dir, project_dir)
    prompt_template = read_text(prompt_path)

    rows = load_eval_rows(
        dataset_path,
        lang=run_config.get("lang"),
        limit=run_config.get("limit"),
        offset=run_config.get("offset", 0),
    )

    fewshot_config = prompt_config.get("fewshot", {})
    train_examples = []
    if fewshot_config.get("enabled"):
        train_dataset_path = resolve_path(fewshot_config["train_dataset"], experiment_dir, project_dir)
        train_examples = load_train_examples(train_dataset_path)

    predictions = []
    answer_labels = []
    debug_records = []
    parse_failures = 0
    total_tokens = 0
    experiment_start = time.time()

    for position, row in enumerate(rows, start=1):
        fewshot_examples = []
        if fewshot_config.get("enabled"):
            fewshot_examples = retrieve_fewshot_examples(
                row,
                train_examples,
                positive_k=int(fewshot_config.get("positive_k", 3)),
                negative_k=int(fewshot_config.get("negative_k", 3)),
            )
        prompt = build_prompt(prompt_template, row, fewshot_examples)

        if dry_run:
            response = "1 dry run placeholder"
        else:
            response, eval_count = call_ollama_chat(model_config, prompt)
            total_tokens += eval_count
            if model_config.get("sleep", 0):
                time.sleep(float(model_config["sleep"]))

        label, reason, strict_parse = parse_response(response, fallback=run_config.get("fallback", "1"))
        if not strict_parse:
            parse_failures += 1

        predictions.append(label)
        answer_labels.append(row["answer_label"])
        debug_records.append(
            {
                "position": position,
                "row_index": row["row_index"],
                "lang": row["lang"],
                "raw_text": row["raw_text"],
                "answer_label": row["answer_label"],
                "prediction": label,
                "reason": reason,
                "raw_response": response,
                "strict_parse": strict_parse,
                "fewshot_examples": fewshot_examples,
            }
        )
        print(f"[{position}/{len(rows)}] row={row['row_index']} label={label}", flush=True)

    elapsed = time.time() - experiment_start
    metrics = benchmark_metrics(answer_labels, predictions)

    write_lines(output_path(experiment_dir, run_config, "predictions"), predictions)
    write_lines(output_path(experiment_dir, run_config, "answer_labels"), answer_labels)
    write_jsonl(output_path(experiment_dir, run_config, "debug"), debug_records)
    write_json(output_path(experiment_dir, run_config, "benchmark_json"), metrics)
    write_benchmark_text(output_path(experiment_dir, run_config, "benchmark_txt"), metrics)
    write_summary(
        output_path(experiment_dir, run_config, "summary"),
        experiment_dir,
        model_config,
        prompt_config,
        run_config,
        metrics,
        parse_failures,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed,
    )

    throughput = f"{total_tokens / elapsed:.1f} tok/s" if elapsed > 0 else "N/A"
    print(f"\nTotal output tokens : {total_tokens}")
    print(f"Elapsed time        : {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"Throughput          : {throughput}")


def main():
    parser = argparse.ArgumentParser(description="Run one config-based LLM detection experiment.")
    parser.add_argument("experiment_dir", help="Folder containing LLM_model_config.json, prompt_config.json, run_config.json.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Ollama; write placeholder outputs.")
    args = parser.parse_args()

    try:
        run_experiment(args.experiment_dir, dry_run=args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
