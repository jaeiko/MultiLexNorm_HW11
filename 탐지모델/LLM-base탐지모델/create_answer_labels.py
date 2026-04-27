import argparse
from pathlib import Path

import pandas as pd


def normalize_value(value, strip=False):
    """raw/norm 값을 비교 가능한 형태로 변환한다.

    input:
        value: 토큰 리스트, numpy array, 문자열 중 하나
        strip: 문자열 앞뒤 공백을 무시할지 여부
    output:
        list[str] 또는 str: 비교에 사용할 값
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        tokens = ["" if token is None else str(token) for token in value]
        return [token.strip() for token in tokens] if strip else tokens
    text = "" if value is None else str(value)
    return text.strip() if strip else text


def make_answer_labels(dataset_path, strip=False):
    """raw와 norm을 비교해서 탐지모델 정답 라벨을 만든다.

    input:
        dataset_path: raw, norm 컬럼이 있는 parquet 파일
        strip: 앞뒤 공백 차이를 무시할지 여부
    output:
        list[int]: raw != norm이면 1, raw == norm이면 0
    """
    df = pd.read_parquet(dataset_path)
    if "raw" not in df.columns or "norm" not in df.columns:
        columns = ", ".join(df.columns)
        raise ValueError(f"Dataset must have raw and norm columns. Available columns: {columns}")

    labels = []
    for raw, norm in zip(df["raw"], df["norm"]):
        raw_value = normalize_value(raw, strip=strip)
        norm_value = normalize_value(norm, strip=strip)
        labels.append(1 if raw_value != norm_value else 0)
    return labels


def write_labels(labels, output_path):
    """라벨 목록을 한 줄에 하나씩 txt 파일로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for label in labels:
            file.write(f"{label}\n")


def print_summary(labels, output_path):
    """생성된 answer label 파일의 간단한 통계를 출력한다."""
    total = len(labels)
    positive = sum(labels)
    negative = total - positive
    positive_ratio = positive / total if total else 0.0

    print(f"Saved: {output_path}")
    print(f"Total: {total}")
    print(f"1 normalization needed    : {positive} ({positive_ratio:.2%})")
    print(f"0 normalization not needed: {negative} ({1 - positive_ratio:.2%})")


def main():
    parser = argparse.ArgumentParser(
        description="Create answer labels by comparing raw and norm columns in a parquet dataset."
    )
    parser.add_argument("dataset", help="Parquet dataset file with raw and norm columns.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output txt file. Defaults to answer_labels/<dataset-stem>.txt.",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Ignore leading/trailing whitespace before comparing values.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output) if args.output else Path("answer_labels") / f"{dataset_path.stem}.txt"

    labels = make_answer_labels(dataset_path, strip=args.strip)
    write_labels(labels, output_path)
    print_summary(labels, output_path)


if __name__ == "__main__":
    main()
