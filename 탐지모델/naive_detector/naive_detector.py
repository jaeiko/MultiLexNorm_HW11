import argparse
from pathlib import Path

'''
항상 1을 반환하는 탐지모델입니다
'''


def count_gold_rows(gold_path):
    with gold_path.open("r", encoding="utf-8-sig") as file:
        return sum(1 for line in file if line.strip())


def write_always_positive_predictions(row_count, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for _ in range(row_count):
            file.write("1\n")


def main():
    parser = argparse.ArgumentParser(
        description="Naive detector baseline: always predicts 1, meaning normalization is needed."
    )
    parser.add_argument("gold", help="Gold label txt file. Used only to match the row count.")
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/naive_always_1_predictions.txt",
        help="Output prediction txt file.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    output_path = Path(args.output)
    row_count = count_gold_rows(gold_path)
    write_always_positive_predictions(row_count, output_path)

    print(f"Saved: {output_path}")
    print(f"Rows: {row_count}")
    print("Prediction: always 1")


if __name__ == "__main__":
    main()
