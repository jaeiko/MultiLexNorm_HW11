from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


MISSING = "<MISSING>"


def as_tokens(value) -> list[str]:
    """Convert a parquet cell or JSON value into a stable token list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return ["" if token is None else str(token) for token in value]
    if value is None:
        return []
    return [str(value)]


def classify_token_error(raw_token: str, norm_token: str, pred_token: str) -> str:
    """Classify one token-level mismatch into a useful debugging category."""
    if pred_token == norm_token:
        return "correct"
    if pred_token == MISSING:
        return "prediction_missing"
    if norm_token == MISSING:
        return "prediction_extra"
    if raw_token == norm_token:
        return "over_normalized"
    if pred_token == raw_token:
        return "missed_change"
    return "wrong_correction"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_rate(numerator: int, denominator: int) -> str:
    """Format a safe percentage string."""
    if denominator == 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def analyze(dataset_path: Path, predictions_path: Path, output_dir: Path) -> dict:
    """Compare full-pipeline predictions against validation labels."""
    df = pd.read_parquet(dataset_path)
    with predictions_path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)

    if len(df) != len(predictions):
        raise ValueError(f"Row count mismatch: dataset={len(df)}, predictions={len(predictions)}")

    language_stats = defaultdict(
        lambda: {
            "sentences": 0,
            "sentence_exact": 0,
            "tokens": 0,
            "token_correct": 0,
            "unchanged_tokens": 0,
            "changed_tokens": 0,
            "over_normalized": 0,
            "missed_change": 0,
            "wrong_correction": 0,
            "prediction_missing": 0,
            "prediction_extra": 0,
            "length_mismatch": 0,
        }
    )

    category_counts = Counter()
    token_pattern_counts = Counter()
    token_error_rows = []
    sentence_error_rows = []

    total_sentences = 0
    exact_sentences = 0
    total_tokens = 0
    correct_tokens = 0
    raw_norm_length_mismatches = 0
    pred_norm_length_mismatches = 0

    for row_number, (prediction, row) in enumerate(zip(predictions, df.itertuples(index=False)), start=1):
        raw_tokens = as_tokens(row.raw)
        norm_tokens = as_tokens(row.norm)
        pred_tokens = as_tokens(prediction)
        lang = str(row.lang)

        total_sentences += 1
        language_stats[lang]["sentences"] += 1

        if raw_tokens == norm_tokens:
            pass
        if len(raw_tokens) != len(norm_tokens):
            raw_norm_length_mismatches += 1
        if len(pred_tokens) != len(norm_tokens):
            pred_norm_length_mismatches += 1
            language_stats[lang]["length_mismatch"] += 1

        if pred_tokens == norm_tokens:
            exact_sentences += 1
            language_stats[lang]["sentence_exact"] += 1

        sentence_token_errors = []
        max_len = max(len(raw_tokens), len(norm_tokens), len(pred_tokens))
        for token_position in range(max_len):
            raw_token = raw_tokens[token_position] if token_position < len(raw_tokens) else MISSING
            norm_token = norm_tokens[token_position] if token_position < len(norm_tokens) else MISSING
            pred_token = pred_tokens[token_position] if token_position < len(pred_tokens) else MISSING

            if norm_token != MISSING:
                total_tokens += 1
                language_stats[lang]["tokens"] += 1
                if raw_token == norm_token:
                    language_stats[lang]["unchanged_tokens"] += 1
                else:
                    language_stats[lang]["changed_tokens"] += 1

            category = classify_token_error(raw_token, norm_token, pred_token)
            if category == "correct":
                if norm_token != MISSING:
                    correct_tokens += 1
                    language_stats[lang]["token_correct"] += 1
                continue

            category_counts[category] += 1
            if category in language_stats[lang]:
                language_stats[lang][category] += 1

            token_error = {
                "row_index": row_number,
                "lang": lang,
                "token_position": token_position + 1,
                "category": category,
                "raw_token": raw_token,
                "norm_token": norm_token,
                "pred_token": pred_token,
                "raw_text": " ".join(raw_tokens),
                "norm_text": " ".join(norm_tokens),
                "pred_text": " ".join(pred_tokens),
            }
            token_error_rows.append(token_error)
            sentence_token_errors.append(token_error)
            token_pattern_counts[(lang, category, raw_token, norm_token, pred_token)] += 1

        if sentence_token_errors:
            sentence_error_rows.append(
                {
                    "row_index": row_number,
                    "lang": lang,
                    "error_count": len(sentence_token_errors),
                    "categories": dict(Counter(error["category"] for error in sentence_token_errors)),
                    "raw": raw_tokens,
                    "norm": norm_tokens,
                    "pred": pred_tokens,
                    "token_errors": [
                        {
                            "token_position": error["token_position"],
                            "category": error["category"],
                            "raw_token": error["raw_token"],
                            "norm_token": error["norm_token"],
                            "pred_token": error["pred_token"],
                        }
                        for error in sentence_token_errors
                    ],
                }
            )

    language_rows = []
    for lang, stats in sorted(language_stats.items()):
        language_rows.append(
            {
                "lang": lang,
                "sentences": stats["sentences"],
                "sentence_exact": stats["sentence_exact"],
                "sentence_exact_rate": format_rate(stats["sentence_exact"], stats["sentences"]),
                "tokens": stats["tokens"],
                "token_correct": stats["token_correct"],
                "token_accuracy": format_rate(stats["token_correct"], stats["tokens"]),
                "changed_tokens": stats["changed_tokens"],
                "unchanged_tokens": stats["unchanged_tokens"],
                "over_normalized": stats["over_normalized"],
                "missed_change": stats["missed_change"],
                "wrong_correction": stats["wrong_correction"],
                "prediction_missing": stats["prediction_missing"],
                "prediction_extra": stats["prediction_extra"],
                "length_mismatch": stats["length_mismatch"],
            }
        )

    top_pattern_rows = []
    for (lang, category, raw_token, norm_token, pred_token), count in token_pattern_counts.most_common(200):
        top_pattern_rows.append(
            {
                "count": count,
                "lang": lang,
                "category": category,
                "raw_token": raw_token,
                "norm_token": norm_token,
                "pred_token": pred_token,
            }
        )

    worst_sentences = sorted(sentence_error_rows, key=lambda item: (-item["error_count"], item["row_index"]))[:80]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "language_summary.csv",
        [
            "lang",
            "sentences",
            "sentence_exact",
            "sentence_exact_rate",
            "tokens",
            "token_correct",
            "token_accuracy",
            "changed_tokens",
            "unchanged_tokens",
            "over_normalized",
            "missed_change",
            "wrong_correction",
            "prediction_missing",
            "prediction_extra",
            "length_mismatch",
        ],
        language_rows,
    )
    write_csv(
        output_dir / "token_errors.csv",
        [
            "row_index",
            "lang",
            "token_position",
            "category",
            "raw_token",
            "norm_token",
            "pred_token",
            "raw_text",
            "norm_text",
            "pred_text",
        ],
        token_error_rows,
    )
    write_csv(
        output_dir / "top_token_error_patterns.csv",
        ["count", "lang", "category", "raw_token", "norm_token", "pred_token"],
        top_pattern_rows,
    )
    write_jsonl(output_dir / "sentence_errors.jsonl", sentence_error_rows)

    changed_tokens = sum(stats["changed_tokens"] for stats in language_stats.values())
    unchanged_tokens = sum(stats["unchanged_tokens"] for stats in language_stats.values())
    summary = {
        "dataset": str(dataset_path),
        "predictions": str(predictions_path),
        "total_sentences": total_sentences,
        "sentence_exact": exact_sentences,
        "sentence_exact_rate": exact_sentences / total_sentences if total_sentences else 0.0,
        "total_tokens": total_tokens,
        "token_correct": correct_tokens,
        "token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        "changed_tokens": changed_tokens,
        "unchanged_tokens": unchanged_tokens,
        "raw_norm_length_mismatches": raw_norm_length_mismatches,
        "pred_norm_length_mismatches": pred_norm_length_mismatches,
        "category_counts": dict(category_counts),
    }
    write_summary(output_dir / "summary.md", summary, language_rows, top_pattern_rows, worst_sentences)
    return summary


def write_summary(path: Path, summary: dict, language_rows: list[dict], top_patterns: list[dict], worst_sentences: list[dict]) -> None:
    """Write a compact Markdown report for human review."""
    category_counts = Counter(summary["category_counts"])
    total_errors = sum(category_counts.values())
    lines = [
        "# Full Pipeline Error Analysis",
        "",
        "## Overall",
        "",
        f"- Sentences: {summary['total_sentences']}",
        f"- Sentence exact match: {summary['sentence_exact']} ({summary['sentence_exact_rate']:.4f})",
        f"- Tokens: {summary['total_tokens']}",
        f"- Token accuracy: {summary['token_accuracy']:.4f}",
        f"- Changed target tokens: {summary['changed_tokens']}",
        f"- Unchanged target tokens: {summary['unchanged_tokens']}",
        f"- raw/norm length mismatch sentences: {summary['raw_norm_length_mismatches']}",
        f"- pred/norm length mismatch sentences: {summary['pred_norm_length_mismatches']}",
        "",
        "## Error Categories",
        "",
    ]

    descriptions = {
        "over_normalized": "raw==norm but prediction changed it",
        "missed_change": "raw!=norm but prediction left raw unchanged",
        "wrong_correction": "raw!=norm and prediction changed it, but not to norm",
        "prediction_missing": "prediction is shorter than norm",
        "prediction_extra": "prediction is longer than norm",
    }
    for category, count in category_counts.most_common():
        lines.append(f"- {category}: {count} ({format_rate(count, total_errors)}) - {descriptions.get(category, '')}")

    lines.extend(["", "## Language Summary", ""])
    lines.append("| lang | token_acc | sent_exact | errors | over | missed | wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(language_rows, key=lambda item: item["token_accuracy"]):
        errors = row["over_normalized"] + row["missed_change"] + row["wrong_correction"] + row["prediction_missing"] + row["prediction_extra"]
        lines.append(
            f"| {row['lang']} | {row['token_accuracy']} | {row['sentence_exact_rate']} | {errors} | "
            f"{row['over_normalized']} | {row['missed_change']} | {row['wrong_correction']} |"
        )

    lines.extend(["", "## Top Token Error Patterns", ""])
    lines.append("| count | lang | category | raw | norm | pred |")
    lines.append("|---:|---|---|---|---|---|")
    for row in top_patterns[:30]:
        lines.append(
            f"| {row['count']} | {row['lang']} | {row['category']} | "
            f"`{row['raw_token']}` | `{row['norm_token']}` | `{row['pred_token']}` |"
        )

    lines.extend(["", "## Worst Sentence Samples", ""])
    for row in worst_sentences[:20]:
        lines.extend(
            [
                f"### row {row['row_index']} ({row['lang']}), errors={row['error_count']}",
                "",
                f"- raw: `{' '.join(row['raw'])}`",
                f"- norm: `{' '.join(row['norm'])}`",
                f"- pred: `{' '.join(row['pred'])}`",
                f"- categories: `{row['categories']}`",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze full-pipeline validation errors.")
    parser.add_argument(
        "--dataset",
        default="multilexnorm2026-dataset/data/validation-00000-of-00001.parquet",
        help="Validation parquet path.",
    )
    parser.add_argument(
        "--predictions",
        default="reports/v1/predictions_validation_full_v1.json",
        help="Full-pipeline prediction JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/full_pipeline_error_analysis",
        help="Directory where analysis files will be written.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    """Locate the repository root from this script's nested tools directory."""
    for candidate in [start, *start.parents]:
        if (candidate / "pipeline.py").exists() and (candidate / "multilexnorm2026-dataset").exists():
            return candidate
    return start


def main() -> None:
    args = parse_args()
    repo_dir = find_repo_root(Path(__file__).resolve().parent)
    dataset_path = Path(args.dataset)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    if not dataset_path.is_absolute():
        dataset_path = repo_dir / dataset_path
    if not predictions_path.is_absolute():
        predictions_path = repo_dir / predictions_path
    if not output_dir.is_absolute():
        output_dir = repo_dir / output_dir

    summary = analyze(dataset_path, predictions_path, output_dir)
    print(f"Wrote analysis to: {output_dir}")
    print(f"Token accuracy: {summary['token_accuracy']:.4f}")
    print(f"Sentence exact match: {summary['sentence_exact_rate']:.4f}")
    print(f"Category counts: {summary['category_counts']}")


if __name__ == "__main__":
    main()
