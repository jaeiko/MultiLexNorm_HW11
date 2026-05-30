#!/usr/bin/env python3
"""
Reusable evaluator and dataset-statistics tool for MultiLexNorm2026.

This script is designed for both:
  1) official validation-00000-of-00001.parquet
  2) internal held-out validation parquet files, including the 17-language split.

Core features
-------------
- Load any parquet file containing raw / norm / lang columns.
- Validate sentence and token alignment.
- Compute dataset statistics for report tables.
- Evaluate arbitrary prediction files against a gold parquet file.
- Evaluate built-in LAI and MFR baselines.
- Report overall, macro, and language-wise metrics.
- Report detection-style precision/recall/F1 inferred from changed-token decisions.
- Optionally compute OOV and MFR coverage statistics using a train parquet file.

Prediction file formats
-----------------------
JSON:
  - [["tok1", "tok2"], ["tok3"]]
  - {"predictions": [[...], [...]]}
  - [{"pred": [...]}, {"pred": [...]}]

JSONL:
  - one token list per line
  - or one object per line containing pred / prediction / predictions / norm / normalized

TSV/TXT:
  - one sentence per line
  - tokens are TAB-separated
  - empty norm placeholders are empty TSV cells

CSV:
  - header must contain one of pred / prediction / predictions / norm / normalized
  - each cell must be a JSON list string

Example usage
-------------
# Dataset statistics
python multilexnorm_evaluator.py stats \
  --gold_parquet validation-00000-of-00001.parquet \
  --train_parquet train-00000-of-00001.parquet \
  --dataset_name official_validation \
  --out_dir outputs/stats

# Evaluate LAI baseline
python multilexnorm_evaluator.py evaluate \
  --gold_parquet validation-00000-of-00001.parquet \
  --baseline lai \
  --dataset_name official_validation \
  --out_dir outputs/results

# Evaluate MFR baseline
python multilexnorm_evaluator.py evaluate \
  --gold_parquet validation-00000-of-00001.parquet \
  --train_parquet train-00000-of-00001.parquet \
  --baseline mfr \
  --dataset_name official_validation \
  --out_dir outputs/results

# Evaluate a model prediction file
python multilexnorm_evaluator.py evaluate \
  --gold_parquet validation-00000-of-00001.parquet \
  --pred_path final_predictions.json \
  --model_name final_hybrid \
  --dataset_name official_validation \
  --out_dir outputs/results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyarrow is required to read parquet files. Install with: pip install pyarrow"
    ) from exc


PREDICTION_KEYS = ("pred", "prediction", "predictions", "norm", "normalized")


# Default language groups for the internal 17-language validation setup.
# These defaults match the split structure used in this project:
# - official12: languages already present in the original official validation file
# - missing5: held-out pseudo-validation languages added from train
DEFAULT_OFFICIAL12_LANGS = {
    "id", "ja", "ko", "th", "vi",
    "de", "en", "hr", "iden", "nl", "sl", "sr",
}

DEFAULT_MISSING5_LANGS = {"da", "es", "it", "tr", "trde"}

SUPPORTED_EVAL_GROUPS = ("all", "official12", "missing5")


class EvaluationError(ValueError):
    """Raised when input files or token alignments are invalid."""


@dataclass(frozen=True)
class TokenCounts:
    total_tokens: int
    correct_tokens: int
    lai_correct_tokens: int
    gold_changed_tokens: int
    pred_changed_tokens: int
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total_tokens": self.total_tokens,
            "correct_tokens": self.correct_tokens,
            "lai_correct_tokens": self.lai_correct_tokens,
            "gold_changed_tokens": self.gold_changed_tokens,
            "pred_changed_tokens": self.pred_changed_tokens,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def safe_div(num: float | int, den: float | int) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def round6(value: float) -> float:
    return round(float(value), 6)


def read_parquet_samples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    table = pq.read_table(path)
    required = {"raw", "norm", "lang"}
    missing = required - set(table.column_names)
    if missing:
        raise EvaluationError(
            f"Missing required columns {sorted(missing)} in {path}. "
            f"Found columns: {table.column_names}"
        )

    rows = table.to_pylist()
    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample = {
            "raw": list(row["raw"]),
            "norm": list(row["norm"]),
            "lang": str(row["lang"]),
        }
        # Preserve optional metadata columns for diagnostics, if present.
        for key, value in row.items():
            if key not in sample:
                sample[key] = value
        validate_sample(sample, idx)
        samples.append(sample)
    return samples


def validate_sample(sample: Mapping[str, Any], idx: int | None = None) -> None:
    prefix = f"sample {idx}: " if idx is not None else ""
    for key in ("raw", "norm", "lang"):
        if key not in sample:
            raise EvaluationError(f"{prefix}missing key {key!r}")
    if not isinstance(sample["raw"], list) or not all(isinstance(x, str) for x in sample["raw"]):
        raise EvaluationError(f"{prefix}raw must be list[str]")
    if not isinstance(sample["norm"], list) or not all(isinstance(x, str) for x in sample["norm"]):
        raise EvaluationError(f"{prefix}norm must be list[str]")
    if not isinstance(sample["lang"], str):
        raise EvaluationError(f"{prefix}lang must be str")
    if len(sample["raw"]) != len(sample["norm"]):
        raise EvaluationError(
            f"{prefix}token length mismatch: len(raw)={len(sample['raw'])} "
            f"!= len(norm)={len(sample['norm'])}"
        )


def samples_to_parallel(samples: Sequence[Mapping[str, Any]]) -> tuple[list[list[str]], list[list[str]], list[str]]:
    raw = [list(s["raw"]) for s in samples]
    gold = [list(s["norm"]) for s in samples]
    langs = [str(s["lang"]) for s in samples]
    return raw, gold, langs


def validate_parallel(
    raw: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]] | None = None,
    langs: Sequence[str] | None = None,
) -> None:
    if len(raw) != len(gold):
        raise EvaluationError(f"sentence mismatch: len(raw)={len(raw)} != len(gold)={len(gold)}")
    if pred is not None and len(raw) != len(pred):
        raise EvaluationError(f"sentence mismatch: len(raw)={len(raw)} != len(pred)={len(pred)}")
    if langs is not None and len(raw) != len(langs):
        raise EvaluationError(f"sentence mismatch: len(raw)={len(raw)} != len(langs)={len(langs)}")

    for i, (r, g) in enumerate(zip(raw, gold)):
        if len(r) != len(g):
            raise EvaluationError(f"token mismatch at sentence {i}: raw={len(r)}, gold={len(g)}")
        if pred is not None and len(r) != len(pred[i]):
            raise EvaluationError(f"token mismatch at sentence {i}: raw={len(r)}, pred={len(pred[i])}")
        if not all(isinstance(x, str) for x in r):
            raise EvaluationError(f"raw sentence {i} contains non-string tokens")
        if not all(isinstance(x, str) for x in g):
            raise EvaluationError(f"gold sentence {i} contains non-string tokens")
        if pred is not None and not all(isinstance(x, str) for x in pred[i]):
            raise EvaluationError(f"pred sentence {i} contains non-string tokens")


def compute_token_counts(raw: Sequence[Sequence[str]], gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> TokenCounts:
    validate_parallel(raw, gold, pred)
    total = correct = lai_correct = gold_changed = pred_changed = 0
    tp = fp = fn = tn = 0

    for raw_sent, gold_sent, pred_sent in zip(raw, gold, pred):
        for r, g, p in zip(raw_sent, gold_sent, pred_sent):
            total += 1
            g_changed = r != g
            p_changed = r != p
            is_correct = p == g
            is_lai_correct = r == g

            correct += int(is_correct)
            lai_correct += int(is_lai_correct)
            gold_changed += int(g_changed)
            pred_changed += int(p_changed)

            if g_changed and is_correct:
                tp += 1
            elif g_changed and not is_correct:
                fn += 1
                if p_changed:
                    fp += 1
            elif (not g_changed) and p_changed and not is_correct:
                fp += 1
            else:
                tn += 1

    return TokenCounts(total, correct, lai_correct, gold_changed, pred_changed, tp, fp, fn, tn)


def err_from_counts(counts: TokenCounts) -> float:
    # NOTE: (tp - fp) / (tp + fn) 의 경우에는 계산상 틀린것 같진 않은데, 
    # LLM이 "고쳤는데 틀린 경우에" 카운팅이 2번 되는 문제가 생기는듯 합니다. 

    denom = counts.total_tokens - counts.lai_correct_tokens
    return 0.0 if denom == 0 else round6((counts.correct_tokens - counts.lai_correct_tokens) / denom)


def detection_metrics_from_counts(counts: TokenCounts) -> dict[str, float | int]:
    # Treat pred_changed as the model's binary detection of "needs normalization".
    precision = safe_div(counts.tp, counts.tp + counts.fp)
    recall = safe_div(counts.tp, counts.tp + counts.fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    overnorm_rate = safe_div(counts.fp, counts.total_tokens)
    undernorm_rate = safe_div(counts.fn, counts.total_tokens)
    return {
        "detection_precision": round6(precision),
        "detection_recall": round6(recall),
        "detection_f1": round6(f1),
        "overnormalization_rate": round6(overnorm_rate),
        "undernormalization_rate": round6(undernorm_rate),
    }


def evaluate_all(
    raw: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    langs: Sequence[str],
    *,
    model_name: str,
    dataset_name: str,
    notes: str | None = None,
) -> dict[str, Any]:
    validate_parallel(raw, gold, pred, langs)
    counts = compute_token_counts(raw, gold, pred)
    overall = build_metric_dict(counts)

    by_lang: dict[str, dict[str, Any]] = {}
    lang_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, lang in enumerate(langs):
        lang_to_idx[lang].append(i)

    for lang in sorted(lang_to_idx):
        idxs = lang_to_idx[lang]
        r = [list(raw[i]) for i in idxs]
        g = [list(gold[i]) for i in idxs]
        p = [list(pred[i]) for i in idxs]
        lang_counts = compute_token_counts(r, g, p)
        by_lang[lang] = {
            "lang": lang,
            "num_sentences": len(idxs),
            **build_metric_dict(lang_counts),
        }

    result = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        **overall,
        "macro_accuracy": round6(mean([m["accuracy"] for m in by_lang.values()])),
        "macro_lai_accuracy": round6(mean([m["lai_accuracy"] for m in by_lang.values()])),
        "macro_err": round6(mean([m["err"] for m in by_lang.values()])),
        "macro_detection_precision": round6(mean([m["detection_precision"] for m in by_lang.values()])),
        "macro_detection_recall": round6(mean([m["detection_recall"] for m in by_lang.values()])),
        "macro_detection_f1": round6(mean([m["detection_f1"] for m in by_lang.values()])),
        "num_languages": len(by_lang),
        "language_metrics": by_lang,
    }
    return result


def build_metric_dict(counts: TokenCounts) -> dict[str, Any]:
    accuracy = safe_div(counts.correct_tokens, counts.total_tokens)
    lai_accuracy = safe_div(counts.lai_correct_tokens, counts.total_tokens)
    return {
        "accuracy": round6(accuracy),
        "lai_accuracy": round6(lai_accuracy),
        "err": err_from_counts(counts),
        **counts.as_dict(),
        **detection_metrics_from_counts(counts),
    }


def build_lai_predictions(raw: Sequence[Sequence[str]]) -> list[list[str]]:
    return [list(sent) for sent in raw]


def build_mfr_dictionary(train_samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for sample in train_samples:
        lang = str(sample["lang"])
        for raw_tok, norm_tok in zip(sample["raw"], sample["norm"]):
            counts[lang][raw_tok][norm_tok] += 1

    dictionary: dict[str, dict[str, str]] = {}
    for lang, raw_map in counts.items():
        dictionary[lang] = {}
        for raw_tok, norm_counter in raw_map.items():
            # deterministic tie-breaker: count desc, then norm string asc
            best_norm, _ = sorted(norm_counter.items(), key=lambda x: (-x[1], x[0]))[0]
            dictionary[lang][raw_tok] = best_norm
    return dictionary


def build_mfr_predictions(samples: Sequence[Mapping[str, Any]], mfr: Mapping[str, Mapping[str, str]]) -> list[list[str]]:
    pred: list[list[str]] = []
    for sample in samples:
        lang = str(sample["lang"])
        lang_mfr = mfr.get(lang, {})
        pred.append([lang_mfr.get(tok, tok) for tok in sample["raw"]])
    return pred


def load_predictions(path: str | Path) -> list[list[str]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return normalize_prediction_payload(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".jsonl":
        preds = []
        for line_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                if not all(isinstance(tok, str) for tok in obj):
                    raise EvaluationError(f"JSONL line {line_idx} must be list[str]")
                preds.append(list(obj))
            elif isinstance(obj, dict):
                preds.append(extract_prediction_field(obj, line_idx))
            else:
                raise EvaluationError(f"Unsupported JSONL object at line {line_idx}: {type(obj)}")
        return preds
    if suffix in {".tsv", ".txt"}:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [list(row) for row in csv.reader(f, delimiter="\t")]
    if suffix == ".csv":
        preds = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise EvaluationError("CSV prediction file has no header")
            col = next((c for c in PREDICTION_KEYS if c in reader.fieldnames), None)
            if col is None:
                raise EvaluationError(f"CSV must contain one prediction column among {PREDICTION_KEYS}")
            for i, row in enumerate(reader):
                value = json.loads(row[col])
                if not isinstance(value, list) or not all(isinstance(tok, str) for tok in value):
                    raise EvaluationError(f"CSV row {i} column {col!r} must be a JSON list[str]")
                preds.append(list(value))
        return preds
    raise EvaluationError(f"Unsupported prediction file extension: {suffix}")


def extract_prediction_field(obj: Mapping[str, Any], idx: int | None = None) -> list[str]:
    for key in PREDICTION_KEYS:
        if key in obj:
            value = obj[key]
            if not isinstance(value, list) or not all(isinstance(tok, str) for tok in value):
                where = f" at item {idx}" if idx is not None else ""
                raise EvaluationError(f"Prediction field {key!r}{where} must be list[str]")
            return list(value)
    raise EvaluationError(f"Could not find prediction field among {PREDICTION_KEYS}")


def normalize_prediction_payload(payload: Any) -> list[list[str]]:
    if isinstance(payload, list):
        if all(isinstance(item, list) for item in payload):
            out = []
            for i, sent in enumerate(payload):
                if not all(isinstance(tok, str) for tok in sent):
                    raise EvaluationError(f"Prediction sentence {i} must be list[str]")
                out.append(list(sent))
            return out
        if all(isinstance(item, dict) for item in payload):
            return [extract_prediction_field(item, i) for i, item in enumerate(payload)]
    if isinstance(payload, dict):
        for key in PREDICTION_KEYS:
            if key in payload:
                return normalize_prediction_payload(payload[key])
    raise EvaluationError("Unsupported JSON prediction structure")


def dataset_has_gold(samples: Sequence[Mapping[str, Any]]) -> bool:
    return any(tok != "" for sample in samples for tok in sample["norm"])


def compute_dataset_stats(
    samples: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    train_samples: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_raw_vocab: dict[str, set[str]] = defaultdict(set)
    train_mfr: dict[str, dict[str, str]] = {}
    if train_samples is not None:
        for sample in train_samples:
            train_raw_vocab[str(sample["lang"])].update(sample["raw"])
        train_mfr = build_mfr_dictionary(train_samples)

    lang_to_samples: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        lang_to_samples[str(sample["lang"])].append(sample)

    rows: list[dict[str, Any]] = []
    for lang in sorted(lang_to_samples):
        rows.append(compute_stats_for_subset(
            lang_to_samples[lang],
            dataset_name=dataset_name,
            lang=lang,
            train_raw_vocab=train_raw_vocab.get(lang) if train_samples is not None else None,
            train_mfr=train_mfr.get(lang) if train_samples is not None else None,
        ))

    overall = aggregate_stats_rows(rows, dataset_name=dataset_name)
    return rows, overall


def compute_stats_for_subset(
    subset: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    lang: str,
    train_raw_vocab: set[str] | None = None,
    train_mfr: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    lengths = [len(s["raw"]) for s in subset]
    raw_tokens = [tok for s in subset for tok in s["raw"]]
    norm_tokens = [tok for s in subset for tok in s["norm"]]
    token_pairs = [(r, n) for s in subset for r, n in zip(s["raw"], s["norm"])]
    changed_pairs = [(r, n) for r, n in token_pairs if r != n]
    total = len(token_pairs)

    empty_norm = sum(1 for _, n in token_pairs if n == "")
    norm_contains_space = sum(1 for r, n in token_pairs if r != n and " " in n)
    raw_contains_space = sum(1 for r, n in token_pairs if r != n and " " in r)

    row: dict[str, Any] = {
        "dataset": dataset_name,
        "lang": lang,
        "num_samples": len(subset),
        "total_tokens": total,
        "avg_sentence_len": round6(mean(lengths)) if lengths else 0.0,
        "max_sentence_len": max(lengths) if lengths else 0,
        "changed_tokens": len(changed_pairs),
        "changed_ratio": round6(safe_div(len(changed_pairs), total)),
        "lai_accuracy": round6(1.0 - safe_div(len(changed_pairs), total)),
        "raw_vocab_size": len(set(raw_tokens)),
        "norm_vocab_size": len(set(norm_tokens)),
        "empty_norm_tokens": empty_norm,
        "empty_norm_ratio": round6(safe_div(empty_norm, total)),
        "norm_contains_space_tokens": norm_contains_space,
        "norm_contains_space_ratio": round6(safe_div(norm_contains_space, total)),
        "raw_contains_space_tokens": raw_contains_space,
    }

    if train_raw_vocab is not None:
        oov = sum(1 for tok in raw_tokens if tok not in train_raw_vocab)
        changed_oov = sum(1 for r, n in changed_pairs if r not in train_raw_vocab)
        row.update({
            "raw_oov_tokens": oov,
            "raw_oov_ratio": round6(safe_div(oov, total)),
            "changed_raw_oov_tokens": changed_oov,
            "changed_raw_oov_ratio": round6(safe_div(changed_oov, max(1, len(changed_pairs)))),
            "train_raw_vocab_size": len(train_raw_vocab),
        })

    if train_mfr is not None:
        seen = sum(1 for r, _ in token_pairs if r in train_mfr)
        mfr_correct = sum(1 for r, n in token_pairs if train_mfr.get(r, r) == n)
        changed_seen = sum(1 for r, n in changed_pairs if r in train_mfr)
        changed_mfr_correct = sum(1 for r, n in changed_pairs if train_mfr.get(r, r) == n)
        row.update({
            "mfr_seen_tokens": seen,
            "mfr_seen_ratio": round6(safe_div(seen, total)),
            "mfr_token_accuracy": round6(safe_div(mfr_correct, total)),
            "mfr_changed_seen_tokens": changed_seen,
            "mfr_changed_seen_ratio": round6(safe_div(changed_seen, max(1, len(changed_pairs)))),
            "mfr_changed_correct_tokens": changed_mfr_correct,
            "mfr_changed_correct_ratio": round6(safe_div(changed_mfr_correct, max(1, len(changed_pairs)))),
        })
    return row


def aggregate_stats_rows(rows: Sequence[Mapping[str, Any]], *, dataset_name: str) -> dict[str, Any]:
    if not rows:
        return {"dataset": dataset_name, "lang": "__overall__"}
    total_tokens = sum(int(r["total_tokens"]) for r in rows)
    total_samples = sum(int(r["num_samples"]) for r in rows)
    changed_tokens = sum(int(r["changed_tokens"]) for r in rows)
    empty_norm = sum(int(r["empty_norm_tokens"]) for r in rows)
    space_norm = sum(int(r["norm_contains_space_tokens"]) for r in rows)
    max_len = max(int(r["max_sentence_len"]) for r in rows)
    weighted_avg_len = safe_div(sum(float(r["avg_sentence_len"]) * int(r["num_samples"]) for r in rows), total_samples)

    out: dict[str, Any] = {
        "dataset": dataset_name,
        "lang": "__overall__",
        "num_languages": len(rows),
        "num_samples": total_samples,
        "total_tokens": total_tokens,
        "avg_sentence_len": round6(weighted_avg_len),
        "max_sentence_len": max_len,
        "changed_tokens": changed_tokens,
        "changed_ratio": round6(safe_div(changed_tokens, total_tokens)),
        "lai_accuracy": round6(1.0 - safe_div(changed_tokens, total_tokens)),
        "empty_norm_tokens": empty_norm,
        "empty_norm_ratio": round6(safe_div(empty_norm, total_tokens)),
        "norm_contains_space_tokens": space_norm,
        "norm_contains_space_ratio": round6(safe_div(space_norm, total_tokens)),
        "macro_changed_ratio": round6(mean([float(r["changed_ratio"]) for r in rows])),
        "macro_lai_accuracy": round6(mean([float(r["lai_accuracy"]) for r in rows])),
    }

    optional_sum_fields = [
        "raw_oov_tokens", "changed_raw_oov_tokens", "mfr_seen_tokens",
        "mfr_changed_seen_tokens", "mfr_changed_correct_tokens",
    ]
    for field in optional_sum_fields:
        if field in rows[0]:
            out[field] = sum(int(r[field]) for r in rows)

    if "raw_oov_tokens" in out:
        out["raw_oov_ratio"] = round6(safe_div(out["raw_oov_tokens"], total_tokens))
        out["changed_raw_oov_ratio"] = round6(safe_div(out["changed_raw_oov_tokens"], changed_tokens))
        out["macro_raw_oov_ratio"] = round6(mean([float(r["raw_oov_ratio"]) for r in rows]))

    if "mfr_seen_tokens" in out:
        out["mfr_seen_ratio"] = round6(safe_div(out["mfr_seen_tokens"], total_tokens))
        out["mfr_changed_seen_ratio"] = round6(safe_div(out["mfr_changed_seen_tokens"], changed_tokens))
        out["mfr_changed_correct_ratio"] = round6(safe_div(out["mfr_changed_correct_tokens"], changed_tokens))
        out["macro_mfr_changed_correct_ratio"] = round6(mean([float(r["mfr_changed_correct_ratio"]) for r in rows]))
    return out


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_language_metrics(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for lang, metrics in result.get("language_metrics", {}).items():
        rows.append({
            "dataset_name": result.get("dataset_name"),
            "eval_group": result.get("eval_group"),
            "model_name": result.get("model_name"),
            **metrics,
        })
    return rows


def summarize_overall_result(result: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"language_metrics", "created_at_utc", "notes"}
    row = {k: v for k, v in result.items() if k not in excluded and not isinstance(v, (dict, list))}
    row["notes"] = result.get("notes")
    row["created_at_utc"] = result.get("created_at_utc")
    return row



def parse_lang_set(value: str | None, *, default: set[str]) -> set[str]:
    """
    Parse a comma-separated language list.

    Examples
    --------
    "en,ko,ja" -> {"en", "ko", "ja"}

    If value is None or empty, returns the provided default set.
    """
    if value is None or not value.strip():
        return set(default)
    return {item.strip() for item in value.split(",") if item.strip()}


def resolve_eval_groups(
    requested_groups: Sequence[str],
    *,
    official_langs: set[str],
    missing_langs: set[str],
) -> dict[str, set[str] | None]:
    """
    Resolve group names into language filters.

    Returns
    -------
    dict[str, set[str] | None]
        None means no filtering (all languages).
    """
    invalid = [group for group in requested_groups if group not in SUPPORTED_EVAL_GROUPS]
    if invalid:
        raise EvaluationError(
            f"Unsupported eval group(s): {invalid}. Supported: {SUPPORTED_EVAL_GROUPS}"
        )

    resolved: dict[str, set[str] | None] = {}
    for group in requested_groups:
        if group == "all":
            resolved[group] = None
        elif group == "official12":
            resolved[group] = set(official_langs)
        elif group == "missing5":
            resolved[group] = set(missing_langs)
    return resolved


def filter_parallel_by_lang(
    raw: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    langs: Sequence[str],
    target_langs: set[str] | None,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[str]]:
    """
    Filter sentence-level raw/gold/pred/lang arrays by language.

    target_langs=None means return all sentences.
    This preserves the original sentence order, which is essential when one
    expensive 17-language LLM prediction file is evaluated in multiple views.
    """
    validate_parallel(raw, gold, pred, langs)
    if target_langs is None:
        return (
            [list(sent) for sent in raw],
            [list(sent) for sent in gold],
            [list(sent) for sent in pred],
            list(langs),
        )

    indices = [idx for idx, lang in enumerate(langs) if lang in target_langs]
    return (
        [list(raw[idx]) for idx in indices],
        [list(gold[idx]) for idx in indices],
        [list(pred[idx]) for idx in indices],
        [langs[idx] for idx in indices],
    )


def save_single_evaluation_result(
    result: Mapping[str, Any],
    *,
    out_dir: str | Path,
    stem: str,
) -> None:
    out_dir = Path(out_dir)
    save_json(out_dir / f"{stem}.json", result)
    save_csv(out_dir / f"{stem}_overall.csv", [summarize_overall_result(result)])
    save_csv(out_dir / f"{stem}_language_metrics.csv", flatten_language_metrics(result))


def evaluate_groups_and_save(
    raw: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    langs: Sequence[str],
    *,
    model_name: str,
    dataset_name: str,
    requested_groups: Sequence[str],
    official_langs: set[str],
    missing_langs: set[str],
    out_dir: str | Path,
    notes: str | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate the same prediction file from multiple language-filtered views.

    This is intended for expensive LLM experiments: run inference once on the
    17-language validation set, then compute metrics for all_17, official12,
    and missing5 without running the model again.
    """
    groups = resolve_eval_groups(
        requested_groups,
        official_langs=official_langs,
        missing_langs=missing_langs,
    )

    results: list[dict[str, Any]] = []
    out_dir = Path(out_dir)
    for group_name, target_langs in groups.items():
        group_raw, group_gold, group_pred, group_langs = filter_parallel_by_lang(
            raw, gold, pred, langs, target_langs
        )
        if not group_raw:
            print(
                f"[WARN] eval_group={group_name!r} selected zero sentences. "
                "Skipping this group."
            )
            continue

        group_dataset_name = f"{dataset_name}_{group_name}"
        group_notes = notes
        if group_notes:
            group_notes = f"{group_notes} | eval_group={group_name}"
        else:
            group_notes = f"eval_group={group_name}"

        result = evaluate_all(
            group_raw,
            group_gold,
            group_pred,
            group_langs,
            model_name=model_name,
            dataset_name=group_dataset_name,
            notes=group_notes,
        )
        result["eval_group"] = group_name
        result["selected_languages"] = sorted(set(group_langs))
        result["selected_num_sentences"] = len(group_raw)

        stem = f"{dataset_name}_{group_name}_{model_name}".replace(" ", "_")
        save_single_evaluation_result(result, out_dir=out_dir, stem=stem)
        results.append(result)

    if results:
        combined_stem = f"{dataset_name}_{model_name}_eval_groups".replace(" ", "_")
        save_csv(out_dir / f"{combined_stem}_overall.csv", [summarize_overall_result(r) for r in results])
        save_json(out_dir / f"{combined_stem}.json", {"results": results})

    return results

def command_stats(args: argparse.Namespace) -> None:
    samples = read_parquet_samples(args.gold_parquet)
    train_samples = read_parquet_samples(args.train_parquet) if args.train_parquet else None
    rows, overall = compute_dataset_stats(samples, dataset_name=args.dataset_name, train_samples=train_samples)

    out_dir = Path(args.out_dir)
    save_csv(out_dir / f"{args.dataset_name}_stats_by_lang.csv", rows)
    save_csv(out_dir / f"{args.dataset_name}_stats_overall.csv", [overall])
    save_json(out_dir / f"{args.dataset_name}_stats.json", {"overall": overall, "by_language": rows})

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"Saved stats to: {out_dir.resolve()}")


def command_evaluate(args: argparse.Namespace) -> None:
    samples = read_parquet_samples(args.gold_parquet)
    if not dataset_has_gold(samples):
        raise EvaluationError("Gold parquet does not contain meaningful norm labels; cannot evaluate.")
    raw, gold, langs = samples_to_parallel(samples)

    if args.baseline == "lai":
        pred = build_lai_predictions(raw)
        model_name = args.model_name or "LAI"
    elif args.baseline == "mfr":
        if not args.train_parquet:
            raise EvaluationError("--train_parquet is required for --baseline mfr")
        train_samples = read_parquet_samples(args.train_parquet)
        mfr = build_mfr_dictionary(train_samples)
        pred = build_mfr_predictions(samples, mfr)
        model_name = args.model_name or "MFR"
    else:
        if not args.pred_path:
            raise EvaluationError("Provide --pred_path or --baseline")
        pred = load_predictions(args.pred_path)
        model_name = args.model_name or Path(args.pred_path).stem

    validate_parallel(raw, gold, pred, langs)

    official_langs = parse_lang_set(args.official_langs, default=DEFAULT_OFFICIAL12_LANGS)
    missing_langs = parse_lang_set(args.missing_langs, default=DEFAULT_MISSING5_LANGS)

    results = evaluate_groups_and_save(
        raw,
        gold,
        pred,
        langs,
        model_name=model_name,
        dataset_name=args.dataset_name,
        requested_groups=args.eval_groups,
        official_langs=official_langs,
        missing_langs=missing_langs,
        out_dir=args.out_dir,
        notes=args.notes,
    )

    print(json.dumps([summarize_overall_result(r) for r in results], ensure_ascii=False, indent=2))
    print(f"Saved evaluation to: {Path(args.out_dir).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MultiLexNorm2026 evaluator and dataset statistics tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="Compute dataset statistics for a gold parquet file")
    p_stats.add_argument("--gold_parquet", required=True, help="Gold parquet file with raw/norm/lang columns")
    p_stats.add_argument("--train_parquet", default=None, help="Optional train parquet for OOV and MFR coverage stats")
    p_stats.add_argument("--dataset_name", required=True, help="Name used in output file names")
    p_stats.add_argument("--out_dir", default="outputs/stats", help="Output directory")
    p_stats.set_defaults(func=command_stats)

    p_eval = sub.add_parser("evaluate", help="Evaluate a prediction file or built-in baseline")
    p_eval.add_argument("--gold_parquet", required=True, help="Gold parquet file with raw/norm/lang columns")
    p_eval.add_argument("--pred_path", default=None, help="Prediction file path")
    p_eval.add_argument("--baseline", choices=["lai", "mfr"], default=None, help="Evaluate a built-in baseline")
    p_eval.add_argument("--train_parquet", default=None, help="Train parquet; required for MFR baseline")
    p_eval.add_argument("--dataset_name", required=True, help="Name used in output file names")
    p_eval.add_argument("--model_name", default=None, help="Model name stored in outputs")
    p_eval.add_argument("--notes", default=None, help="Optional notes stored in JSON/CSV outputs")
    p_eval.add_argument("--out_dir", default="outputs/results", help="Output directory")
    p_eval.add_argument(
        "--eval_groups",
        nargs="+",
        default=["all"],
        choices=list(SUPPORTED_EVAL_GROUPS),
        help=(
            "Language-filtered evaluation views to compute from the same prediction file. "
            "Use 'all official12 missing5' for one-shot 17-language LLM evaluation."
        ),
    )
    p_eval.add_argument(
        "--official_langs",
        default=None,
        help=(
            "Optional comma-separated language codes for the official12 group. "
            "Default: id,ja,ko,th,vi,de,en,hr,iden,nl,sl,sr"
        ),
    )
    p_eval.add_argument(
        "--missing_langs",
        default=None,
        help=(
            "Optional comma-separated language codes for the missing5 group. "
            "Default: da,es,it,tr,trde"
        ),
    )
    p_eval.set_defaults(func=command_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
