"""Token-level evaluation metrics calculation for lexical normalization.

This module provides standard token-level evaluation metrics (Precision, Recall,
F1-score, and Error Reduction Rate — ERR) and features a command-line interface
(CLI) to evaluate json/jsonl prediction results directly against gold parquets.
"""

import ast
import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

# Centralized paths and import setup
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths_config
paths_config.setup_imports()

from multilexnorm_eval_package_v2 import multilexnorm_evaluator_v2 as multilexnorm_evaluator


def _to_tokens(value: Any) -> List[str]:
    """Converts a value (list, numpy array, or string representation) to a list of token strings.

    Args:
        value: The token sequence in raw, serialized, or array format.

    Returns:
        List[str]: A list of clean token strings.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(t) for t in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(t) for t in parsed]
        except (ValueError, SyntaxError):
            pass
        return value.split()
    return []


def compute_metrics(
    predictions: List[str], ground_truth: List[str], raw_tokens: List[str]
) -> Dict[str, Union[int, float]]:
    """Calculates token-level metrics: TP, FP, FN, TN, Precision, Recall, F1, and ERR.

    Delegates the metrics calculation internally to the official unified package
    `multilexnorm_evaluator` to ensure consistent and standard mathematical definitions.

    Args:
        predictions: List of predicted tokens or a nested list of sentence tokens.
        ground_truth: List of gold normalized tokens or a nested list.
        raw_tokens: List of original raw tokens or a nested list.

    Returns:
        Dict[str, Union[int, float]]: Dictionary containing both legacy metric keys
            (tp, fp, fn, tn, precision, recall, f1, err, total) and the unified ones.
    """
    # Standardize flat/nested lists to nested list of lists
    if raw_tokens and isinstance(raw_tokens[0], list):
        raw_seq = raw_tokens
        gold_seq = ground_truth
        pred_seq = predictions
    else:
        raw_seq = [raw_tokens]
        gold_seq = [ground_truth]
        pred_seq = [predictions]

    counts = multilexnorm_evaluator.compute_token_counts(raw_seq, gold_seq, pred_seq)
    eval_metrics = multilexnorm_evaluator.build_metric_dict(counts)

    # Bridge unified metrics to legacy keys for full backward compatibility
    res = {
        'tp': eval_metrics['tp'],
        'fp': eval_metrics['fp'],
        'fn': eval_metrics['fn'],
        'tn': eval_metrics['tn'],
        'precision': eval_metrics['detection_precision'],
        'recall': eval_metrics['detection_recall'],
        'f1': eval_metrics['detection_f1'],
        'err': eval_metrics['err'],
        'total': eval_metrics['total_tokens'],
    }
    
    # Expose any new metric attributes from the evaluator
    for k, v in eval_metrics.items():
        if k not in res:
            res[k] = v
            
    return res


def print_metrics(metrics: Dict[str, Union[int, float]], name: str = "") -> None:
    """Prints the calculated metrics in a clean, human-readable terminal panel.

    Args:
        metrics: The metrics dictionary returned by compute_metrics.
        name: Optional descriptor or setup name to display at the header.
    """
    print(f"\n{'='*60}")
    if name:
        print(f"  Evaluation Results (Unified): {name}")
    else:
        print("  Evaluation Results (Unified)")
    print(f"{'='*60}")
    print(f"  TP: {int(metrics['tp']):>5d}  |  FP: {int(metrics['fp']):>5d}")
    print(f"  FN: {int(metrics['fn']):>5d}  |  TN: {int(metrics['tn']):>5d}")
    print(f"{'─'*60}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ERR:       {metrics['err']:.4f}")
    print(f"{'='*60}")


def format_metrics_table(results_dict: Dict[str, Dict[str, Union[int, float]]]) -> None:
    """Formats and prints multiple evaluation results as a clean markdown-like terminal table.

    Args:
        results_dict: Mapping of setup names to their corresponding metrics dictionary.
    """
    header = f"{'Setup':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ERR':>10}"
    print(f"\n{header}")
    print("─" * 120)

    for name, metrics in results_dict.items():
        print(f"{name:<30} {int(metrics['tp']):>5d} {int(metrics['fp']):>5d} {int(metrics['fn']):>5d} {int(metrics['tn']):>5d} "
              f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['err']:>10.4f}")


def load_gold_data(gold_path: Union[str, Path]) -> Tuple[List[List[str]], List[List[str]]]:
    """Loads raw and ground truth tokens from a parquet dataset.

    Args:
        gold_path: Absolute or relative path to the gold standard parquet file.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: (raw_token_sentences, gold_token_sentences)
    """
    samples = multilexnorm_evaluator.read_parquet_samples(gold_path)
    raw, gold, _ = multilexnorm_evaluator.samples_to_parallel(samples)
    return raw, gold


def load_predictions(pred_path: Union[str, Path], num_sentences: int = -1) -> List[List[str]]:
    """Loads prediction files dynamically and standardizes them into a list of sentences.

    Args:
        pred_path: Absolute or relative path to the prediction file.
        num_sentences: Ignored helper argument for backward compatibility.

    Returns:
        List[List[str]]: Standardized list of predicted token sentences.
    """
    return multilexnorm_evaluator.load_predictions(pred_path)


def main() -> None:
    """Executes evaluation from the command line, routing to multilexnorm_evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate lexical normalization token-level predictions against gold parquet files using unified evaluator."
    )
    parser.add_argument(
        "--pred", "-p", required=True, type=str,
        help="Path to prediction JSON or JSONL file"
    )
    parser.add_argument(
        "--gold", "-g", required=True, type=str,
        help="Path to gold standard validation Parquet file (must contain 'raw' and 'norm' columns)"
    )
    parser.add_argument(
        "--name", "-n", default="", type=str,
        help="Custom setup name for metrics output header"
    )
    args = parser.parse_args()

    try:
        # 1. Load Gold Standard Data
        print(f"Loading gold standard dataset: {args.gold} ...")
        samples = multilexnorm_evaluator.read_parquet_samples(args.gold)
        raw_sents, gold_sents, lang_sents = multilexnorm_evaluator.samples_to_parallel(samples)
        print(f"Loaded {len(raw_sents)} gold sentences.")

        # 2. Load Predictions
        print(f"Loading predictions: {args.pred} ...")
        pred_sents = multilexnorm_evaluator.load_predictions(args.pred)
        print(f"Loaded {len(pred_sents)} prediction sentences.")

        # 3. Row and token count validation and alignment
        if len(pred_sents) != len(gold_sents):
            print(f"[Warning] Mismatched row counts! Predictions: {len(pred_sents)}, Gold: {len(gold_sents)}")
            limit = min(len(pred_sents), len(gold_sents))
            raw_sents = raw_sents[:limit]
            gold_sents = gold_sents[:limit]
            pred_sents = pred_sents[:limit]
            lang_sents = lang_sents[:limit]

        flat_raw = []
        flat_gold = []
        flat_pred = []
        aligned_pred_sents = []
        
        for idx, (raw, gold, pred) in enumerate(zip(raw_sents, gold_sents, pred_sents)):
            if len(pred) != len(raw):
                aligned_pred = pred[:len(raw)]
                if len(aligned_pred) < len(raw):
                    aligned_pred.extend(raw[len(aligned_pred):])
                pred = aligned_pred
            
            aligned_pred_sents.append(pred)
            flat_raw.extend(raw)
            flat_gold.extend(gold)
            flat_pred.extend(pred)

        # 4. Run unified evaluate_all to get both detailed language-wise breakdown and overall stats
        setup_name = args.name or Path(args.pred).stem
        print(f"\nRunning official evaluator pipeline for: {setup_name}")

        result = multilexnorm_evaluator.evaluate_all(
            raw=raw_sents,
            gold=gold_sents,
            pred=aligned_pred_sents,
            langs=lang_sents,
            model_name=setup_name,
            dataset_name=Path(args.gold).stem,
        )

        # 5. Print metrics using both styles
        metrics = compute_metrics(flat_pred, flat_gold, flat_raw)
        print_metrics(metrics, name=setup_name)
        
        print("\nOfficial Evaluator Summary:")
        print(json.dumps(multilexnorm_evaluator.summarize_overall_result(result), ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
