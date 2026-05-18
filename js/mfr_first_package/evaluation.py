"""
평가 메트릭 계산: TP, FP, FN, TN, Precision, Recall, F1, ERR
"""

def compute_metrics(predictions, ground_truth, raw_tokens):
    """
    토큰 단위 평가 메트릭 계산

    Args:
        predictions: 예측된 정규화 토큰 리스트
        ground_truth: 정답 정규화 토큰 리스트
        raw_tokens: 원본 토큰 리스트 (정상 판별용)

    Returns:
        dict: TP, FP, FN, TN, precision, recall, f1, err
    """
    tp = fp = fn = tn = 0

    for pred, truth, raw in zip(predictions, ground_truth, raw_tokens):
        is_normal = (raw == truth)  # 원본이 정상인가?

        if is_normal:
            if pred == raw:
                tn += 1
            else:
                fp += 1  # 정상을 잘못 변환
        else:  # 비표준
            if pred == truth:
                tp += 1  # 올바른 변환
            else:
                fn += 1  # 변환 실패/오류

    # 메트릭 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    err = (tp - fp) / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'err': err,
        'total': tp + fp + fn + tn,
    }


def print_metrics(metrics, name=""):
    """메트릭을 보기 좋게 출력"""
    print(f"\n{'='*60}")
    if name:
        print(f"  {name}")
    print(f"{'='*60}")
    print(f"  TP: {metrics['tp']:>5d}  |  FP: {metrics['fp']:>5d}")
    print(f"  FN: {metrics['fn']:>5d}  |  TN: {metrics['tn']:>5d}")
    print(f"{'─'*60}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ERR:       {metrics['err']:.4f}")
    print(f"{'='*60}")


def format_metrics_table(results_dict):
    """여러 결과를 테이블로 포맷"""
    header = f"{'Setup':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ERR':>10}"
    print(f"\n{header}")
    print("─" * 120)

    for name, metrics in results_dict.items():
        print(f"{name:<30} {metrics['tp']:>5d} {metrics['fp']:>5d} {metrics['fn']:>5d} {metrics['tn']:>5d} "
              f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['err']:>10.4f}")
