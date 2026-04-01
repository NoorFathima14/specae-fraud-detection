"""
evaluation/anomaly_score.py

Anomaly scoring and evaluation metrics for SpecAE.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def compute_anomaly_scores(
    model,
    X: torch.Tensor,
    A_norm: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute per-node anomaly scores from reconstruction errors.

    score_i = alpha * ||X_i - X_hat_i||^2  +  (1 - alpha) * ||A_i - A_hat_i||^2

    Args:
        model:  Trained SpecAE model
        X:      Node feature matrix
        A_norm: Normalized adjacency matrix
        alpha:  Weighting between feature and structure error

    Returns:
        anomaly_scores: Tensor of shape (N,)
    """
    model.eval()
    with torch.no_grad():
        X_hat, A_hat, _, _ = model(X, A_norm)

        error_x = torch.mean((X - X_hat) ** 2, dim=1)       # per-node feature error
        error_a = torch.mean((A_norm - A_hat) ** 2, dim=1)  # per-node structure error

        scores = alpha * error_x + (1 - alpha) * error_a

    return scores


def print_score_stats(scores: torch.Tensor):
    """Print basic statistics of the anomaly scores."""
    print(f"Anomaly Score Stats:")
    print(f"  Min  : {scores.min().item():.6f}")
    print(f"  Max  : {scores.max().item():.6f}")
    print(f"  Mean : {scores.mean().item():.6f}")
    print(f"  Std  : {scores.std().item():.6f}")


def evaluate(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = None,
):
    """
    Evaluate anomaly detection performance against ground-truth labels.

    Args:
        scores:    Anomaly scores, shape (N,)  — higher = more anomalous
        labels:    Ground-truth binary labels, shape (N,)  — 1 = anomaly
        threshold: Decision threshold; if None, uses median score

    Returns:
        dict with AUC, Precision, Recall, F1
    """
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    auc = roc_auc_score(labels_np, scores_np)

    if threshold is None:
        threshold = np.median(scores_np)

    preds = (scores_np >= threshold).astype(int)

    precision = precision_score(labels_np, preds, zero_division=0)
    recall    = recall_score(labels_np, preds, zero_division=0)
    f1        = f1_score(labels_np, preds, zero_division=0)

    metrics = {
        "AUC":       auc,
        "Precision": precision,
        "Recall":    recall,
        "F1":        f1,
    }

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    return metrics


def top_k_anomalies(scores: torch.Tensor, k: int = 20):
    """
    Return indices of the top-k most anomalous nodes.

    Args:
        scores: Anomaly scores, shape (N,)
        k:      Number of top anomalies to return

    Returns:
        indices: LongTensor of shape (k,)
    """
    return torch.topk(scores, k).indices
