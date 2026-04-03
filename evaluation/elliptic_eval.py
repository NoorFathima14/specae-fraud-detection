"""
evaluation/elliptic_eval.py
============================
Member 2 — Phase 3: Evaluation

Metrics: ROC-AUC, Average Precision, Precision, Recall, F1 (at optimal threshold).
Optimal threshold = argmax F1 on the PR curve (important for imbalanced ~2% fraud).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve, confusion_matrix,
)


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold that maximises F1 on the labelled set."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    denom  = prec + rec + 1e-8
    f1s    = np.where(denom == 0, 0.0, 2 * prec * rec / denom)
    best_i = np.argmax(f1s[:-1])
    return float(thresholds[best_i])


def evaluate_fraud_detection(y_true: np.ndarray,
                              y_score: np.ndarray) -> dict:
    """
    Parameters
    ----------
    y_true  : 1=illicit, 0=licit  (no -1 unknowns)
    y_score : anomaly scores, higher = more anomalous

    Returns
    -------
    dict with ROC-AUC, Average-Precision, Precision, Recall,
              F1-Score, Threshold, TP, FP, TN, FN
    """
    # Inversion check: if AUC < 0.5 the scores are backwards — flip them
    raw_auc = roc_auc_score(y_true, y_score)
    if raw_auc < 0.5:
        y_score = -y_score
    roc_auc  = roc_auc_score(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)

    threshold = find_optimal_threshold(y_true, y_score)
    y_pred    = (y_score >= threshold).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "ROC-AUC"           : round(roc_auc,  4),
        "Average-Precision" : round(avg_prec, 4),
        "Precision"         : round(prec, 4),
        "Recall"            : round(rec,  4),
        "F1-Score"          : round(f1,   4),
        "Threshold"         : round(threshold, 6),
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
    }