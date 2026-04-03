"""
evaluation/elliptic_viz.py
===========================
Member 2 — Phase 4: Visualisation

Plots:
  1. Anomaly score distribution (illicit vs licit)
  2. ROC curve
  3. Precision-Recall curve
  4. Fraud graph sample (NetworkX, coloured by label, sized by score)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp


# ── 1. Score distribution ─────────────────────────────────────────────

def plot_score_distribution(anomaly_scores: np.ndarray,
                             labels: np.ndarray,
                             save_path: str = "score_distribution.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(anomaly_scores[labels == 0], bins=80, density=True,
            alpha=0.65, color="#2196F3", label="Licit (normal)")
    ax.hist(anomaly_scores[labels == 1], bins=80, density=True,
            alpha=0.65, color="#F44336", label="Illicit (fraud)")
    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Density",       fontsize=12)
    ax.set_title("SpecAE — Anomaly Score Distribution\n(Elliptic Bitcoin Dataset)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Score distribution → {save_path}")
    return fig


# ── 2. ROC curve ──────────────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                   auc: float, save_path: str = "roc_curve.png"):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#9C27B0", lw=2.2,
            label=f"SpecAE  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Fraud Detection", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] ROC curve → {save_path}")
    return fig


# ── 3. Precision-Recall curve ─────────────────────────────────────────

def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray,
                                 save_path: str = "pr_curve.png"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#FF5722", lw=2.2,
            label=f"SpecAE  (AP = {ap:.4f})")
    ax.axhline(y_true.mean(), color="gray", lw=1.2, linestyle="--",
               label=f"Random ({y_true.mean():.3f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Fraud Detection", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] PR curve → {save_path}")
    return fig


# ── 4. Fraud graph sample ─────────────────────────────────────────────

def plot_fraud_graph_sample(A: sp.csr_matrix,
                             labels: np.ndarray,
                             anomaly_scores: np.ndarray,
                             node_order: list,
                             save_path: str = "fraud_graph_sample.png",
                             n_sample: int = 400):
    try:
        import networkx as nx
    except ImportError:
        print("[viz] networkx not installed — skipping graph plot. pip install networkx")
        return

    # Sample: bias toward illicit so they're visible
    rng         = np.random.default_rng(42)
    illicit_idx = np.where(labels == 1)[0]
    licit_idx   = np.where(labels == 0)[0]
    unknown_idx = np.where(labels == -1)[0]

    n_ill = min(len(illicit_idx), n_sample // 4)
    n_lic = min(len(licit_idx),   n_sample // 2)
    n_unk = min(len(unknown_idx), n_sample - n_ill - n_lic)

    parts = [rng.choice(illicit_idx, n_ill, replace=False),
             rng.choice(licit_idx,   n_lic, replace=False)]
    if n_unk > 0:
        parts.append(rng.choice(unknown_idx, n_unk, replace=False))
    chosen = np.concatenate(parts).astype(int)

    sub_A      = A[np.ix_(chosen, chosen)]
    sub_labels = labels[chosen]
    sub_scores = anomaly_scores[chosen]

    G           = nx.from_scipy_sparse_array(sub_A)
    color_map   = {1: "#F44336", 0: "#2196F3", -1: "#9E9E9E"}
    node_colors = [color_map[sub_labels[i]] for i in range(len(chosen))]
    norm_scores = (sub_scores - sub_scores.min()) / (sub_scores.max() - sub_scores.min() + 1e-8)
    node_sizes  = 20 + norm_scores * 180

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.4)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15,
                           edge_color="#BBBBBB", width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes, alpha=0.85)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#F44336", label="Illicit (fraud)"),
        Patch(color="#2196F3", label="Licit (normal)"),
        Patch(color="#9E9E9E", label="Unknown"),
    ], fontsize=11, loc="upper left")
    ax.set_title(
        f"Elliptic Transaction Graph — {len(chosen)}-node sample\n"
        "(node size ∝ anomaly score)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Graph sample → {save_path}")
    return fig