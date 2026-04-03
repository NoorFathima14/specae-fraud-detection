"""
evaluation/noor_viz.py
======================
Noor — Phase 4: Visualisations (Member 1 responsibilities)

Plots:
  1. Training loss curve
  2. Latent space (Z_X) visualised via PCA/TSNE, coloured by anomaly score
  3. Anomaly score analysis (distribution + top-k + false positive breakdown)
  4. Temporal AUC — ROC-AUC per time step (for Temporal SpecAE)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── 1. Loss curve ─────────────────────────────────────────────────────

def plot_loss_curve(loss_history: list,
                    save_path: str = "loss_curve.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(loss_history, color="#3F51B5", lw=1.8)
    ax.set_xlabel("Epoch",  fontsize=12)
    ax.set_ylabel("Loss",   fontsize=12)
    ax.set_title("SpecAE Training Loss", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[noor_viz] Loss curve → {save_path}")


# ── 2. Latent space visualisation ────────────────────────────────────

def plot_latent_space(model,
                      X,
                      A_norm,
                      labels: np.ndarray,
                      method: str = "pca",
                      save_path: str = "latent_space.png"):
    """
    Encode nodes into Z_X, reduce to 2D via PCA or t-SNE, plot.
    Colour = label (illicit/licit/unknown). Shows how well SpecAE
    separates fraud in latent space.

    Args:
        model   : trained SpecAE
        X       : torch.Tensor (N, F)
        A_norm  : torch.Tensor (N, N)
        labels  : np.ndarray (N,)  1/0/-1
        method  : 'pca' (fast) or 'tsne' (slower, better separation)
    """
    import torch

    model.eval()
    with torch.no_grad():
        Z_X, _ = model.encode(X, A_norm)
        Z      = Z_X.cpu().numpy()

    # Dimensionality reduction
    if method == "tsne":
        from sklearn.manifold import TSNE
        Z2 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Z)
        method_label = "t-SNE"
    else:
        from sklearn.decomposition import PCA
        Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
        method_label = "PCA"

    fig, ax = plt.subplots(figsize=(10, 8))

    color_map = {0: ("#2196F3", "Licit"),
                 1: ("#F44336", "Illicit"),
                -1: ("#BDBDBD", "Unknown")}

    for label_val, (color, label_name) in color_map.items():
        mask = labels == label_val
        if mask.sum() == 0:
            continue
        # Subsample unknown to avoid overplotting
        idx = np.where(mask)[0]
        if label_val == -1 and len(idx) > 2000:
            idx = np.random.default_rng(42).choice(idx, 2000, replace=False)
        ax.scatter(Z2[idx, 0], Z2[idx, 1],
                   c=color, label=f"{label_name} (n={mask.sum():,})",
                   s=8 if label_val == -1 else 20,
                   alpha=0.5 if label_val == -1 else 0.8,
                   linewidths=0)

    ax.set_title(f"SpecAE Latent Space (Z_X) — {method_label}\n"
                 f"Elliptic Bitcoin Dataset", fontsize=13)
    ax.set_xlabel(f"{method_label} 1", fontsize=11)
    ax.set_ylabel(f"{method_label} 2", fontsize=11)
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[noor_viz] Latent space → {save_path}")


# ── 3. Anomaly score analysis ─────────────────────────────────────────

def plot_anomaly_score_analysis(anomaly_scores: np.ndarray,
                                 labels: np.ndarray,
                                 save_path: str = "score_analysis.png"):
    """
    3-panel figure:
      Left  : Score distribution by class (labelled nodes only)
      Middle : Ranked anomaly scores (top-k highlighted)
      Right  : False positive / true positive breakdown at various thresholds
    """
    labelled = labels != -1
    y_true   = labels[labelled]
    y_score  = anomaly_scores[labelled]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: distribution ───────────────────
    ax = axes[0]
    ax.hist(y_score[y_true == 0], bins=60, density=True,
            alpha=0.65, color="#2196F3", label="Licit")
    ax.hist(y_score[y_true == 1], bins=60, density=True,
            alpha=0.65, color="#F44336", label="Illicit")
    ax.set_title("Score Distribution", fontsize=12)
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # ── Panel 2: ranked scores ──────────────────
    ax = axes[1]
    sorted_idx   = np.argsort(y_score)[::-1]
    sorted_scores = y_score[sorted_idx]
    sorted_labels = y_true[sorted_idx]
    colors = ["#F44336" if l == 1 else "#2196F3" for l in sorted_labels]
    ax.bar(range(len(sorted_scores)), sorted_scores, color=colors,
           width=1.0, linewidth=0)
    ax.set_title("Nodes Ranked by Anomaly Score\n(red=illicit, blue=licit)",
                 fontsize=12)
    ax.set_xlabel("Rank"); ax.set_ylabel("Anomaly Score")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: TP/FP at thresholds ───────────
    ax = axes[2]
    thresholds = np.percentile(y_score, np.arange(70, 100, 2))
    tps, fps   = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tps.append(((pred == 1) & (y_true == 1)).sum())
        fps.append(((pred == 1) & (y_true == 0)).sum())

    x = np.arange(len(thresholds))
    ax.bar(x - 0.2, tps, 0.4, color="#4CAF50", label="True Positives (fraud caught)")
    ax.bar(x + 0.2, fps, 0.4, color="#FF9800", label="False Positives")
    ax.set_xticks(x)
    ax.set_xticklabels([f"p{int(p)}" for p in np.arange(70, 100, 2)], fontsize=8)
    ax.set_title("TP vs FP at Score Percentile Thresholds", fontsize=12)
    ax.set_xlabel("Score Percentile Threshold")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("SpecAE — Anomaly Score Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[noor_viz] Score analysis → {save_path}")


# ── 4. Temporal AUC per time step ─────────────────────────────────────

def plot_temporal_auc(per_step_metrics: dict,
                      save_path: str = "temporal_auc.png"):
    """
    Line plot of ROC-AUC across time steps.

    Args:
        per_step_metrics : dict {time_step (int) : metrics_dict}
                           where metrics_dict has key "ROC-AUC"
    """
    from sklearn.metrics import roc_auc_score

    steps = sorted(per_step_metrics.keys())
    aucs  = [per_step_metrics[s]["ROC-AUC"] for s in steps]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, aucs, color="#9C27B0", lw=2.2, marker="o", markersize=5)
    ax.axhline(0.5, color="gray", lw=1.2, linestyle="--", label="Random baseline")
    ax.fill_between(steps, 0.5, aucs,
                    where=[a > 0.5 for a in aucs],
                    alpha=0.15, color="#9C27B0", label="Above random")

    ax.set_xlabel("Time Step",  fontsize=12)
    ax.set_ylabel("ROC-AUC",    fontsize=12)
    ax.set_title("Temporal SpecAE — ROC-AUC Across Time Steps\n"
                 "(shows how fraud detectability evolves)", fontsize=13)
    ax.set_ylim([0.3, 1.0])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[noor_viz] Temporal AUC → {save_path}")