"""
models/temporal_specae.py
==========================
Noor — Innovation: Temporal SpecAE (Option B)

Extends SpecAE with sliding-window temporal training over Elliptic's 49 time steps.

Key idea:
  - Standard SpecAE treats all transactions as a static graph
  - Bitcoin fraud evolves over time — patterns in step 5 differ from step 40
  - Temporal SpecAE trains on a sliding window of W consecutive time steps,
    accumulating graph structure across the window while keeping features per-step
  - Anomaly scores are aggregated across windows → more robust detection

Reference motivation:
  Elliptic dataset paper (Weber et al., 2019) explicitly notes temporal structure
  as a key property for fraud detection.
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple

# Note: SpecAE, train, compute_anomaly_scores are imported in run_temporal.py
# and passed in to avoid circular imports (trainer.py imports from models.specae)


# ──────────────────────────────────────────────────────────────────────
# SLIDING WINDOW BUILDER
# ──────────────────────────────────────────────────────────────────────

def build_windows(all_time_steps: List[int], window_size: int = 3, stride: int = 1):
    """
    Generate sliding windows of time step indices.

    Example: time_steps=[1..49], window=3, stride=1
      → [(1,2,3), (2,3,4), ..., (47,48,49)]

    Args:
        all_time_steps : sorted list of available time step integers
        window_size    : number of consecutive steps per window
        stride         : how many steps to advance between windows

    Returns:
        List of tuples, each a window of time step indices
    """
    windows = []
    steps   = sorted(all_time_steps)
    for i in range(0, len(steps) - window_size + 1, stride):
        windows.append(tuple(steps[i : i + window_size]))
    return windows


# ──────────────────────────────────────────────────────────────────────
# PER-WINDOW DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────

def prepare_window_data(features_df,
                        edges_df,
                        label_series,
                        window_steps: Tuple[int, ...],
                        device: torch.device):
    """
    Build (A, X, labels, node_order) for a given window of time steps.

    Nodes   : union of all nodes in the window steps
    Features: mean-pooled across steps a node appears in
    Edges   : union of all edges across window steps
    Labels  : from label_series (unchanged)

    Returns
    -------
    A_dense : torch.Tensor (N, N)
    X       : torch.Tensor (N, 165)
    labels  : np.ndarray  (N,)
    node_order : list of txIds
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # ── Nodes: union across window steps ─────────────────────────
    mask        = features_df["time_step"].isin(window_steps)
    window_feat = features_df[mask].copy()

    feat_cols  = [f"f{i}" for i in range(1, 166)]

    # Mean-pool features for nodes that appear in multiple steps
    agg        = window_feat.groupby("txId")[feat_cols].mean().reset_index()
    node_order = agg["txId"].tolist()
    idx_map    = {txid: i for i, txid in enumerate(node_order)}
    N          = len(node_order)

    # ── Feature matrix ────────────────────────────────────────────
    X_raw = agg[feat_cols].values.astype(np.float32)
    col_means      = np.nanmean(X_raw, axis=0)
    nan_mask       = np.isnan(X_raw)
    X_raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    X_scaled       = StandardScaler().fit_transform(X_raw).astype(np.float32)
    X_tensor       = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # ── Adjacency: union of edges across window ───────────────────
    src_raw = edges_df["txId1"].values
    dst_raw = edges_df["txId2"].values
    valid   = np.array([(s in idx_map) and (d in idx_map)
                        for s, d in zip(src_raw, dst_raw)])
    src = np.array([idx_map[s] for s in src_raw[valid]])
    dst = np.array([idx_map[d] for d in dst_raw[valid]])

    A = sp.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N))
    A = (A + A.T).sign()
    A = A + sp.eye(N, format="csr", dtype=np.float32)

    deg        = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    D          = sp.diags(d_inv_sqrt)
    A_norm     = (D @ A @ D).toarray()
    A_tensor   = torch.tensor(A_norm, dtype=torch.float32).to(device)

    # ── Labels ────────────────────────────────────────────────────
    labels = np.array([label_series.get(n, -1) for n in node_order], dtype=np.int32)

    return A_tensor, X_tensor, labels, node_order


# ──────────────────────────────────────────────────────────────────────
# TEMPORAL SPECAE TRAINING + SCORING
# ──────────────────────────────────────────────────────────────────────

def run_temporal_specae(features_df,
                        edges_df,
                        label_series,
                        SpecAE,
                        train_fn,
                        score_fn,
                        hidden_dim: int = 64,
                        latent_dim: int = 32,
                        epochs: int = 100,
                        lr: float = 0.005,
                        alpha: float = 0.5,
                        window_size: int = 3,
                        stride: int = 1,
                        device: torch.device = torch.device("cpu")):
    """
    Full Temporal SpecAE pipeline.

    For each sliding window:
      1. Build windowed (A, X)
      2. Train a fresh SpecAE on the window
      3. Compute per-node anomaly scores
      4. Accumulate scores (nodes seen in multiple windows get averaged)

    Returns
    -------
    final_scores  : dict {txId: float}  — aggregated anomaly score per node
    final_labels  : dict {txId: int}    — label per node
    """
    all_steps = sorted(features_df["time_step"].unique().tolist())
    windows   = build_windows(all_steps, window_size=window_size, stride=stride)

    print(f"[temporal] {len(all_steps)} time steps  →  "
          f"{len(windows)} windows  (size={window_size}, stride={stride})")

    score_accumulator = {}   # txId → list of scores
    label_accumulator = {}   # txId → label

    for w_idx, window in enumerate(windows):
        print(f"\n── Window {w_idx+1}/{len(windows)}: steps {window} ──")

        A, X, labels, node_order = prepare_window_data(
            features_df, edges_df, label_series, window, device
        )
        print(f"   N={len(node_order):,}  "
              f"illicit={(labels==1).sum()}  licit={(labels==0).sum()}")

        # Fresh model per window (each window = different fraud landscape)
        model = SpecAE(
            input_dim  = X.shape[1],
            hidden_dim = hidden_dim,
            latent_dim = latent_dim,
        ).to(device)

        train_fn(model=model, X=X, A_norm=A,
              epochs=epochs, lr=lr, alpha=alpha,
              log_every=max(1, epochs // 5))

        scores_tensor = score_fn(model, X, A, alpha=alpha)
        scores_np     = scores_tensor.cpu().numpy()

        # Accumulate
        for txid, score, label in zip(node_order, scores_np, labels):
            score_accumulator.setdefault(txid, []).append(float(score))
            label_accumulator[txid] = int(label)

    # Average scores across windows
    final_scores = {txid: float(np.mean(s)) for txid, s in score_accumulator.items()}
    final_labels = label_accumulator

    print(f"\n[temporal] ✓ Done. Scored {len(final_scores):,} unique nodes.")
    return final_scores, final_labels