"""
preprocessing/elliptic_loader.py
=================================
Member 2 — Elliptic Bitcoin Dataset: Full Data Pipeline

Expected files in data_dir/:
  elliptic_txs_features.csv   — no header; cols: txId, time_step, f1..f165
  elliptic_txs_edgelist.csv   — header: txId1, txId2
  elliptic_txs_classes.csv    — header: txId, class
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler


# ── 1. Raw loading ────────────────────────────────────────────────────

def load_raw_elliptic(data_dir: str):
    feat_path    = os.path.join(data_dir, "elliptic_txs_features.csv")
    edge_path    = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")

    feat_cols   = ["txId", "time_step"] + [f"f{i}" for i in range(1, 166)]
    features_df = pd.read_csv(feat_path,    header=None, names=feat_cols)
    edges_df    = pd.read_csv(edge_path)
    classes_df  = pd.read_csv(classes_path)

    print(f"[loader] Transactions : {len(features_df):,}")
    print(f"[loader] Edges        : {len(edges_df):,}")
    print(f"[loader] Labelled     : {(classes_df['class'] != 'unknown').sum():,}")
    return features_df, edges_df, classes_df


# ── 2. Label processing ───────────────────────────────────────────────

def process_labels(classes_df: pd.DataFrame) -> pd.Series:
    """'1' (illicit)→1, '2' (licit)→0, 'unknown'→-1. Indexed by txId."""
    label_map = {"1": 1, "2": 0, "unknown": -1}
    labels    = classes_df.set_index("txId")["class"].map(label_map)
    print(f"[labels] illicit={(labels==1).sum():,}  "
          f"licit={(labels==0).sum():,}  "
          f"unknown={(labels==-1).sum():,}")
    return labels


# ── 3. Feature matrix X ───────────────────────────────────────────────

def build_feature_matrix(features_df: pd.DataFrame,
                          node_order: list) -> np.ndarray:
    """
    Returns X (N, 165) float32, NaN-imputed and StandardScaler-normalised.
    """
    df   = features_df.set_index("txId")
    cols = [f"f{i}" for i in range(1, 166)]
    X    = df.loc[node_order, cols].values.astype(np.float32)

    # Column-mean imputation
    col_means        = np.nanmean(X, axis=0)
    nan_mask         = np.isnan(X)
    X[nan_mask]      = np.take(col_means, np.where(nan_mask)[1])

    X = StandardScaler().fit_transform(X)
    print(f"[features] shape={X.shape}  NaNs={np.isnan(X).sum()}")
    return X.astype(np.float32)


# ── 4. Adjacency matrix A ─────────────────────────────────────────────

def build_adjacency(edges_df: pd.DataFrame,
                     node_order: list) -> sp.csr_matrix:
    """
    Returns D^{-1/2}(A+I)D^{-1/2} sparse CSR matrix.
    """
    N       = len(node_order)
    idx_map = {txid: i for i, txid in enumerate(node_order)}

    src_raw, dst_raw = edges_df["txId1"].values, edges_df["txId2"].values
    valid = [(s in idx_map) and (d in idx_map) for s, d in zip(src_raw, dst_raw)]
    valid = np.array(valid)
    src   = np.array([idx_map[s] for s in src_raw[valid]])
    dst   = np.array([idx_map[d] for d in dst_raw[valid]])

    A = sp.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N))
    A = (A + A.T).sign()                        # symmetric, 0/1
    A = A + sp.eye(N, format="csr", dtype=np.float32)  # self-loops

    deg        = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    D          = sp.diags(d_inv_sqrt, format="csr")
    A_norm     = D @ A @ D

    assert A_norm.shape == (N, N) and A_norm.nnz > 0
    assert not np.any(np.isnan(A_norm.data))
    print(f"[adjacency] shape={A_norm.shape}  nnz={A_norm.nnz:,}")
    return A_norm


# ── 5. Master pipeline ────────────────────────────────────────────────

def load_elliptic(data_dir: str = "data/elliptic_bitcoin_dataset",
                  time_step: int = None):
    """
    Full pipeline.

    Returns
    -------
    A          : scipy.sparse.csr_matrix  (N, N)  normalised adjacency
    X          : np.ndarray float32       (N, 165)
    labels     : np.ndarray int32         (N,)    1/0/-1
    node_order : list of txIds            length N
    """
    features_df, edges_df, classes_df = load_raw_elliptic(data_dir)
    label_series = process_labels(classes_df)

    if time_step is not None:
        features_df = features_df[features_df["time_step"] == time_step]
        print(f"[loader] time_step={time_step}: {len(features_df):,} nodes")

    node_order = features_df["txId"].tolist()
    X          = build_feature_matrix(features_df, node_order)
    A          = build_adjacency(edges_df, node_order)
    labels     = np.array([label_series.get(n, -1) for n in node_order],
                           dtype=np.int32)

    print(f"\n[pipeline] ✓  N={len(node_order):,}  "
          f"illicit={(labels==1).sum():,}  licit={(labels==0).sum():,}")
    return A, X, labels, node_order