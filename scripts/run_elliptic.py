"""
scripts/run_elliptic.py

Phase 2: Fraud detection on the Elliptic Bitcoin Transaction dataset.
Lead: Member 2 (data pipeline + evaluation)

Dataset:
    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
    Place downloaded CSV files inside data/elliptic/

Expected files:
    data/elliptic/
        elliptic_txs_features.csv
        elliptic_txs_edgelist.csv
        elliptic_txs_classes.csv

Usage:
    python scripts/run_elliptic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from models.specae import SpecAE
from preprocessing.graph_utils import normalize_adjacency
from training.trainer import train
from evaluation.anomaly_score import compute_anomaly_scores, print_score_stats, evaluate


# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_DIM  = 256
LATENT_DIM  = 64
EPOCHS      = 100
LR          = 0.005
ALPHA       = 0.5
LOG_EVERY   = 10
DATA_DIR    = "data/elliptic"
# ──────────────────────────────────────────────────────────────────────────────


def load_elliptic(data_dir: str):
    """
    Load and preprocess the Elliptic dataset.

    Returns:
        X:      Feature tensor (N, F)
        A_norm: Normalized adjacency tensor (N, N)
        labels: Binary labels tensor — 1=illicit (anomaly), 0=licit (normal)
                Nodes with unknown class are excluded from evaluation only.
        node_ids: list of node ids in order
    """
    print("    Loading features...")
    features_df = pd.read_csv(
        os.path.join(data_dir, "elliptic_txs_features.csv"), header=None
    )
    # Column 0 = node id, column 1 = time step, columns 2..end = features
    node_ids = features_df.iloc[:, 0].tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    X_raw = features_df.iloc[:, 1:].values.astype(np.float32)
    scaler = StandardScaler()
    X_raw = scaler.fit_transform(X_raw)

    print("    Loading edges...")
    edges_df = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))
    N = len(node_ids)
    A = np.zeros((N, N), dtype=np.float32)
    for _, row in edges_df.iterrows():
        src, dst = row.iloc[0], row.iloc[1]
        if src in id_to_idx and dst in id_to_idx:
            i, j = id_to_idx[src], id_to_idx[dst]
            A[i, j] = 1.0
            A[j, i] = 1.0  # undirected

    print("    Loading labels...")
    classes_df = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))
    label_map = {"illicit": 1, "licit": 0, "unknown": -1}
    id_to_label = {
        row["txId"]: label_map.get(row["class"], -1)
        for _, row in classes_df.iterrows()
    }
    labels_raw = np.array([id_to_label.get(nid, -1) for nid in node_ids])

    X_tensor = torch.tensor(X_raw)
    A_norm   = normalize_adjacency(torch.tensor(A))
    labels   = torch.tensor(labels_raw, dtype=torch.long)

    return X_tensor, A_norm, labels, node_ids


def main():
    print("=" * 60)
    print("SpecAE — Phase 2: Elliptic Fraud Detection")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading Elliptic dataset...")
    X, A_norm, labels, node_ids = load_elliptic(DATA_DIR)
    print(f"    Nodes   : {X.shape[0]}")
    print(f"    Features: {X.shape[1]}")
    known_mask = labels != -1
    n_illicit = (labels == 1).sum().item()
    n_licit   = (labels == 0).sum().item()
    print(f"    Illicit (anomaly): {n_illicit}  |  Licit (normal): {n_licit}")

    # 2. Build model
    print("\n[2] Initialising SpecAE model...")
    model = SpecAE(
        input_dim  = X.shape[1],
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
    )

    # 3. Train (unsupervised — no labels used)
    print(f"\n[3] Training for {EPOCHS} epochs (unsupervised)...")
    loss_history = train(
        model     = model,
        X         = X,
        A_norm    = A_norm,
        epochs    = EPOCHS,
        lr        = LR,
        alpha     = ALPHA,
        log_every = LOG_EVERY,
    )

    # 4. Anomaly scores
    print("\n[4] Computing anomaly scores...")
    scores = compute_anomaly_scores(model, X, A_norm, alpha=ALPHA)
    print_score_stats(scores)

    # 5. Evaluate on labelled nodes only
    print("\n[5] Evaluating on labelled nodes...")
    evaluate(scores[known_mask], labels[known_mask].float())

    print("\n✅ Phase 2 complete.\n")


if __name__ == "__main__":
    main()
