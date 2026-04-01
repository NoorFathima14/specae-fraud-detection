"""
scripts/run_cora.py

Phase 1: Reproduce SpecAE on the Cora dataset.
Goal: Validate that the model runs correctly and produces anomaly scores.

Usage:
    python scripts/run_cora.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.datasets import Planetoid

from models.specae import SpecAE
from preprocessing.graph_utils import edge_index_to_adjacency, normalize_adjacency
from training.trainer import train
from evaluation.anomaly_score import compute_anomaly_scores, print_score_stats


# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 128
LATENT_DIM = 64
EPOCHS     = 100
LR         = 0.005
ALPHA      = 0.5
LOG_EVERY  = 10
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("SpecAE — Phase 1: Cora Dataset")
    print("=" * 60)

    # 1. Load Cora
    print("\n[1] Loading Cora dataset...")
    dataset = Planetoid(root="data/Cora", name="Cora")
    data = dataset[0]
    print(f"    Nodes: {data.num_nodes}  |  Features: {data.num_features}  |  Edges: {data.num_edges}")

    X = data.x  # (2708, 1433)

    # 2. Build & normalize adjacency
    print("\n[2] Building adjacency matrix...")
    A = edge_index_to_adjacency(data.edge_index, data.num_nodes)
    A_norm = normalize_adjacency(A)
    print(f"    X shape     : {X.shape}")
    print(f"    A_norm shape: {A_norm.shape}")

    # 3. Build model
    print("\n[3] Initialising SpecAE model...")
    model = SpecAE(
        input_dim  = X.shape[1],
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")

    # 4. Train
    print(f"\n[4] Training for {EPOCHS} epochs...")
    loss_history = train(
        model    = model,
        X        = X,
        A_norm   = A_norm,
        epochs   = EPOCHS,
        lr       = LR,
        alpha    = ALPHA,
        log_every= LOG_EVERY,
    )

    # 5. Anomaly scores
    print("\n[5] Computing anomaly scores...")
    scores = compute_anomaly_scores(model, X, A_norm, alpha=ALPHA)
    print_score_stats(scores)

    # 6. Top anomalies
    print("\n[6] Top 10 anomalous nodes (by index):")
    from evaluation.anomaly_score import top_k_anomalies
    top_nodes = top_k_anomalies(scores, k=10)
    print(f"    {top_nodes.tolist()}")

    print("\n✅ Phase 1 complete — model validated on Cora.\n")


if __name__ == "__main__":
    main()
