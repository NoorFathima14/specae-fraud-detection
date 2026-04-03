"""
scripts/run_elliptic.py
=======================
Member 2 — Phase 2 Entry Point: Fraud Detection on Elliptic Bitcoin Dataset

Integrates with Noor's files:
  models/specae.py              → SpecAE(input_dim, hidden_dim, latent_dim)
  training/trainer.py           → train(model, X, A_norm, ...) → loss_history
  evaluation/anomaly_score.py   → compute_anomaly_scores(model, X, A_norm, alpha)

Member 2 files:
  preprocessing/elliptic_loader.py
  evaluation/elliptic_eval.py
  evaluation/elliptic_viz.py

Usage:
  python scripts/run_elliptic.py                        # full dataset
  python scripts/run_elliptic.py --timestep 5           # single time step (quick test)
  python scripts/run_elliptic.py --timestep 5 --epochs 50
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from preprocessing.elliptic_loader import load_elliptic
from models.specae import SpecAE
from training.trainer import train
from evaluation.anomaly_score import compute_anomaly_scores, print_score_stats
from evaluation.elliptic_eval import evaluate_fraud_detection
from evaluation.elliptic_viz import (
    plot_score_distribution,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_fraud_graph_sample,
)
from evaluation.noor_viz import (
    plot_loss_curve,
    plot_latent_space,
    plot_anomaly_score_analysis,
)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SpecAE — Elliptic Fraud Detection")
    parser.add_argument("--data_dir",  default="data/elliptic_bitcoin_dataset")
    parser.add_argument("--timestep",  type=int,   default=None,
                        help="Single time step only — good for quick testing (~4k nodes)")
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--lr",        type=float, default=0.005)
    parser.add_argument("--hidden",    type=int,   default=64)
    parser.add_argument("--latent",    type=int,   default=32)
    parser.add_argument("--alpha",     type=float, default=0.5)
    parser.add_argument("--outdir",    default="results/elliptic")
    return parser.parse_args()


# ──────────────────────────────────────────────
# SPARSE → DENSE
# SpecAE uses torch.matmul(A, X) so needs dense.
# Full graph (200k nodes) = ~30 GB; use --timestep
# for development.
# ──────────────────────────────────────────────

def sparse_to_dense_tensor(A_sparse, device):
    N = A_sparse.shape[0]
    if N > 20_000:
        print(f"[warn] Dense A = {N}×{N} ({N*N*4/1e9:.1f} GB). "
              f"Use --timestep for dev runs.")
    return torch.tensor(A_sparse.toarray(), dtype=torch.float32).to(device)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    args   = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  SpecAE  —  Elliptic Bitcoin Fraud Detection")
    print(f"  device={device}  epochs={args.epochs}  alpha={args.alpha}")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/4] Loading Elliptic dataset …")
    A_sparse, X_np, labels, node_order = load_elliptic(
        data_dir  = args.data_dir,
        time_step = args.timestep,
    )
    X      = torch.tensor(X_np, dtype=torch.float32).to(device)
    A_norm = sparse_to_dense_tensor(A_sparse, device)
    print(f"      N={X.shape[0]:,}  F={X.shape[1]}")

    # ── 2. Build model ────────────────────────────────────────
    print("\n[2/4] Building SpecAE …")
    model = SpecAE(
        input_dim  = X.shape[1],   # 165
        hidden_dim = args.hidden,
        latent_dim = args.latent,
    ).to(device)

    # ── 3. Train ──────────────────────────────────────────────
    print(f"\n[3/4] Training for {args.epochs} epochs …")
    loss_history = train(
        model     = model,
        X         = X,
        A_norm    = A_norm,
        epochs    = args.epochs,
        lr        = args.lr,
        alpha     = args.alpha,
        log_every = max(1, args.epochs // 20),
    )
    # Save the trained model
    model_path = os.path.join(args.outdir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # ── 4. Anomaly scoring (Noor's function) ──────────────────
    print("\n[4/4] Computing anomaly scores …")
    scores_tensor = compute_anomaly_scores(model, X, A_norm, alpha=args.alpha)
    print_score_stats(scores_tensor)
    anomaly_scores = scores_tensor.cpu().numpy()   # (N,) np.ndarray for our eval/viz

    # ── 5. Evaluate ───────────────────────────────────────────
    print("\nEvaluating …")
    labelled_mask = labels != -1          # exclude unknowns
    y_true  = labels[labelled_mask]       # 1=illicit, 0=licit
    y_score = anomaly_scores[labelled_mask]

    metrics = evaluate_fraud_detection(y_true, y_score)
    print("\n── Results ──────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")
    print("─────────────────────────────────────────────────")

    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # ── 6. Visualise ──────────────────────────────────────────
    print("\nGenerating visualisations …")
    plot_score_distribution(
        anomaly_scores, labels,
        save_path=os.path.join(args.outdir, "score_distribution.png"))
    plot_roc_curve(
        y_true, y_score, auc=metrics["ROC-AUC"],
        save_path=os.path.join(args.outdir, "roc_curve.png"))
    plot_precision_recall_curve(
        y_true, y_score,
        save_path=os.path.join(args.outdir, "pr_curve.png"))
    plot_fraud_graph_sample(
        A=A_sparse, labels=labels,
        anomaly_scores=anomaly_scores, node_order=node_order,
        save_path=os.path.join(args.outdir, "fraud_graph_sample.png"))

    # Noor's visualisations
    plot_loss_curve(
        loss_history,
        save_path=os.path.join(args.outdir, "loss_curve.png"))
    plot_latent_space(
        model, X, A_norm, labels,
        method="pca",
        save_path=os.path.join(args.outdir, "latent_space_pca.png"))
    plot_anomaly_score_analysis(
        anomaly_scores, labels,
        save_path=os.path.join(args.outdir, "score_analysis.png"))

    print(f"\n✓ Done. Results → {args.outdir}/")
    print(f"  ROC-AUC={metrics['ROC-AUC']:.4f}  F1={metrics['F1-Score']:.4f}  "
          f"Precision={metrics['Precision']:.4f}  Recall={metrics['Recall']:.4f}")


if __name__ == "__main__":
    main()