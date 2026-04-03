"""
scripts/optimize.py
====================
Noor — Model Optimization & Loss Engineering

Experiments:
  1. Alpha sweep         — how loss weighting (feature vs structure) affects AUC
  2. Architecture search — hidden_dim × latent_dim grid
  3. Learning rate sweep — convergence speed vs final performance

Each experiment trains multiple models and logs ROC-AUC.
Results saved to results/optimization/.

Usage:
  python scripts/optimize.py --timestep 10
  python scripts/optimize.py --timestep 10 --experiment alpha
  python scripts/optimize.py --timestep 10 --experiment arch
  python scripts/optimize.py --timestep 10 --experiment lr
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing.elliptic_loader import load_elliptic
from models.specae import SpecAE
from training.trainer import train
from evaluation.anomaly_score import compute_anomaly_scores
from evaluation.elliptic_eval import evaluate_fraud_detection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/elliptic_bitcoin_dataset")
    parser.add_argument("--timestep",   type=int, default=10,
                        help="Time step to run experiments on (single step = fast)")
    parser.add_argument("--epochs",     type=int, default=150)
    parser.add_argument("--outdir",     default="results/optimization")
    parser.add_argument("--experiment", default="all",
                        choices=["alpha", "arch", "lr", "all"])
    return parser.parse_args()


def run_single(A, X, labels, hidden_dim, latent_dim, epochs, lr, alpha, device):
    """Train one model and return ROC-AUC."""
    model = SpecAE(X.shape[1], hidden_dim, latent_dim).to(device)
    train(model=model, X=X, A_norm=A,
          epochs=epochs, lr=lr, alpha=alpha, log_every=epochs+1)  # silent
    scores   = compute_anomaly_scores(model, X, A, alpha=alpha).cpu().numpy()
    labelled = labels != -1
    if labelled.sum() < 10 or (labels[labelled] == 1).sum() == 0:
        return None   # not enough labelled data for this step
    metrics  = evaluate_fraud_detection(labels[labelled], scores[labelled])
    return metrics["ROC-AUC"]


# ── Experiment 1: Alpha sweep ─────────────────────────────────────────

def experiment_alpha(A, X, labels, epochs, device, outdir):
    print("\n── Experiment 1: Alpha Sweep ──────────────────────────")
    alphas  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    for alpha in alphas:
        auc = run_single(A, X, labels, 64, 32, epochs, 0.005, alpha, device)
        tag = f"alpha={alpha:.1f}"
        results[tag] = auc if auc else 0.0
        print(f"  {tag}  →  AUC={results[tag]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, list(results.values()), "o-", color="#3F51B5", lw=2)
    ax.axhline(0.5, color="gray", lw=1, linestyle="--")
    ax.set_xlabel("Alpha (weight on feature loss vs structure loss)", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("SpecAE — Effect of Loss Weighting (Alpha)\n"
                 "alpha=1.0: pure feature  |  alpha=0.0: pure structure", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "alpha_sweep.png"), dpi=150)
    plt.close()
    print(f"  → Saved alpha_sweep.png")
    return results


# ── Experiment 2: Architecture search ────────────────────────────────

def experiment_arch(A, X, labels, epochs, device, outdir):
    print("\n── Experiment 2: Architecture Search ──────────────────")
    hidden_dims = [32, 64, 128]
    latent_dims = [16, 32, 64]
    grid        = np.zeros((len(hidden_dims), len(latent_dims)))

    for i, h in enumerate(hidden_dims):
        for j, l in enumerate(latent_dims):
            if l >= h:   # latent must be smaller than hidden
                grid[i, j] = np.nan
                continue
            auc = run_single(A, X, labels, h, l, epochs, 0.005, 0.5, device)
            grid[i, j] = auc if auc else 0.0
            print(f"  hidden={h}  latent={l}  →  AUC={grid[i,j]:.4f}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(latent_dims))); ax.set_xticklabels(latent_dims)
    ax.set_yticks(range(len(hidden_dims))); ax.set_yticklabels(hidden_dims)
    ax.set_xlabel("Latent Dim", fontsize=12)
    ax.set_ylabel("Hidden Dim", fontsize=12)
    ax.set_title("ROC-AUC by Architecture", fontsize=13)
    for i in range(len(hidden_dims)):
        for j in range(len(latent_dims)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                        fontsize=10, color="black")
    plt.colorbar(im, ax=ax, label="ROC-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "arch_search.png"), dpi=150)
    plt.close()
    print(f"  → Saved arch_search.png")
    return grid


# ── Experiment 3: Learning rate sweep ────────────────────────────────

def experiment_lr(A, X, labels, epochs, device, outdir):
    print("\n── Experiment 3: Learning Rate Sweep ──────────────────")
    lrs     = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    results = {}

    for lr in lrs:
        auc = run_single(A, X, labels, 64, 32, epochs, lr, 0.5, device)
        results[lr] = auc if auc else 0.0
        print(f"  lr={lr}  →  AUC={results[lr]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(lrs, list(results.values()), "s-", color="#E91E63", lw=2)
    ax.axhline(0.5, color="gray", lw=1, linestyle="--")
    ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("SpecAE — Effect of Learning Rate on Fraud Detection", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lr_sweep.png"), dpi=150)
    plt.close()
    print(f"  → Saved lr_sweep.png")
    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[optimize] Loading time step {args.timestep} …")
    A_sparse, X_np, labels, node_order = load_elliptic(
        data_dir=args.data_dir, time_step=args.timestep)
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    A = torch.tensor(A_sparse.toarray(), dtype=torch.float32).to(device)

    all_results = {}

    if args.experiment in ("alpha", "all"):
        all_results["alpha"] = experiment_alpha(A, X, labels, args.epochs, device, args.outdir)

    if args.experiment in ("arch", "all"):
        all_results["arch"]  = experiment_arch(A, X, labels, args.epochs, device, args.outdir)

    if args.experiment in ("lr", "all"):
        all_results["lr"]    = experiment_lr(A, X, labels, args.epochs, device, args.outdir)

    # Save summary
    summary_path = os.path.join(args.outdir, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist()
                   for k, v in all_results.items()}, f, indent=2)
    print(f"\n✓ All results saved to {args.outdir}/")


if __name__ == "__main__":
    main()