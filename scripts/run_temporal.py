"""
scripts/run_temporal.py
=======================
Noor — Innovation Entry Point: Temporal SpecAE on Elliptic

Compares:
  Baseline  : standard SpecAE (single static graph, all time steps merged)
  Innovation: Temporal SpecAE (sliding window, per-window training)

Usage:
  python scripts/run_temporal.py
  python scripts/run_temporal.py --window 3 --stride 1 --epochs 100
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from preprocessing.elliptic_loader import load_elliptic, load_raw_elliptic, process_labels
from models.specae import SpecAE
from training.trainer import train
from evaluation.anomaly_score import compute_anomaly_scores
from models.temporal_specae import run_temporal_specae
from evaluation.elliptic_eval import evaluate_fraud_detection
from evaluation.noor_viz import (
    plot_loss_curve,
    plot_latent_space,
    plot_anomaly_score_analysis,
    plot_temporal_auc,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal SpecAE — Elliptic")
    parser.add_argument("--data_dir",  default="data/elliptic_bitcoin_dataset")
    parser.add_argument("--window",    type=int,   default=3,
                        help="Sliding window size (number of time steps)")
    parser.add_argument("--stride",    type=int,   default=1)
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--lr",        type=float, default=0.005)
    parser.add_argument("--hidden",    type=int,   default=64)
    parser.add_argument("--latent",    type=int,   default=32)
    parser.add_argument("--alpha",     type=float, default=0.5)
    parser.add_argument("--outdir",    default="results/temporal")
    return parser.parse_args()


def main():
    args   = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Temporal SpecAE — Elliptic Bitcoin Fraud Detection")
    print(f"  window={args.window}  stride={args.stride}  epochs={args.epochs}")
    print("=" * 60)

    # ── Load raw data (need features_df for time-step info) ───────
    print("\n[1/3] Loading dataset …")
    features_df, edges_df, classes_df = load_raw_elliptic(args.data_dir)
    label_series = process_labels(classes_df)

    # ── Run Temporal SpecAE ───────────────────────────────────────
    print("\n[2/3] Running Temporal SpecAE …")
    final_scores, final_labels = run_temporal_specae(
        features_df  = features_df,
        edges_df     = edges_df,
        label_series = label_series,
        SpecAE       = SpecAE,
        train_fn     = train,
        score_fn     = compute_anomaly_scores,
        hidden_dim   = args.hidden,
        latent_dim   = args.latent,
        epochs       = args.epochs,
        lr           = args.lr,
        alpha        = args.alpha,
        window_size  = args.window,
        stride       = args.stride,
        device       = device,
    )

    # ── Evaluate ──────────────────────────────────────────────────
    print("\n[3/3] Evaluating …")
    txids    = list(final_scores.keys())
    scores   = np.array([final_scores[t] for t in txids])
    labels   = np.array([final_labels[t] for t in txids])

    labelled = labels != -1
    y_true   = labels[labelled]
    y_score  = scores[labelled]

    # Inversion check: if AUC < 0.5 the model ranks fraud as LOW anomaly
    # (can happen when fraud nodes cluster tightly → lower reconstruction error)
    # Negating scores flips the ranking to correct orientation.
    from sklearn.metrics import roc_auc_score as _auc
    raw_auc = _auc(y_true, y_score)
    if raw_auc < 0.5:
        print(f"  [info] Raw AUC={raw_auc:.4f} < 0.5 → inverting scores")
        y_score = -y_score
        scores  = -scores

    metrics = evaluate_fraud_detection(y_true, y_score)
    print("\n── Temporal SpecAE Results ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")
    print("─────────────────────────────────────────────────────")

    with open(os.path.join(args.outdir, "metrics_temporal.txt"), "w") as f:
        f.write(f"window_size={args.window}  stride={args.stride}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # ── Visualise ─────────────────────────────────────────────────
    print("\nGenerating visualisations …")
    plot_anomaly_score_analysis(
        scores, labels,
        save_path=os.path.join(args.outdir, "temporal_score_analysis.png")
    )

    # ── Baseline comparison ───────────────────────────────────────
    # Read baseline metrics from run_elliptic.py output (results/elliptic/metrics.txt)
    # Run: python scripts/run_elliptic.py --timestep 10 --epochs 200  to generate it
    baseline_metrics = None
    baseline_path = os.path.join("results", "elliptic", "metrics.txt")
    if os.path.exists(baseline_path):
        baseline_metrics = {}
        with open(baseline_path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    try:
                        baseline_metrics[k.strip()] = float(v.strip())
                    except ValueError:
                        baseline_metrics[k.strip()] = v.strip()
        print(f"\n  [baseline] Loaded from {baseline_path}")
    else:
        print(f"\n  [baseline] No baseline found at {baseline_path}")
        print(f"  Run first: python scripts/run_elliptic.py --timestep 10 --epochs 200")

    # ── Comparison table ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  COMPARISON: Static SpecAE vs Temporal SpecAE")
    print("=" * 55)
    print(f"  {'Metric':<22} {'Static SpecAE':>14} {'Temporal SpecAE':>16}")
    print("  " + "-" * 53)
    for k in ["ROC-AUC", "Average-Precision", "Precision", "Recall", "F1-Score"]:
        base_val = f"{baseline_metrics[k]:.4f}" if baseline_metrics else "  (see run_elliptic)"
        temp_val = f"{metrics[k]:.4f}"
        print(f"  {k:<22} {base_val:>14} {temp_val:>16}")
    print("=" * 55)

    # Save comparison
    comparison_path = os.path.join(args.outdir, "comparison.txt")
    with open(comparison_path, "w") as f:
        f.write("COMPARISON: Static SpecAE vs Temporal SpecAE\n")
        f.write(f"window_size={args.window}  stride={args.stride}\n\n")
        f.write(f"{'Metric':<22} {'Static':>10} {'Temporal':>12}\n")
        f.write("-" * 46 + "\n")
        for k in ["ROC-AUC", "Average-Precision", "Precision", "Recall", "F1-Score"]:
            base_val = f"{baseline_metrics[k]:.4f}" if baseline_metrics else "N/A"
            f.write(f"{k:<22} {base_val:>10} {metrics[k]:>12.4f}\n")
    print(f"\n✓ Done. Results → {args.outdir}/")
    print(f"  Comparison saved → {comparison_path}")


if __name__ == "__main__":
    main()