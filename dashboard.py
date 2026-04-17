"""
dashboard.py
============
SpecAE Fraud Detection Dashboard
Loads pre-computed results from results/ by default.
Falls back to interactive training if results not found.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, json, sys

sys.path.insert(0, os.path.abspath('.'))

st.set_page_config(
    page_title="SpecAE — Fraud Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.metric-card {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 6px 0;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 32px;
    font-weight: 600;
    color: #e8e8f0;
}
.metric-value.good  { color: #4ade80; }
.metric-value.mid   { color: #fbbf24; }
.metric-value.low   { color: #f87171; }
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #7c7c9e;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #2d2d4e;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.comparison-table th {
    background: #1a1a2e;
    color: #888;
    padding: 10px 16px;
    text-align: left;
    border-bottom: 2px solid #2d2d4e;
}
.comparison-table td {
    padding: 10px 16px;
    border-bottom: 1px solid #1e1e38;
    color: #d0d0e8;
}
.comparison-table tr:hover td { background: #1a1a2e; }
.better { color: #4ade80; font-weight: 600; }
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
}
.tag-baseline   { background: #1e3a5f; color: #60a5fa; }
.tag-temporal   { background: #1a3a2e; color: #4ade80; }
.tag-innovation { background: #3a1a2e; color: #f472b6; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────

def load_metrics(path):
    """Load metrics.txt into a dict."""
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                try:
                    m[k.strip()] = float(v.strip())
                except ValueError:
                    m[k.strip()] = v.strip()
    return m

def color_class(val):
    if isinstance(val, float):
        if val >= 0.8: return "good"
        if val >= 0.5: return "mid"
        return "low"
    return ""

def metric_card(label, value):
    css = color_class(value) if isinstance(value, float) else ""
    display = f"{value:.4f}" if isinstance(value, float) else str(value)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css}">{display}</div>
    </div>""", unsafe_allow_html=True)

def show_image(path, caption=""):
    if os.path.exists(path):
        img = mpimg.imread(path)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(img)
        ax.axis("off")
        if caption:
            ax.set_title(caption, fontsize=11, color="#888")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info(f"Plot not found: `{path}`  \nRun the script to generate it.")


# ── Sidebar ────────────────────────────────────────────────────────────

st.sidebar.markdown("##  SpecAE")
st.sidebar.markdown("**Spectral Autoencoder**  \nFraud Detection on  \nElliptic Bitcoin Dataset")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    " Overview",
    " Phase 1 — Cora",
    " Phase 2 — Elliptic Baseline",
    " Innovation — Temporal",
    " Optimization",
    " Live Demo",
])
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:11px; color:#666; font-family:monospace'>
Member 1 (Noor)<br>
Member 2 (Sarnika)
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == " Overview":
    st.markdown("# SpecAE — Fraud Detection")
    st.markdown("### Spectral Autoencoder for Anomaly Detection in Attributed Networks")
    st.markdown("> *Li et al., 2019 — applied to the Elliptic Bitcoin Transaction Dataset*")

    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Transactions", "203,769")
    with c2: metric_card("Edges", "234,355")
    with c3: metric_card("Illicit", "4,545")
    with c4: metric_card("Licit", "42,019")

    st.markdown('<div class="section-header">Key Results</div>', unsafe_allow_html=True)
    baseline = load_metrics("results/elliptic/metrics.txt")
    temporal = load_metrics("results/temporal/metrics_temporal.txt")

    c1, c2, c3, c4 = st.columns(4)
    if baseline:
        with c1: metric_card("Baseline ROC-AUC", baseline.get("ROC-AUC", "—"))
        with c2: metric_card("Baseline F1", baseline.get("F1-Score", "—"))
    if temporal:
        with c3: metric_card("Temporal ROC-AUC", temporal.get("ROC-AUC", "—"))
        with c4: metric_card("Temporal F1", temporal.get("F1-Score", "—"))

    st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    ```
    Input X (N×165)  +  Adjacency A (N×N)
            │
            ▼
    GCN Layer 1: A·X·W₁  →  ReLU  →  H  (N×64)
            │
            ▼
    GCN Layer 2: A·H·W₂  →  Z  (N×32)
           / \\
          /   \\
        Z_X   Z_G          dual latent spaces
          |     |
     decoder  inner
      (X_hat) product
               (A_hat)
            │
            ▼
    Anomaly Score = α·||X−X̂||² + (1−α)·||A−Â||²
    ```
    """)

    st.markdown('<div class="section-header">Project Structure</div>', unsafe_allow_html=True)
    st.markdown("""
    | Phase | Description |
    |---|---|
    | Phase 1 | Reproduce SpecAE on Cora dataset |
    | Phase 2 | Fraud detection on Elliptic |
    | Innovation | Temporal SpecAE (sliding window) |
    | Optimization | Alpha, architecture, LR sweep |
    | Evaluation | ROC-AUC, F1, PR curve |
    | Visualization | All plots + dashboard |
    """)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — CORA
# ══════════════════════════════════════════════════════════════════════
elif page == " Phase 1 — Cora":
    st.markdown("# Phase 1 — Cora Validation")
    st.markdown("Reproducing SpecAE on Cora to validate the model works before applying to Elliptic.")

    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Nodes", "2,708")
    with c2: metric_card("Features", "1,433")
    with c3: metric_card("Final Loss", "0.1308")
    with c4: metric_card("Parameters", "293,273")

    st.markdown('<div class="section-header">Top 10 Anomalous Nodes</div>', unsafe_allow_html=True)
    st.code("[677, 442, 921, 1794, 2308, 874, 2527, 317, 92, 990]")
    st.markdown("These nodes have the highest reconstruction error — most structurally deviant from their neighbourhood.")

    st.markdown('<div class="section-header">Training Log</div>', unsafe_allow_html=True)
    epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    losses = [0.189006, 0.132255, 0.131173, 0.130959, 0.130888,
              0.130858, 0.130847, 0.130842, 0.130838, 0.130832, 0.130825]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, losses, "o-", color="#7c3aed", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Cora Training Loss", fontsize=12)
    ax.grid(alpha=0.2)
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")
    ax.tick_params(colors="#888"); ax.xaxis.label.set_color("#888"); ax.yaxis.label.set_color("#888")
    ax.title.set_color("#ccc")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d2d4e")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.success(" Phase 1 complete — model validated on Cora. Proceeding to Elliptic.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — ELLIPTIC BASELINE
# ══════════════════════════════════════════════════════════════════════
elif page == " Phase 2 — Elliptic Baseline":
    st.markdown("# Phase 2 — Elliptic Baseline")
    st.markdown("Static SpecAE trained on time step 10, 200 epochs.")

    metrics = load_metrics("results/elliptic/metrics.txt")

    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("ROC-AUC", metrics.get("ROC-AUC", 0))
        with c2: metric_card("F1-Score", metrics.get("F1-Score", 0))
        with c3: metric_card("Precision", metrics.get("Precision", 0))
        with c4: metric_card("Recall", metrics.get("Recall", 0))

        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("True Positives", int(metrics.get("TP", 0)))
        with c2: metric_card("False Positives", int(metrics.get("FP", 0)))
        with c3: metric_card("True Negatives", int(metrics.get("TN", 0)))
        with c4: metric_card("False Negatives", int(metrics.get("FN", 0)))
    else:
        st.warning("Run `python scripts/run_elliptic.py --timestep 10 --epochs 200` to generate results.")

    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Score Distribution", "ROC Curve", "PR Curve",
        "Graph Sample", "Latent Space", "Loss Curve"
    ])
    with tab1: show_image("results/elliptic/score_distribution.png", "Anomaly Score Distribution")
    with tab2: show_image("results/elliptic/roc_curve.png", "ROC Curve")
    with tab3: show_image("results/elliptic/pr_curve.png", "Precision-Recall Curve")
    with tab4: show_image("results/elliptic/fraud_graph_sample.png", "Transaction Graph Sample")
    with tab5: show_image("results/elliptic/latent_space_pca.png", "Latent Space (PCA)")
    with tab6: show_image("results/elliptic/loss_curve.png", "Training Loss")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — TEMPORAL INNOVATION
# ══════════════════════════════════════════════════════════════════════
elif page == " Innovation — Temporal":
    st.markdown("# Innovation — Temporal SpecAE")
    st.markdown("""
    **Key idea:** Standard SpecAE treats all 203k transactions as a single static graph.
    Bitcoin fraud evolves over time — patterns in step 5 differ from step 40.
    
    **Temporal SpecAE** trains on a **sliding window** of W consecutive time steps,
    aggregating scores across windows for more robust detection.
    """)

    st.markdown('<div class="section-header">Approach</div>', unsafe_allow_html=True)
    st.markdown("""
    ```
    Time steps: 1 ──────────────────────────── 49
    
    Window 1:  [1, 2, 3]
    Window 2:     [2, 3, 4]
    Window 3:        [3, 4, 5]
    ...                              47 windows total
    Window 47:                  [47, 48, 49]
    
    Per window: train fresh SpecAE → compute scores
    Final score: average across all windows a node appears in
    ```
    """)

    temporal = load_metrics("results/temporal/metrics_temporal.txt")
    baseline = load_metrics("results/elliptic/metrics.txt")

    st.markdown('<div class="section-header">Temporal Results</div>', unsafe_allow_html=True)
    if temporal:
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("ROC-AUC", temporal.get("ROC-AUC", 0))
        with c2: metric_card("F1-Score", temporal.get("F1-Score", 0))
        with c3: metric_card("Precision", temporal.get("Precision", 0))
        with c4: metric_card("Recall", temporal.get("Recall", 0))
    else:
        st.warning("Run `python scripts/run_temporal.py --window 3 --stride 1 --epochs 100` to generate results.")

    # Comparison table
    if baseline and temporal:
        st.markdown('<div class="section-header">Comparison: Static vs Temporal</div>', unsafe_allow_html=True)
        keys = ["ROC-AUC", "Average-Precision", "Precision", "Recall", "F1-Score"]
        rows = ""
        for k in keys:
            b = baseline.get(k, 0)
            t = temporal.get(k, 0)
            if isinstance(b, float) and isinstance(t, float):
                b_str = f"{b:.4f}"
                t_str = f"{t:.4f}"
                better = "better" if t > b else ""
                rows += f"<tr><td>{k}</td><td>{b_str}</td><td class='{better}'>{t_str}</td></tr>"
        st.markdown(f"""
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th><span class="tag tag-baseline">Static SpecAE</span></th>
                <th><span class="tag tag-temporal">Temporal SpecAE</span></th>
            </tr>
            {rows}
        </table>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)
    show_image("results/temporal/temporal_score_analysis.png", "Temporal Score Analysis")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════
elif page == " Optimization":
    st.markdown("# Optimization Experiments")
    st.markdown("Hyperparameter experiments to find the best SpecAE configuration.")

    tab1, tab2, tab3 = st.tabs(["Alpha Sweep", "Architecture Search", "Learning Rate"])
    with tab1:
        st.markdown("**Effect of loss weighting (α) on fraud detection quality.**  \nalpha=1.0: pure feature reconstruction loss | alpha=0.0: pure structure loss")
        show_image("results/optimization/alpha_sweep.png")
    with tab2:
        st.markdown("**ROC-AUC across different hidden × latent dimension combinations.**")
        show_image("results/optimization/arch_search.png")
    with tab3:
        st.markdown("**Effect of learning rate on final ROC-AUC.**")
        show_image("results/optimization/lr_sweep.png")

    # Load summary if exists
    summary_path = "results/optimization/optimization_summary.json"
    if os.path.exists(summary_path):
        st.markdown('<div class="section-header">Best Configuration</div>', unsafe_allow_html=True)
        with open(summary_path) as f:
            summary = json.load(f)
        if "alpha" in summary:
            best_alpha = max(summary["alpha"], key=summary["alpha"].get)
            st.metric("Best Alpha", best_alpha, f"AUC={summary['alpha'][best_alpha]:.4f}")
    else:
        st.info("Run `python scripts/optimize.py --timestep 10` to generate optimization plots.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — LIVE DEMO
# ══════════════════════════════════════════════════════════════════════
elif page == " Live Demo":
    st.markdown("# Live Demo — Interactive Training")
    st.markdown("Train SpecAE on a time step and see results in real time.")

    c1, c2 = st.columns(2)
    with c1:
        timestep = st.selectbox("Time Step", list(range(1, 50)), index=9)
        epochs   = st.slider("Epochs", 50, 300, 150)
        alpha    = st.slider("Alpha", 0.0, 1.0, 0.5, 0.1)
    with c2:
        hidden = st.slider("Hidden Dim", 32, 128, 64, 32)
        latent = st.slider("Latent Dim", 16, 64, 32, 16)
        lr     = st.select_slider("Learning Rate", [0.001, 0.005, 0.01], value=0.005)

    if st.button("▶ Train & Evaluate", type="primary"):
        import torch
        from preprocessing.elliptic_loader import load_elliptic
        from models.specae import SpecAE
        from training.trainer import train
        from evaluation.anomaly_score import compute_anomaly_scores
        from evaluation.elliptic_eval import evaluate_fraud_detection

        with st.spinner("Loading data..."):
            A_sparse, X_np, labels, node_order = load_elliptic(
                data_dir="data/elliptic_bitcoin_dataset", time_step=timestep)
            X      = torch.tensor(X_np, dtype=torch.float32)
            A_norm = torch.tensor(A_sparse.toarray(), dtype=torch.float32)

        st.info(f"N={len(node_order):,}  illicit={(labels==1).sum()}  licit={(labels==0).sum()}")

        model = SpecAE(X.shape[1], hidden, latent)
        progress = st.progress(0)
        loss_placeholder = st.empty()
        loss_history = []

        # Manual training loop with progress
        import torch.optim as optim
        import torch.nn.functional as F
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            X_hat, A_hat, _, _ = model(X, A_norm)
            loss = alpha * F.mse_loss(X_hat, X) + (1-alpha) * F.mse_loss(A_hat, A_norm)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if epoch % max(1, epochs//20) == 0:
                progress.progress((epoch+1)/epochs)
                loss_placeholder.text(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

        progress.empty(); loss_placeholder.empty()

        scores  = compute_anomaly_scores(model, X, A_norm, alpha).cpu().numpy()
        labelled = labels != -1
        y_true  = labels[labelled]; y_score = scores[labelled]
        metrics = evaluate_fraud_detection(y_true, y_score)

        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("ROC-AUC",   metrics["ROC-AUC"])
        with c2: metric_card("F1-Score",  metrics["F1-Score"])
        with c3: metric_card("Precision", metrics["Precision"])
        with c4: metric_card("Recall",    metrics["Recall"])

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(loss_history, color="#7c3aed", lw=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — Time Step {timestep}", fontsize=11)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
