import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import sys, os

# Add project path
sys.path.insert(0, os.path.abspath('.'))

from preprocessing.elliptic_loader import load_elliptic
from models.specae import SpecAE
from training.trainer import train
from evaluation.elliptic_eval import evaluate_fraud_detection
from evaluation.elliptic_viz import (
    plot_score_distribution,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_fraud_graph_sample,
)

st.set_page_config(page_title="SpecAE Fraud Detection Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Loading", "Fraud Detection", "Results", "Prediction"])

# Global variables for storing results
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scores' not in st.session_state:
    st.session_state.scores = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

def load_data(timestep=None):
    """Load Elliptic dataset"""
    try:
        A_sparse, X_np, labels, node_order = load_elliptic(
            data_dir="data/elliptic_bitcoin_dataset",
            time_step=timestep,
        )
        return A_sparse, X_np, labels, node_order
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def train_model(X, A_norm, epochs, lr, hidden, latent, alpha):
    """Train the SpecAE model"""
    model = SpecAE(input_dim=X.shape[1], hidden_dim=hidden, latent_dim=latent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = X.to(device)
    A_norm = A_norm.to(device)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Custom progress callback
    def progress_callback(epoch, loss):
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
    
    loss_history = train(
        model=model,
        X=X,
        A_norm=A_norm,
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        log_every=max(1, epochs // 20),
    )
    
    progress_bar.empty()
    status_text.empty()
    return model, loss_history

def compute_scores(model, X, A_norm, alpha):
    """Compute anomaly scores"""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    A_norm = A_norm.to(device)
    
    with torch.no_grad():
        X_hat, A_hat, _, _ = model(X, A_norm)
        feat_err = ((X - X_hat) ** 2).mean(dim=1)
        struct_err = ((A_norm - A_hat) ** 2).mean(dim=1)
        scores = alpha * feat_err + (1 - alpha) * struct_err
    
    return scores.cpu().numpy()

# Page 1: Data Overview
if page == "Data Overview":
    st.title("Data Overview")
    st.write("Explore the Elliptic Bitcoin Dataset")
    
    timestep = st.selectbox("Select timestep (None for full dataset)", [None] + list(range(1, 50)), index=5)
    
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            A_sparse, X_np, labels, node_order = load_data(timestep)
            
            if A_sparse is not None:
                st.success("Data loaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Nodes", f"{X_np.shape[0]:,}")
                with col2:
                    st.metric("Features", X_np.shape[1])
                with col3:
                    labelled = np.sum(labels != -1)
                    st.metric("Labelled Nodes", f"{labelled:,}")
                
                # Label distribution
                unique, counts = np.unique(labels[labels != -1], return_counts=True)
                label_counts = dict(zip(unique, counts))
                st.subheader("Label Distribution")
                st.bar_chart(pd.DataFrame({
                    'Label': ['Licit' if k == 0 else 'Illicit' for k in label_counts.keys()],
                    'Count': list(label_counts.values())
                }).set_index('Label'))
                
                # Feature statistics
                st.subheader("Feature Statistics")
                st.dataframe(pd.DataFrame(X_np).describe())

# Page 2: Model Setup
elif page == "Model Loading":
    st.title("Model Setup")
    st.write("Train a new model or load a pre-trained one")
    
    model_option = st.radio("Choose option:", ["Train New Model", "Load Pre-trained Model"])
    
    timestep = st.selectbox("Timestep (for data loading)", [None] + list(range(1, 50)), index=5)
    
    if model_option == "Train New Model":
        st.subheader("Training Configuration")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 10, 200, 50)
            lr = st.slider("Learning Rate", 0.001, 0.01, 0.005, 0.001)
        
        with col2:
            hidden = st.slider("Hidden Dimension", 32, 128, 64)
            latent = st.slider("Latent Dimension", 16, 64, 32)
        
        alpha = st.slider("Alpha (feature vs structure)", 0.0, 1.0, 0.5, 0.1)
        
        if st.button("Train Model"):
            with st.spinner("Loading data..."):
                A_sparse, X_np, labels, node_order = load_data(timestep)
            
            if A_sparse is not None:
                X = torch.tensor(X_np, dtype=torch.float32)
                
                # Convert sparse to dense (with warning)
                N = A_sparse.shape[0]
                if N > 10000:
                    st.warning(f"Converting {N}x{N} sparse matrix to dense. This may take time and memory.")
                
                A_norm = torch.tensor(A_sparse.toarray(), dtype=torch.float32)
                
                st.subheader("Training Progress")
                model, loss_history = train_model(X, A_norm, epochs, lr, hidden, latent, alpha)
                
                # Store results
                st.session_state.model = model
                st.session_state.labels = labels
                
                # Compute scores
                scores = compute_scores(model, X, A_norm, alpha)
                st.session_state.scores = scores
                
                # Compute metrics
                labelled_mask = labels != -1
                y_true = labels[labelled_mask]
                y_score = scores[labelled_mask]
                metrics = evaluate_fraud_detection(y_true, y_score)
                st.session_state.metrics = metrics
                
                st.success("Training completed!")
                
                # Plot loss history
                fig, ax = plt.subplots()
                ax.plot(loss_history)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                st.pyplot(fig)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
                with col2:
                    st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
                with col3:
                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                with col4:
                    st.metric("Recall", f"{metrics['Recall']:.4f}")
    
    else:  # Load Pre-trained Model
        st.subheader("Load Pre-trained Model")
        model_file = st.file_uploader("Upload model file (.pth)", type="pth")
        hidden = st.number_input("Hidden Dimension (must match training)", value=64, min_value=32, max_value=128)
        latent = st.number_input("Latent Dimension (must match training)", value=32, min_value=16, max_value=64)
        alpha = st.slider("Alpha for scoring", 0.0, 1.0, 0.5, 0.1)
        
        if model_file is not None and st.button("Load Model"):
            with st.spinner("Loading data..."):
                A_sparse, X_np, labels, node_order = load_data(timestep)
            
            if A_sparse is not None:
                X = torch.tensor(X_np, dtype=torch.float32)
                A_norm = torch.tensor(A_sparse.toarray(), dtype=torch.float32)
                
                # Load model
                model = SpecAE(input_dim=X.shape[1], hidden_dim=hidden, latent_dim=latent)
                model.load_state_dict(torch.load(model_file, map_location='cpu'))
                model.eval()
                
                st.session_state.model = model
                st.session_state.labels = labels
                
                # Compute scores
                scores = compute_scores(model, X, A_norm, alpha)
                st.session_state.scores = scores
                
                # Compute metrics
                labelled_mask = labels != -1
                y_true = labels[labelled_mask]
                y_score = scores[labelled_mask]
                metrics = evaluate_fraud_detection(y_true, y_score)
                st.session_state.metrics = metrics
                
                st.success("Model loaded and evaluated successfully!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
                with col2:
                    st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
                with col3:
                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                with col4:
                    st.metric("Recall", f"{metrics['Recall']:.4f}")

# Page 3: Fraud Detection
elif page == "Fraud Detection":
    st.title("Fraud Detection & Flagging")
    
    if st.session_state.model is None or st.session_state.scores is None:
        st.warning("Please load or train a model first in the 'Model Loading' page.")
    else:
        st.write("Detect and flag suspicious transactions based on anomaly scores")
        
        # Threshold setting
        col1, col2 = st.columns(2)
        with col1:
            threshold_method = st.radio("Set threshold by:", ["Percentile", "Manual Value"])
        
        scores = st.session_state.scores
        labels = st.session_state.labels
        
        if threshold_method == "Percentile":
            percentile = st.slider("Percentile threshold (highest % flagged)", 1, 20, 5)
            threshold = np.percentile(scores, 100 - percentile)
        else:
            threshold = st.number_input("Anomaly score threshold", value=float(np.percentile(scores, 95)), step=0.1)
        
        # Flag fraudulent transactions
        fraud_flags = scores >= threshold
        n_flagged = np.sum(fraud_flags)
        
        st.subheader(f"Detection Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Threshold", f"{threshold:.4f}")
        with col2:
            st.metric("Flagged Transactions", int(n_flagged))
        with col3:
            pct_flagged = (n_flagged / len(scores)) * 100
            st.metric("% Flagged", f"{pct_flagged:.2f}%")
        
        # Show score distribution with threshold
        st.subheader("Score Distribution with Threshold")
        fig, ax = plt.subplots(figsize=(10, 5))
        labelled_mask = labels != -1
        licit_scores = scores[(labels == 0) & labelled_mask]
        illicit_scores = scores[(labels == 1) & labelled_mask]
        
        ax.hist(licit_scores, bins=80, alpha=0.6, color="#2196F3", label="Licit (Actual)")
        ax.hist(illicit_scores, bins=80, alpha=0.6, color="#F44336", label="Illicit (Actual)")
        ax.axvline(threshold, color="green", linestyle="--", linewidth=2.5, label=f"Detection Threshold ({threshold:.3f})")
        ax.set_xlabel("Anomaly Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Score Distribution with Detection Threshold", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show top flagged transactions
        st.subheader("Top Suspicious Transactions")
        flagged_indices = np.argsort(scores)[::-1][:min(50, n_flagged)]
        flagged_data = pd.DataFrame({
            'Transaction ID': flagged_indices,
            'Anomaly Score': scores[flagged_indices],
            'Label': ['Illicit' if labels[i] == 1 else 'Licit' if labels[i] == 0 else 'Unknown' for i in flagged_indices],
            'Flagged': [fraud_flags[i] for i in flagged_indices]
        })
        
        st.dataframe(flagged_data, use_container_width=True)
        
        # Download results
        csv = flagged_data.to_csv(index=False)
        st.download_button(
            label="Download Flagged Transactions (CSV)",
            data=csv,
            file_name="flagged_transactions.csv",
            mime="text/csv"
        )
        
        # Detection metrics based on threshold
        if np.sum(labels != -1) > 0:
            labelled_fraud_flags = fraud_flags[labelled_mask]
            y_true_labelled = labels[labelled_mask]
            
            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
            tn, fp, fn, tp = confusion_matrix(y_true_labelled, labelled_fraud_flags).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            st.subheader("Threshold-based Detection Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("True Positives", int(tp))
            with col2:
                st.metric("False Positives", int(fp))
            with col3:
                st.metric("False Negatives", int(fn))
            with col4:
                st.metric("True Negatives", int(tn))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision", f"{precision:.4f}")
            with col2:
                st.metric("Recall", f"{recall:.4f}")
            with col3:
                st.metric("F1-Score", f"{f1:.4f}")

# Page 4: Results
elif page == "Results":
    st.title("Results")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
    else:
        st.subheader("Model Performance")
        
        metrics = st.session_state.metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
        with col2:
            st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        with col3:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col4:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        
        # Visualizations
        st.subheader("Visualizations")
        
        # Create temporary directory for plots
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Score distribution
            fig1 = plot_score_distribution(st.session_state.scores, st.session_state.labels, save_path=None)
            st.pyplot(fig1)
            
            # ROC curve
            fig2 = plot_roc_curve(st.session_state.labels[st.session_state.labels != -1], 
                                st.session_state.scores[st.session_state.labels != -1], 
                                auc=metrics['ROC-AUC'], save_path=None)
            st.pyplot(fig2)
            
            # Precision-Recall curve
            fig3 = plot_precision_recall_curve(st.session_state.labels[st.session_state.labels != -1], 
                                             st.session_state.scores[st.session_state.labels != -1], 
                                             save_path=None)
            st.pyplot(fig3)

# Page 5: Prediction
elif page == "Prediction":
    st.title("Prediction")
    st.write("Make predictions on new data (placeholder - would need new data format)")
    
    st.info("This feature would allow uploading new transaction data for fraud prediction. Currently, it's a placeholder.")
    
    # Placeholder for future implementation
    st.text_area("Paste transaction features (JSON format)", height=200)
    if st.button("Predict"):
        st.info("Prediction functionality to be implemented based on your data format.")

if __name__ == "__main__":
    st.write("---")
    st.write("SpecAE Fraud Detection Dashboard")
    st.write("Built with Streamlit")