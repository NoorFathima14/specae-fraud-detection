# SpecAE — Spectral Autoencoder for Anomaly Detection in Attributed Networks

Implementation of **SpecAE** from the paper:
> *"Spectral Autoencoder for Anomaly Detection in Attributed Networks"* — Li et al., 2019

Applied to **fraud detection** using the Elliptic Bitcoin dataset.

---

## Project Structure

```
specae-fraud-detection/
├── models/
│   └── specae.py           # SpecAE model architecture
├── preprocessing/
│   └── graph_utils.py      # Adjacency normalization & graph utilities
├── training/
│   └── trainer.py          # Training loop & loss function
├── evaluation/
│   └── anomaly_score.py    # Anomaly scoring & metrics
├── scripts/
│   ├── run_cora.py         # Phase 1: Reproduce paper on Cora
│   └── run_elliptic.py     # Phase 2: Fraud detection on Elliptic
├── data/                   # Datasets (auto-downloaded or placed here)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Phase 1 — Reproduce on Cora (Validation)

```bash
python scripts/run_cora.py
```

---

## Phase 2 — Fraud Detection on Elliptic

```bash
python scripts/run_elliptic.py
```

---

## Streamlit Dashboard

Interactive web application for fraud detection:

```bash
streamlit run dashboard.py
```

### Features
- **Data Overview**: Explore Elliptic dataset statistics
- **Model Setup**: Train new models or load pre-trained ones
- **Fraud Detection**: Set thresholds and flag suspicious transactions
- **Results**: View performance metrics and visualizations

---

## Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub** (already done)
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Select repository**: `NoorFathima14/specae-fraud-detection`
5. **Set main file path**: `dashboard.py`
6. **Click Deploy**

### Other Options

#### Heroku
```bash
# Create requirements.txt (already done)
# Create Procfile
echo "web: streamlit run dashboard.py --server.port $PORT --server.headless true" > Procfile
# Deploy via Heroku CLI or GitHub integration
```

#### Local Deployment
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## Team

| Member | Responsibilities |
|--------|-----------------|
| Member 1 (Noor) | Model architecture, optimization, innovation, loss engineering |
| Member 2 | Elliptic data pipeline, evaluation, visualization |

---

## Reference

Li, Y., Huang, X., Li, J., Du, M., & Zou, N. (2019). Spectral Autoencoder for Anomaly Detection in Attributed Networks. *arXiv:1908.03849*
