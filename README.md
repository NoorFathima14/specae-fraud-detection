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

## Team

| Member | Responsibilities |
|--------|-----------------|
| Member 1 (Noor) | Model architecture, optimization, innovation, loss engineering |
| Member 2 | Elliptic data pipeline, evaluation, visualization |

---

## Reference

Li, Y., Huang, X., Li, J., Du, M., & Zou, N. (2019). Spectral Autoencoder for Anomaly Detection in Attributed Networks. *arXiv:1908.03849*
