"""
models/specae.py

Spectral Autoencoder (SpecAE) for anomaly detection in attributed networks.

Architecture:
  - Graph convolutional encoder  → dual latent spaces Z_X (attribute) and Z_G (structure)
  - Feature decoder              → reconstructs X from Z_X
  - Structure decoder            → reconstructs A from Z_G via inner product

Reference:
  Li et al. (2019). Spectral Autoencoder for Anomaly Detection in Attributed Networks.
  arXiv:1908.03849
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecAE(nn.Module):
    """
    Spectral Autoencoder for anomaly detection in attributed networks.

    Args:
        input_dim  (int): Number of node feature dimensions.
        hidden_dim (int): Hidden layer size.
        latent_dim (int): Latent space size for Z_X and Z_G.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(SpecAE, self).__init__()

        # --- Encoder ---
        # Two-layer graph convolutional encoder
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, latent_dim)

        # --- Dual latent projections ---
        # Z_X: captures global attribute information (for global anomalies)
        # Z_G: captures structural/community information (for community anomalies)
        self.zx_layer = nn.Linear(latent_dim, latent_dim)
        self.zg_layer = nn.Linear(latent_dim, latent_dim)

        # --- Feature decoder ---
        # Reconstructs node features X from Z_X
        self.decoder_x = nn.Linear(latent_dim, input_dim)

    # ------------------------------------------------------------------
    def encode(self, X: torch.Tensor, A: torch.Tensor):
        """
        Graph convolutional encoding.

        Args:
            X: Node feature matrix, shape (N, input_dim)
            A: Normalized adjacency matrix, shape (N, N)

        Returns:
            Z_X: Attribute latent space, shape (N, latent_dim)
            Z_G: Structure latent space, shape (N, latent_dim)
        """
        # Layer 1: spectral graph convolution + ReLU
        H = torch.matmul(A, X)
        H = F.relu(self.gc1(H))

        # Layer 2: spectral graph convolution (no activation — raw latent)
        H = torch.matmul(A, H)
        H = self.gc2(H)

        # Project to dual latent spaces
        Z_X = self.zx_layer(H)   # global attribute space
        Z_G = self.zg_layer(H)   # community/structure space

        return Z_X, Z_G

    # ------------------------------------------------------------------
    def decode(self, Z_X: torch.Tensor, Z_G: torch.Tensor):
        """
        Decode latent spaces back to features and adjacency.

        Args:
            Z_X: Attribute latent space, shape (N, latent_dim)
            Z_G: Structure latent space, shape (N, latent_dim)

        Returns:
            X_hat: Reconstructed features, shape (N, input_dim)
            A_hat: Reconstructed adjacency, shape (N, N)
        """
        X_hat = self.decoder_x(Z_X)                           # feature reconstruction
        A_hat = torch.sigmoid(torch.matmul(Z_G, Z_G.t()))     # structure reconstruction

        return X_hat, A_hat

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """
        Full forward pass.

        Returns:
            X_hat, A_hat, Z_X, Z_G
        """
        Z_X, Z_G = self.encode(X, A)
        X_hat, A_hat = self.decode(Z_X, Z_G)
        return X_hat, A_hat, Z_X, Z_G
