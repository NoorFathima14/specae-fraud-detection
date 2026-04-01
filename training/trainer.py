"""
training/trainer.py

Training loop and loss function for SpecAE.
"""

import torch
import torch.nn.functional as F
from models.specae import SpecAE


def loss_function(
    X: torch.Tensor,
    X_hat: torch.Tensor,
    A: torch.Tensor,
    A_hat: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combined reconstruction loss.

    Loss = alpha * MSE(X, X_hat) + (1 - alpha) * MSE(A, A_hat)

    Args:
        X:     Original node features
        X_hat: Reconstructed node features
        A:     Normalized adjacency matrix
        A_hat: Reconstructed adjacency matrix
        alpha: Weight for feature vs structure loss (default 0.5)

    Returns:
        Scalar loss tensor
    """
    loss_x = F.mse_loss(X_hat, X)
    loss_a = F.mse_loss(A_hat, A)
    return alpha * loss_x + (1 - alpha) * loss_a


def train(
    model: SpecAE,
    X: torch.Tensor,
    A_norm: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.005,
    alpha: float = 0.5,
    log_every: int = 10,
) -> list:
    """
    Train SpecAE end-to-end.

    Args:
        model:     SpecAE model instance
        X:         Node feature matrix
        A_norm:    Normalized adjacency matrix
        epochs:    Number of training epochs
        lr:        Learning rate
        alpha:     Loss weighting (feature vs structure)
        log_every: Print loss every N epochs

    Returns:
        loss_history: List of loss values per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        X_hat, A_hat, _, _ = model(X, A_norm)
        loss = loss_function(X, X_hat, A_norm, A_hat, alpha=alpha)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % log_every == 0:
            print(f"Epoch {epoch:>4d}/{epochs}  |  Loss: {loss.item():.6f}")

    print(f"\nTraining complete. Final loss: {loss_history[-1]:.6f}")
    return loss_history
