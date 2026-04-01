"""
preprocessing/graph_utils.py

Graph utilities: adjacency normalization and construction helpers.
"""

import torch
import numpy as np


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetrically normalize adjacency matrix with self-loops.

    Computes: D^{-1/2} * (A + I) * D^{-1/2}
    as described in SpecAE / Kipf & Welling (2017).

    Args:
        A: Raw adjacency matrix, shape (N, N). numpy array or torch tensor.

    Returns:
        A_norm: Normalized adjacency matrix, shape (N, N).
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)

    I = torch.eye(A.size(0), device=A.device)
    A_hat = A + I                                         # Add self-loops

    D = torch.diag(A_hat.sum(dim=1))                     # Degree matrix
    D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))

    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


def edge_index_to_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Convert PyG edge_index format to a dense adjacency matrix.

    Args:
        edge_index: shape (2, E)
        num_nodes:  number of nodes N

    Returns:
        A: dense adjacency matrix, shape (N, N)
    """
    A = torch.zeros((num_nodes, num_nodes))
    for i, j in edge_index.t():
        A[i, j] = 1.0

    # Make undirected
    A = (A + A.t())
    A[A > 0] = 1.0
    return A
