"""Input data standardization layers."""

import numpy as np
import torch
import torch.nn as nn


class ZScoreNorm(nn.Module):
    """
    Per-feature z-score normalisation layer.

    For each feature, computes the sample mean and standard deviation from
    a calibration set.  At forward time the raw input is whitened
    feature-wise: z = (x - mean) / std.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input.
    eps : float
        Small constant added to the standard deviation for numerical
        stability.
    """

    def __init__(self, n_features: int, eps: float = 1e-8):
        """
        Initialise the layer and register per-feature statistic buffers.

        Parameters
        ----------
        n_features : int
            Dimensionality of the input.
        eps : float
            Small constant added to the standard deviation for numerical
            stability.
        """
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.fitted = False

        self.register_buffer("mean_t", torch.zeros(n_features))
        self.register_buffer("std_t", torch.ones(n_features))

    def fit(self, x_np: np.ndarray) -> None:
        """
        Compute per-feature mean and std from a numpy array.

        Parameters
        ----------
        x_np : np.ndarray
            Array of shape (N, n_features) used to compute the statistics.

        Returns
        -------
        None
        """
        N, D = x_np.shape
        assert D == self.n_features, (
            f"Expected {self.n_features} features, got {D}."
        )

        x = x_np.astype(np.float64)
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std = np.where(std > self.eps, std, 1.0).astype(np.float32)

        self.mean_t.copy_(torch.from_numpy(mean))
        self.std_t.copy_(torch.from_numpy(std))
        self.fitted = True

        print(
            f"[ZScoreNorm] fitted {D} features from {N} samples.\n"
            f"  mean : min={mean.min():.6f}  max={mean.max():.6f}\n"
            f"  std  : min={std.min():.6f}   max={std.max():.6f}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply per-feature z-score normalisation to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., n_features).

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``, z-score normalised per
            feature.  If the layer has not been fitted, ``x`` is returned
            unchanged.
        """
        if not self.fitted:
            return x
        return (x - self.mean_t) / self.std_t
