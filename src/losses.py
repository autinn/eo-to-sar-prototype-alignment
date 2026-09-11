"""Custom loss used by the unpaired MMD experiment."""

from __future__ import annotations

import torch
import torch.nn as nn


# Multi-bandwidth kernel used to compare SAR and EO distributions.
class RBFKernel(nn.Module):
    """Sum of RBF kernels using multiple bandwidths."""

    def __init__(
        self,
        num_kernels: int = 5,
        multiplier: float = 2.0,
        bandwidth_estimator: str = "median",
    ) -> None:
        super().__init__()
        if bandwidth_estimator not in {"mean", "median"}:
            raise ValueError("bandwidth_estimator must be 'mean' or 'median'.")

        offsets = torch.arange(num_kernels) - num_kernels // 2
        self.register_buffer("bandwidth_multipliers", multiplier**offsets)
        self.bandwidth_estimator = bandwidth_estimator

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        squared_distances = torch.cdist(features, features).square()

        if self.bandwidth_estimator == "median":
            bandwidth = squared_distances.median()
        else:
            sample_count = squared_distances.shape[0]
            denominator = sample_count**2 - sample_count
            bandwidth = squared_distances.sum() / max(denominator, 1)

        bandwidth = bandwidth.clamp_min(torch.finfo(features.dtype).eps)
        bandwidths = bandwidth * self.bandwidth_multipliers.to(features.dtype)
        return torch.exp(
            -squared_distances.unsqueeze(0) / bandwidths[:, None, None]
        ).sum(dim=0)


# Distribution alignment objective for the unpaired MMD baseline.
class MMDLoss(nn.Module):
    """Empirical estimate of squared MMD (maximum mean discrepancy) used in the paper."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = RBFKernel(bandwidth_estimator="median")

    def forward(
        self,
        sar_features: torch.Tensor,
        eo_features: torch.Tensor,
    ) -> torch.Tensor:
        sar_count = sar_features.shape[0]
        kernel_matrix = self.kernel(torch.cat([sar_features, eo_features], dim=0))

        sar_sar = kernel_matrix[:sar_count, :sar_count].mean()
        sar_eo = kernel_matrix[:sar_count, sar_count:].mean()
        eo_eo = kernel_matrix[sar_count:, sar_count:].mean()
        return sar_sar - 2 * sar_eo + eo_eo
