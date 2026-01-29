"""
Custom domain-specific losses.

Add your own losses here and register them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import LossRegistry


@LossRegistry.register('ssim')
class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss for image quality.

    Registered name: 'ssim'

    Args:
        window_size: Size of sliding window (default: 11)
        C1, C2: Stability constants

    Example:
        loss = LossRegistry.create('ssim', window_size=7)
    """

    def __init__(self, window_size: int = 11, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Simplified SSIM - replace with proper implementation
        # or use torchmetrics.functional.structural_similarity_index_measure
        mse = F.mse_loss(pred, target, reduction='none')
        ssim_value = torch.exp(-mse)
        return 1.0 - ssim_value.mean()


# Example: Add your InSAR-specific losses here!
@LossRegistry.register('insar_phase')
class InSARPhaseLoss(nn.Module):
    """
    Custom loss for InSAR phase denoising.

    Combines MSE with coherence weighting and edge preservation.

    Args:
        coherence_threshold: Threshold for coherence masking (default: 0.5)
        spatial_weight: Enable spatial smoothness term (default: False)
        edge_weight: Weight for edge preservation (default: 1.0)

    Example:
        loss = LossRegistry.create('insar_phase',
                                     coherence_threshold=0.6,
                                     spatial_weight=True)
    """

    def __init__(self,
                 coherence_threshold: float = 0.5,
                 spatial_weight: bool = False,
                 edge_weight: float = 1.0):
        super().__init__()
        self.coherence_threshold = coherence_threshold
        self.spatial_weight = spatial_weight
        self.edge_weight = edge_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                coherence: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted phase
            target: Target phase
            coherence: Optional coherence map
        """
        # Base MSE loss
        mse_loss = F.mse_loss(pred, target)

        # Coherence weighting (if provided)
        if coherence is not None:
            mask = coherence > self.coherence_threshold
            coherence_loss = (((pred - target) ** 2) * mask).mean()
            mse_loss = 0.7 * mse_loss + 0.3 * coherence_loss

        # Spatial smoothness (Total Variation)
        if self.spatial_weight:
            tv_loss = self._total_variation(pred)
            mse_loss = mse_loss + 0.1 * tv_loss

        # Edge preservation
        if self.edge_weight > 0:
            edge_loss = self._edge_loss(pred, target)
            mse_loss = mse_loss + self.edge_weight * edge_loss

        return mse_loss

    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation for spatial smoothness."""
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return tv_h + tv_w

    def _edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Edge-aware loss using Sobel filters."""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        # Compute edges
        pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)

        # Edge loss
        edge_loss = (((pred_edges_x - target_edges_x) ** 2) +
                     ((pred_edges_y - target_edges_y) ** 2)).mean()

        return edge_loss
