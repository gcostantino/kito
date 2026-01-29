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
        raise NotImplementedError('SSIMLoss is not yet implemented.')
