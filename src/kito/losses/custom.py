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

    def __init__(self, window_size: int = 11, sigma: float = 1.5, K1: float = 0.01, K2: float = 0.03, L: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.K1 = K1
        self.K2 = K2
        self.L = L

    def _gaussian_kernel(self, device, dtype):
        """Create a 2D Gaussian kernel (window_size×window_size)."""
        coords = torch.arange(self.window_size, device=device, dtype=dtype) - (self.window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        # outer product to get 2D kernel
        gauss2d = g.unsqueeze(1) @ g.unsqueeze(0)
        return gauss2d

    def _ssim_2d_single(self, img1, img2):
        """
        Compute SSIM over a single 2D windowed image (float range [0, L]).
        img1, img2: shape (1, 1, H, W)  (batch=1, channel=1)
        Returns mean SSIM over all pixels in that 2D image.
        """
        # Create Gaussian window
        device, dtype = img1.device, img1.dtype
        kernel = self._gaussian_kernel(device, dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,win,win)

        # Convolve to get local means
        mu1 = F.conv2d(img1, kernel, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, kernel, padding=self.window_size // 2)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        # local variances: E[x^2] - (E[x])^2
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=self.window_size // 2) - mu1_mu2

        sigma1_sq = torch.clamp(sigma1_sq, min=1e-6)
        sigma2_sq = torch.clamp(sigma2_sq, min=1e-6)

        # SSIM constants
        C1 = (self.K1 * self.L) ** 2
        C2 = (self.K2 * self.L) ** 2
        C3 = C2 / 2.0

        # Luminance term
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        # Contrast term
        contrast = (2 * torch.sqrt(sigma1_sq * sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
        # Structure term
        structure = (sigma12 + C3) / (torch.sqrt(sigma1_sq * sigma2_sq) + C3)

        # Full SSIM map
        ssim_map = luminance * contrast * structure
        return ssim_map.mean()

    def _minmax_normalize_sample(self, x):
        # x: (T, H, W, 1)
        mn = x.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        mx = x.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        denom = (mx - mn).clamp(min=1e-6)
        return (x - mn) / denom

    def _batch_time_ssim(self, pred, target):
        """
        pred, target: tensors of shape (B, T, H, W, 1), normalized to [0,1].
        Returns SSIM per (batch, time) as a tensor (B, T), and also a mean per-batch if desired.
        """
        B, T, H, W, C = pred.shape
        assert C == 1, "SSIM code below assumes single channel"
        ssim_scores = torch.zeros((B, T), device=pred.device, dtype=pred.dtype)

        for b in range(B):
            for t in range(T):
                im1 = target[b, t, ..., :].permute(2, 0, 1).unsqueeze(0)  # shape (1,1,H,W)
                im2 = pred[b, t, ..., :].permute(2, 0, 1).unsqueeze(0)  # shape (1,1,H,W)
                ssim_scores[b, t] = self._ssim_2d_single(im1, im2)
        return ssim_scores  # shape (B, T)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        no_disp_mask = (target.sum(dim=tuple(range(1, target.dim()))) == 0)
        valid_regress_targets = target[~no_disp_mask]
        valid_regress_outputs = predicted[~no_disp_mask]
        B = len(valid_regress_targets)
        pred = torch.stack([self._minmax_normalize_sample(valid_regress_outputs[b]) for b in range(B)], dim=0)
        target = torch.stack([self._minmax_normalize_sample(valid_regress_targets[b]) for b in range(B)], dim=0)

        ssim_bt = self._batch_time_ssim(pred, target)  # shape (B, T)
        ssim_per_sample = ssim_bt.mean(dim=1)  # average SSIM over time for each batch
        loss_ssim = (1.0 - ssim_per_sample).mean()
        return loss_ssim
