import torch
import torch.nn.functional as F


def gaussian_kernel(window_size: int, sigma: float, device, dtype):
    """Create a 2D Gaussian kernel (window_size×window_size)."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # outer product to get 2D kernel
    gauss2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return gauss2d


def ssim_2d_single(img1, img2, window_size=11, sigma=1.5, K1=0.01, K2=0.03, L=1.0):
    """
    Compute SSIM over a single 2D windowed image (float range [0, L]).
    img1, img2: shape (1, 1, H, W)  (batch=1, channel=1)
    Returns mean SSIM over all pixels in that 2D image.
    """
    # Create Gaussian window
    device, dtype = img1.device, img1.dtype
    kernel = gaussian_kernel(window_size, sigma, device, dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,win,win)

    # Convolve to get local means
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # local variances: E[x^2] - (E[x])^2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2) - mu1_mu2

    sigma1_sq = torch.clamp(sigma1_sq, min=1e-6)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-6)

    # SSIM constants
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
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


def batch_time_ssim(pred, target, window_size=11, sigma=1.5):
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
            ssim_scores[b, t] = ssim_2d_single(im1, im2, window_size, sigma)
    return ssim_scores  # shape (B, T)


def minmax_normalize_sample(x):
    # x: (T, H, W, 1)
    mn = x.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    mx = x.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    denom = (mx - mn).clamp(min=1e-6)
    return (x - mn) / denom

def ssim_loss(target, predicted):
    no_disp_mask = (target.sum(dim=tuple(range(1, target.dim()))) == 0)
    valid_regress_targets = target[~no_disp_mask]
    valid_regress_outputs = predicted[~no_disp_mask]
    B = len(valid_regress_targets)
    pred = torch.stack([minmax_normalize_sample(valid_regress_outputs[b]) for b in range(B)], dim=0)
    target = torch.stack([minmax_normalize_sample(valid_regress_targets[b]) for b in range(B)], dim=0)

    ssim_bt = batch_time_ssim(pred, target, window_size=11, sigma=1.5)  # shape (B, T)
    ssim_per_sample = ssim_bt.mean(dim=1)  # average SSIM over time for each batch
    loss_ssim = (1.0 - ssim_per_sample).mean()
    return loss_ssim