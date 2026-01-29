"""
Built-in PyTorch loss wrappers with backward compatibility.

Automatically registers losses with:
- Short names: 'mse', 'l1', etc.
- PyTorch names: 'MSELoss', 'L1Loss', etc.
- Verbose names: 'mean_squared_error', 'mean_absolute_error', etc. (from old code)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import LossRegistry


# ============================================================================
# Mapping from verbose names to PyTorch class names (from your old code)
# ============================================================================

_TORCH_LOSS_MAPPING = {
    # Verbose name -> PyTorch class name
    'mean_squared_error': 'MSELoss',
    'mean_absolute_error': 'L1Loss',
    'cross_entropy_loss': 'CrossEntropyLoss',
    'ctc_loss': 'CTCLoss',
    'negative_log_likelihood_loss': 'NLLLoss',
    'negative_log_likelihood_poisson_loss': 'PoissonNLLLoss',
    'negative_log_likelihood_gaussian_loss': 'GaussianNLLLoss',
    'kullback_leibler_divergence_loss': 'KLDivLoss',
    'binary_cross_entropy_loss': 'BCELoss',
    'binary_cross_entropy_logits_loss': 'BCEWithLogitsLoss',
    'margin_ranking_loss': 'MarginRankingLoss',
    'hinge_embedding_loss': 'HingeEmbeddingLoss',
    'multi_label_margin_loss': 'MultiLabelMarginLoss',
    'huber_loss': 'HuberLoss',
    'smooth_l1_loss': 'SmoothL1Loss',
    'soft_margin_loss': 'SoftMarginLoss',
    'multi_label_soft_margin_loss': 'MultiLabelSoftMarginLoss',
    'cosine_embedding_loss': 'CosineEmbeddingLoss',
    'multi_margin_loss': 'MultiMarginLoss',
    'triplet_margin_loss': 'TripletMarginLoss',
    'triplet_margin_distance_loss': 'TripletMarginWithDistanceLoss'
}


# ============================================================================
# Auto-Registration Helper
# ============================================================================

def _auto_register_pytorch_losses():
    """
    Automatically register PyTorch losses with multiple aliases.

    Each loss gets registered with:
    1. PyTorch class name: 'MSELoss'
    2. Short name: 'mse'
    3. Verbose name: 'mean_squared_error' (if in mapping)
    """
    # Define short names for common losses
    SHORT_NAMES = {
        'MSELoss': 'mse',
        'L1Loss': ['l1', 'mae'],
        'CrossEntropyLoss': 'cross_entropy',
        'NLLLoss': 'nll',
        'BCELoss': 'bce',
        'BCEWithLogitsLoss': 'bce_with_logits',
        'KLDivLoss': 'kl_div',
        'HuberLoss': 'huber',
        'SmoothL1Loss': 'smooth_l1',
        'PoissonNLLLoss': 'poisson_nll',
        'GaussianNLLLoss': 'gaussian_nll',
        'CTCLoss': 'ctc',
        'MarginRankingLoss': 'margin_ranking',
        'HingeEmbeddingLoss': 'hinge_embedding',
        'MultiLabelMarginLoss': 'multi_label_margin',
        'SoftMarginLoss': 'soft_margin',
        'MultiLabelSoftMarginLoss': 'multi_label_soft_margin',
        'CosineEmbeddingLoss': 'cosine_embedding',
        'MultiMarginLoss': 'multi_margin',
        'TripletMarginLoss': 'triplet_margin',
        'TripletMarginWithDistanceLoss': 'triplet_margin_distance',
    }

    # Get unique PyTorch class names from the mapping
    pytorch_classes = set(_TORCH_LOSS_MAPPING.values())

    for pytorch_class_name in pytorch_classes:
        # Get the actual PyTorch class
        if not hasattr(nn, pytorch_class_name):
            continue  # Skip if not available in current PyTorch version

        pytorch_class = getattr(nn, pytorch_class_name)

        # Create a wrapper class
        class_name = f"_{pytorch_class_name}Wrapper"
        wrapper_class = type(class_name, (pytorch_class,), {
            '__doc__': f"{pytorch_class_name} wrapper for registry.\n\n{pytorch_class.__doc__}"
        })

        # Collect all names for this loss
        names_to_register = set()

        # 1. Add PyTorch class name
        names_to_register.add(pytorch_class_name)

        # 2. Add short name(s)
        if pytorch_class_name in SHORT_NAMES:
            short = SHORT_NAMES[pytorch_class_name]
            if isinstance(short, list):
                names_to_register.update(short)
            else:
                names_to_register.add(short)

        # 3. Add verbose names from old mapping
        for verbose_name, class_name in _TORCH_LOSS_MAPPING.items():
            if class_name == pytorch_class_name:
                names_to_register.add(verbose_name)

        # Register with all names
        for name in names_to_register:
            try:
                LossRegistry.register(name)(wrapper_class)
            except ValueError:
                # Already registered (e.g., if short name = verbose name)
                pass


# Auto-register all PyTorch losses
_auto_register_pytorch_losses()


# ============================================================================
# Common Custom Losses (manually registered for clarity)
# ============================================================================

@LossRegistry.register('weighted_mse')
class WeightedMSELoss(nn.Module):
    """
    MSE loss with global weight multiplier.

    Registered name: 'weighted_mse'

    Args:
        weight: Global weight multiplier (default: 1.0)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')

    Example:
        loss = LossRegistry.create('weighted_mse', weight=2.0)
    """

    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return self.weight * loss


@LossRegistry.register('combined')
class CombinedLoss(nn.Module):
    """
    Combine multiple losses with weights.

    Registered name: 'combined'

    Args:
        losses: List of dicts with 'name', 'weight', and optional 'params'

    Example:
        loss = LossRegistry.create('combined', losses=[
            {'name': 'mse', 'weight': 1.0},
            {'name': 'l1', 'weight': 0.5}
        ])
    """

    def __init__(self, losses: list):
        super().__init__()
        self.losses = nn.ModuleList()
        self.weights = []

        for loss_config in losses:
            name = loss_config['name']
            weight = loss_config.get('weight', 1.0)
            params = loss_config.get('params', {})

            # Create loss from registry
            loss_fn = LossRegistry.create(name, **params)
            self.losses.append(loss_fn)
            self.weights.append(weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss
