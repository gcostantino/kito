from torch import nn

from src.utils.ssim_utils import ssim_loss

_torch_loss_dict = {
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


def get_loss(loss: str):
    assert loss != ''
    loss = loss.lower()
    if loss in _torch_loss_dict:
        return getattr(nn, _torch_loss_dict[loss])()  # returns the instantiated object
    if loss == 'ssim_loss':
        return ssim_loss  # return the signature of the custom loss: /!\ this is not the object
    raise ValueError(
        f"Loss '{loss}' not valid. Supported values are: {', '.join(map(repr, _torch_loss_dict.keys()))}")
