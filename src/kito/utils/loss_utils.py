from torch import nn

from kito.losses.builtin import _TORCH_LOSS_MAPPING
from kito.utils.ssim_utils import ssim_loss


def get_loss(loss: str):
    assert loss != ''
    loss = loss.lower()
    if loss in _TORCH_LOSS_MAPPING:
        return getattr(nn, _TORCH_LOSS_MAPPING[loss])()  # returns the instantiated object
    if loss == 'ssim_loss':
        return ssim_loss  # return the signature of the custom loss: /!\ this is not the object
    raise ValueError(
        f"Loss '{loss}' not valid. Supported values are: {', '.join(map(repr, _torch_loss_dict.keys()))}")
