import warnings

from kito.losses import get_loss as _get_loss


def get_loss(loss: str):
    warnings.warn("kito.utils.loss_utils.get_loss is deprecated; use kito.losses.get_loss",
                  DeprecationWarning, stacklevel=2)
    if loss == '':
        raise ValueError("loss name must be non-empty")
    return _get_loss(loss)
