import torch.distributed as dist

from kito.callbacks.callback_base import Callback


class DDPAwareCallback(Callback):
    """
    Wrapper that makes any callback DDP-safe.

    Only executes on the given rank (default: 0) in distributed training.

    Args:
        callback: The callback to wrap

    Example:
        checkpoint = ModelCheckpoint('weights/best.pt')
        ddp_checkpoint = DDPAwareCallback(checkpoint)
    """

    def __init__(self, callback: Callback, master_rank=0):
        self.callback = callback
        self.master_rank = master_rank
        self.is_driver = self._check_if_driver()

    def _check_if_driver(self):
        """Check if this is the driver process (rank 0)."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == self.master_rank
        return True  # Single GPU or CPU

    def on_train_begin(self, engine, model, **kwargs):
        if self.is_driver:
            self.callback.on_train_begin(engine, model, **kwargs)

    def on_train_end(self, engine, model, **kwargs):
        if self.is_driver:
            self.callback.on_train_end(engine, model, **kwargs)

    def on_epoch_begin(self, epoch, engine, model, **kwargs):
        if self.is_driver:
            self.callback.on_epoch_begin(epoch, engine, model, **kwargs)

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        if self.is_driver:
            self.callback.on_epoch_end(epoch, engine, model, logs, **kwargs)

    def on_train_batch_begin(self, batch, engine, model, **kwargs):
        if self.is_driver:
            self.callback.on_train_batch_begin(batch, engine, model, **kwargs)

    def on_train_batch_end(self, batch, engine, model, logs=None, **kwargs):
        if self.is_driver:
            self.callback.on_train_batch_end(batch, engine, model, logs, **kwargs)

    def on_validation_begin(self, epoch, engine, model, **kwargs):
        if self.is_driver:
            self.callback.on_validation_begin(epoch, engine, model, **kwargs)

    def on_validation_end(self, epoch, engine, model, logs=None, **kwargs):
        if self.is_driver:
            self.callback.on_validation_end(epoch, engine, model, logs, **kwargs)
