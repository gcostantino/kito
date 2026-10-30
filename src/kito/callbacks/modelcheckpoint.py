import os

import torch

from kito.callbacks.callback_base import Callback
from kito.utils.model_utils import unwrap_model


class ModelCheckpoint(Callback):
    """
    Save model weights during training.

    Args:
        filepath: Path template for saving weights (can include {epoch}, {val_loss}, etc.)
        monitor: Metric to monitor (e.g., 'val_loss')
        save_best_only: Only save when monitored metric improves
        mode: 'min' or 'max' depending on whether lower/higher is better
        verbose: Print message when saving

    Example::

        checkpoint = ModelCheckpoint(
            filepath='weights/model_epoch{epoch:02d}_valloss{val_loss:.4f}.pt',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    """

    def __init__(
            self,
            filepath: str,
            monitor: str = 'val_loss',
            save_best_only: bool = True,
            mode: str = 'min',
            verbose: bool = False
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self._warned_missing = False

        # Track best metric
        self.best = float('inf') if mode == 'min' else float('-inf')

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    def on_train_begin(self, engine, model, **kwargs):
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self._warned_missing = False

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Save model if metric improved."""
        if logs is None:
            return
        current = logs.get(self.monitor)
        if current is None:
            if not self._warned_missing:
                engine.logger.log_warning(
                    f"ModelCheckpoint: '{self.monitor}' not in logs {list(logs)}; nothing will be saved."
                )
                self._warned_missing = True
            return

        improved = (self.mode == 'min' and current < self.best) or \
                   (self.mode == 'max' and current > self.best)
        if improved or not self.save_best_only:
            self.best = current
            filepath = self.filepath.format(epoch=epoch, **logs)
            tmp = f"{filepath}.tmp"
            torch.save(unwrap_model(model).state_dict(), tmp)
            os.replace(tmp, filepath)  # atomic: no half-written checkpoint
            if self.verbose:
                engine.logger.log_info(
                    f"Epoch {epoch}: {self.monitor} improved to {current:.4f}, saved to {filepath}"
                )
