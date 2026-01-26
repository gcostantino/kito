import os
import torch

from kito.callbacks.callback_base import Callback


class ModelCheckpoint(Callback):
    """
    Save model weights during training.

    Args:
        filepath: Path template for saving weights (can include {epoch}, {val_loss}, etc.)
        monitor: Metric to monitor (e.g., 'val_loss')
        save_best_only: Only save when monitored metric improves
        mode: 'min' or 'max' depending on whether lower/higher is better
        verbose: Print message when saving

    Example:
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
            verbose: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose

        # Track best metric
        self.best = float('inf') if mode == 'min' else float('-inf')

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Save model if metric improved."""
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        # Check if improved
        improved = (
                (self.mode == 'min' and current < self.best) or
                (self.mode == 'max' and current > self.best)
        )

        if improved or not self.save_best_only:
            self.best = current

            # Format filepath
            filepath = self.filepath.format(epoch=epoch, **logs)

            # Save model (handle DDP)
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, filepath)

            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved to {current:.4f}, "
                      f"saving model to {filepath}")
