from kito.callbacks.callback_base import Callback


class EarlyStoppingCallback(Callback):
    """
    Stop training when monitored metric stops improving.

    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement before stopping
        mode: 'min' or 'max'
    """

    def __init__(self, monitor='val_loss', patience=10, mode='min'):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Check if training should stop."""
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

        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                engine.stop_training = True  # Requires Engine to support this
                print(f"\nEarly stopping at epoch {epoch}")
