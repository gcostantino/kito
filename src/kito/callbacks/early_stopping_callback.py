from kito.callbacks.callback_base import Callback


class EarlyStoppingCallback(Callback):
    """
    Stop training when monitored metric stops improving.

    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement before stopping
        mode: 'min' or 'max'
    """

    def __init__(self, monitor='val_loss', patience=10, mode='min', min_delta=0.):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.min_delta = min_delta
        self._warned_missing = False

    def on_train_begin(self, engine, model, **kwargs):
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self._warned_missing = False

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Check if training should stop."""
        if logs is None:
            return
        current = logs.get(self.monitor)
        if current is None:
            if not self._warned_missing:
                engine.logger.log_warning(
                    f"EarlyStopping: '{self.monitor}' not in logs {list(logs)}; it will never trigger."
                )
                self._warned_missing = True
            return

        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta

        if improved:
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                engine.stop_training = True  # engine broadcasts this to all ranks
                engine.logger.log_info(f"Early stopping at epoch {epoch}")
